package main

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
)

// ---- learning-to-rank MLP ----
//
// Instead of hand-tuning (or per-query heuristically adapting) the fusion
// weights of the hand-crafted similarities, a small multi-layer perceptron is
// trained to predict, for a (query, ref) pair, how likely the ref is the true
// source of the query. The input is the 8 per-pair similarities (the 7 from
// scores() plus the rotation-aligned mask score) augmented with 16 per-query
// distribution statistics (best / second-best gap per feature), so the model
// can condition on query difficulty the way compositeAdaptive does by hand.

const (
	nnH1 = 96
	nnH2 = 48
	nnSE = 24 // hidden-layer SE bottleneck width
	nnIP = 12 // input (pair-feature) SE bottleneck width
)

// Input dimension. Default: 93 = 31 pair features + 2*31 query statistics.
// With NN_NOGM=1 the hand-coded gate/mono inductive-bias dims (pair-feature
// indices 8,9) are dropped, so the pair block shrinks to 29 and the input to
// 87 = 29 + 2*29. With ATTN set, 8 attention-localization features are appended
// to the pair block (pair 39, input 117). gob weights are dimension-specific;
// keep NN_NOGM / ATTN consistent between training and evaluation.
var (
	nnInput   = 93 // 31 pair features + 2*31 query statistics
	nnP       = 31 // pair-feature block at the head of the input (indices 0..30)
	noGm      = false
	attnFeats = false // attention conditioning features enabled (ATTN set)
)

func init() {
	if os.Getenv("NN_NOGM") == "1" {
		noGm = true
		nnP = 29
	}
	if os.Getenv("ATTN") != "" {
		attnFeats = true
		nnP += attnNStat
	}
	nnInput = 3 * nnP
}

// MLP is a 3-layer feed-forward network with ReLU hidden units and a sigmoid
// output (relevance in [0,1]), plus an optional attention mechanism selected by
// the NN_ATN env var at construction time:
//
//	none (NN_ATN=none)  : plain MLP, ~12.5k params
//	hidden (default)    : Squeeze-and-Excitation channel attention on the first
//	                      hidden layer (WSE1/WSE2)
//	input (NN_ATN=input): SE-style attention on the 27 pair-feature positions
//	      (WA1/WA2), letting the model gate which expert
//	      (hist / polar shape / sczl / ...) to trust per sample
//
// The default is the plain MLP (NN_ATN=none): experiments on the held-out
// test_set showed neither attention variant improves on the plain network
// (see NN_PLAN.md §7).
// Hidden attention:
//
//	h1  = ReLU(W1 x + b1)
//	g   = sigmoid(WSE2 · ReLU(WSE1 h1 + bSE1) + bSE2)   // per-unit gate
//	h1  = h1 ⊙ g
//
// Input attention:
//
//	g  = sigmoid(WA2 · ReLU(WA1 x[0:27] + bA1) + bA2)   // per-feature gate
//	x[0:27] = x[0:27] ⊙ g
//
// Either attention lets the model re-weight features per sample (soft feature
// attention / soft routing between the hand-crafted experts). Weights with no
// attention fields (WSE1 == nil && WA1 == nil) degrade to the plain MLP, so
// old weights keep working unchanged.
type MLP struct {
	W1, B1 []float64 // H1 x input  (row-major: W1[j*nnInput+i])
	W2, B2 []float64 // H2 x H1
	W3, B3 []float64 // 1  x H2
	WSE1   []float64 // nnSE x H1 (hidden attention down-projection)
	BSE1   []float64 // nnSE
	WSE2   []float64 // H1 x nnSE (hidden attention up-projection)
	BSE2   []float64 // H1
	WA1    []float64 // nnIP x nnP (input attention down-projection)
	BA1    []float64 // nnIP
	WA2    []float64 // nnP x nnIP (input attention up-projection)
	BA2    []float64 // nnP

	// dropout regularization (training only). dropKeep is the keep-probability
	// (1.0 = disabled); dropMask is the per-h1 Bernoulli mask drawn on the loss
	// forward pass and reused by the matching backprop (reuse=true skips a
	// fresh draw so the gradient matches the activations the loss saw).
	dropKeep float64
	dropMask []float64
	reuse    bool
	train    bool
}

func newMLP(seed int64) *MLP {
	rng := rand.New(rand.NewSource(seed))
	const scale = 0.6
	m := &MLP{
		W1: randVec(rng, nnH1*nnInput, scale),
		B1: make([]float64, nnH1),
		W2: randVec(rng, nnH2*nnH1, scale),
		B2: make([]float64, nnH2),
		W3: randVec(rng, nnH2, scale),
		B3: make([]float64, 1),
	}
	// Attention weights are drawn from a separate RNG stream so the base
	// weights' init is bit-identical across all NN_ATN modes (clean ablation).
	switch atnMode() {
	case "input":
		r2 := rand.New(rand.NewSource(seed + 2))
		m.WA1 = randVec(r2, nnIP*nnP, scale)
		m.BA1 = make([]float64, nnIP)
		m.WA2 = randVec(r2, nnP*nnIP, scale)
		m.BA2 = make([]float64, nnP)
	case "none":
		// plain MLP
	default: // hidden
		r2 := rand.New(rand.NewSource(seed + 1))
		m.WSE1 = randVec(r2, nnSE*nnH1, scale)
		m.BSE1 = make([]float64, nnSE)
		m.WSE2 = randVec(r2, nnH1*nnSE, scale)
		m.BSE2 = make([]float64, nnH1)
	}
	return m
}

// atnMode returns the attention mechanism to build at construction time:
// NN_ATN=none (default, the proven best), NN_ATN=hidden, NN_ATN=input.
func atnMode() string {
	m := os.Getenv("NN_ATN")
	if m != "" {
		return m
	}
	return "none"
}

func randVec(rng *rand.Rand, n int, scale float64) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = (rng.Float64()*2 - 1) * scale
	}
	return out
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// forward runs the network and returns the output probability plus the
// activations needed for backprop:
//
//	xIn  — the effective input the first layer sees (pair block gated, if any)
//	h1   — first hidden layer (pre hidden-attention gate, ReLU)
//	gate — hidden-attention per-unit gate (1 when not used)
//	vh   — hidden-attention bottleneck activations (zeros when not used)
//	gIn  — input-attention per-feature gate (nil when not used)
//	vIn  — input-attention bottleneck activations (nil when not used)
//	h2   — second hidden layer (ReLU)
func (m *MLP) forward(x []float64) (float64, []float64, []float64, []float64, []float64, []float64, []float64, []float64) {
	xIn := x
	var gIn, vIn []float64
	if m.WA1 != nil {
		xIn = make([]float64, nnInput)
		copy(xIn, x)
		gIn = make([]float64, nnP)
		vIn = make([]float64, nnIP)
		// squeeze: pair block (27) -> bottleneck (12)
		for k := 0; k < nnIP; k++ {
			acc := m.BA1[k]
			row := m.WA1[k*nnP : (k+1)*nnP]
			for j := 0; j < nnP; j++ {
				acc += row[j] * x[j]
			}
			vIn[k] = relu(acc)
		}
		// excitation: bottleneck (12) -> per-feature gate (27)
		for i := 0; i < nnP; i++ {
			acc := m.BA2[i]
			row := m.WA2[i*nnIP : (i+1)*nnIP]
			for k := 0; k < nnIP; k++ {
				acc += row[k] * vIn[k]
			}
			gIn[i] = sigmoid(acc)
			xIn[i] = x[i] * gIn[i]
		}
	}

	h1 := make([]float64, nnH1)
	for j := 0; j < nnH1; j++ {
		acc := m.B1[j]
		row := m.W1[j*nnInput : (j+1)*nnInput]
		for i := 0; i < nnInput; i++ {
			acc += xIn[i] * row[i]
		}
		h1[j] = relu(acc)
	}

	// dropout on h1 (training only): invert mask drawn once per sample (the loss
	// forward) and reused by the backprop forward (reuse=true).
	if m.train && m.dropKeep < 1.0 {
		if m.dropMask == nil {
			m.dropMask = make([]float64, nnH1)
		}
		if !m.reuse {
			for j := 0; j < nnH1; j++ {
				if rand.Float64() < m.dropKeep {
					m.dropMask[j] = 1
				} else {
					m.dropMask[j] = 0
				}
			}
		}
		for j := 0; j < nnH1; j++ {
			h1[j] *= m.dropMask[j] / m.dropKeep
		}
	}

	gate := make([]float64, nnH1)
	vh := make([]float64, nnSE)
	if m.WSE1 != nil {
		// squeeze: h1 (96) -> bottleneck (24)
		for k := 0; k < nnSE; k++ {
			acc := m.BSE1[k]
			row := m.WSE1[k*nnH1 : (k+1)*nnH1]
			for j := 0; j < nnH1; j++ {
				acc += row[j] * h1[j]
			}
			vh[k] = relu(acc)
		}
		// excitation: bottleneck (24) -> per-unit gate (96)
		for i := 0; i < nnH1; i++ {
			acc := m.BSE2[i]
			row := m.WSE2[i*nnSE : (i+1)*nnSE]
			for k := 0; k < nnSE; k++ {
				acc += row[k] * vh[k]
			}
			gate[i] = sigmoid(acc)
		}
	} else {
		for i := range gate {
			gate[i] = 1
		}
	}

	h2 := make([]float64, nnH2)
	for j := 0; j < nnH2; j++ {
		acc := m.B2[j]
		row := m.W2[j*nnH1 : (j+1)*nnH1]
		for i := 0; i < nnH1; i++ {
			acc += h1[i] * gate[i] * row[i]
		}
		h2[j] = relu(acc)
	}
	out := m.B3[0]
	for i := 0; i < nnH2; i++ {
		out += h2[i] * m.W3[i]
	}
	return sigmoid(out), xIn, h1, gate, vh, gIn, vIn, h2
}

func (m *MLP) predict(x []float64) float64 {
	if m.train {
		prev := m.train
		m.train = false
		defer func() { m.train = prev }()
	}
	p, _, _, _, _, _, _, _ := m.forward(x)
	return p
}

// save writes the weights with gob; loadMLP reads them back.
func (m *MLP) save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(m)
}

func loadMLP(path string) (*MLP, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	m := &MLP{}
	if err := gob.NewDecoder(f).Decode(m); err != nil {
		return nil, err
	}
	return m, nil
}

// checkMLPDims verifies a loaded MLP's input dimension matches the current
// feature layout (which depends on NN_NOGM); weights trained under a different
// layout would otherwise be silently mis-sliced or panic.
func checkMLPDims(m *MLP) error {
	if m.W1 != nil && len(m.W1) != nnH1*nnInput {
		return fmt.Errorf("weights input dim %d != current %d (trained with different NN_NOGM?)", len(m.W1)/nnH1, nnInput)
	}
	return nil
}

// ---- training (Adam) ----

type adamState struct {
	mW1, vW1   []float64
	mW2, vW2   []float64
	mW3, vW3   []float64
	mb1, vb1   []float64
	mb2, vb2   []float64
	mb3, vb3   []float64
	mS1, vS1   []float64 // hidden attention down
	mS2, vS2   []float64 // hidden attention up
	mbs1, vbs1 []float64
	mbs2, vbs2 []float64
	mA1, vA1   []float64 // input attention down
	mA2, vA2   []float64 // input attention up
	mba1, vba1 []float64
	mba2, vba2 []float64
	t          int
}

func newAdam(m *MLP) *adamState {
	a := &adamState{
		mW1: make([]float64, len(m.W1)), vW1: make([]float64, len(m.W1)),
		mW2: make([]float64, len(m.W2)), vW2: make([]float64, len(m.W2)),
		mW3: make([]float64, len(m.W3)), vW3: make([]float64, len(m.W3)),
		mb1: make([]float64, len(m.B1)), vb1: make([]float64, len(m.B1)),
		mb2: make([]float64, len(m.B2)), vb2: make([]float64, len(m.B2)),
		mb3: make([]float64, len(m.B3)), vb3: make([]float64, len(m.B3)),
	}
	if m.WSE1 != nil {
		a.mS1 = make([]float64, len(m.WSE1))
		a.vS1 = make([]float64, len(m.WSE1))
		a.mS2 = make([]float64, len(m.WSE2))
		a.vS2 = make([]float64, len(m.WSE2))
		a.mbs1 = make([]float64, len(m.BSE1))
		a.vbs1 = make([]float64, len(m.BSE1))
		a.mbs2 = make([]float64, len(m.BSE2))
		a.vbs2 = make([]float64, len(m.BSE2))
	}
	if m.WA1 != nil {
		a.mA1 = make([]float64, len(m.WA1))
		a.vA1 = make([]float64, len(m.WA1))
		a.mA2 = make([]float64, len(m.WA2))
		a.vA2 = make([]float64, len(m.WA2))
		a.mba1 = make([]float64, len(m.BA1))
		a.vba1 = make([]float64, len(m.BA1))
		a.mba2 = make([]float64, len(m.BA2))
		a.vba2 = make([]float64, len(m.BA2))
	}
	return a
}

// grads holds the accumulated gradient of the mean BCE loss over a mini-batch.
type grads struct {
	W1, b1  []float64
	W2, b2  []float64
	W3, b3  []float64
	S1, bs1 []float64 // hidden attention down
	S2, bs2 []float64 // hidden attention up
	A1, bA1 []float64 // input attention down
	A2, bA2 []float64 // input attention up
}

func newGrads(m *MLP) *grads {
	g := &grads{
		W1: make([]float64, len(m.W1)), b1: make([]float64, len(m.B1)),
		W2: make([]float64, len(m.W2)), b2: make([]float64, len(m.B2)),
		W3: make([]float64, len(m.W3)), b3: make([]float64, len(m.B3)),
	}
	if m.WSE1 != nil {
		g.S1 = make([]float64, len(m.WSE1))
		g.bs1 = make([]float64, len(m.BSE1))
		g.S2 = make([]float64, len(m.WSE2))
		g.bs2 = make([]float64, len(m.BSE2))
	}
	if m.WA1 != nil {
		g.A1 = make([]float64, len(m.WA1))
		g.bA1 = make([]float64, len(m.BA1))
		g.A2 = make([]float64, len(m.WA2))
		g.bA2 = make([]float64, len(m.BA2))
	}
	return g
}

// backprop accumulates the (weighted) BCE gradient for one (x, y) sample into
// g. w is a per-sample loss weight used to balance positive/negative classes.
func (m *MLP) backprop(g *grads, x []float64, y float64, w float64) {
	prev := m.reuse
	m.reuse = true
	p, xIn, h1, gate, vh, gIn, vIn, h2 := m.forward(x)
	m.reuse = prev
	dO := (p - y) * w // dL/dz for BCE with sigmoid

	// output layer
	for i := 0; i < nnH2; i++ {
		g.W3[i] += dO * h2[i]
	}
	g.b3[0] += dO

	// hidden 2 -> 1
	dh2 := make([]float64, nnH2)
	for i := 0; i < nnH2; i++ {
		dh2[i] = dO * m.W3[i]
		if h2[i] <= 0 {
			dh2[i] = 0 // ReLU derivative
		}
	}
	for j := 0; j < nnH2; j++ {
		g.b2[j] += dh2[j]
		grow := g.W2[j*nnH1 : (j+1)*nnH1]
		for i := 0; i < nnH1; i++ {
			grow[i] += dh2[j] * h1[i] * gate[i]
		}
	}

	// gradient into h1a[i] = h1[i] * gate[i]
	da := make([]float64, nnH1)
	for i := 0; i < nnH1; i++ {
		var acc float64
		for j := 0; j < nnH2; j++ {
			acc += dh2[j] * m.W2[j*nnH1+i]
		}
		da[i] = acc
	}

	dh1 := make([]float64, nnH1)
	if m.WSE1 != nil {
		dz := make([]float64, nnH1)
		for i := 0; i < nnH1; i++ {
			dh1[i] = da[i] * gate[i]
			if h1[i] <= 0 {
				dh1[i] = 0 // ReLU derivative
			}
			dg := da[i] * h1[i]
			dz[i] = dg * gate[i] * (1 - gate[i]) // sigmoid derivative

			g.bs2[i] += dz[i]
			srow := g.S2[i*nnSE : (i+1)*nnSE]
			for k := 0; k < nnSE; k++ {
				srow[k] += dz[i] * vh[k]
			}
		}
		// through the hidden-attention bottleneck (ReLU down-projection)
		du := make([]float64, nnSE)
		for k := 0; k < nnSE; k++ {
			var acc float64
			for i := 0; i < nnH1; i++ {
				acc += dz[i] * m.WSE2[i*nnSE+k]
			}
			du[k] = acc
			if vh[k] <= 0 {
				du[k] = 0
			}
			g.bs1[k] += du[k]
			srow := g.S1[k*nnH1 : (k+1)*nnH1]
			for j := 0; j < nnH1; j++ {
				srow[j] += du[k] * h1[j]
			}
		}
	} else {
		// identity hidden attention: dh1 = da
		for i := 0; i < nnH1; i++ {
			dh1[i] = da[i]
			if h1[i] <= 0 {
				dh1[i] = 0
			}
		}
	}

	// first layer (uses the gated input xIn)
	for j := 0; j < nnH1; j++ {
		g.b1[j] += dh1[j]
		grow := g.W1[j*nnInput : (j+1)*nnInput]
		for i := 0; i < nnInput; i++ {
			grow[i] += dh1[j] * xIn[i]
		}
	}

	// input (pair-feature) attention: xIn[i] = x[i] * gIn[i] for i < nnP
	if m.WA1 != nil {
		dIn := make([]float64, nnP)
		for i := 0; i < nnP; i++ {
			var acc float64
			for j := 0; j < nnH1; j++ {
				acc += dh1[j] * m.W1[j*nnInput+i]
			}
			dIn[i] = acc
		}
		dz := make([]float64, nnP)
		for i := 0; i < nnP; i++ {
			dg := dIn[i] * x[i]
			dz[i] = dg * gIn[i] * (1 - gIn[i])

			g.bA2[i] += dz[i]
			row := g.A2[i*nnIP : (i+1)*nnIP]
			for k := 0; k < nnIP; k++ {
				row[k] += dz[i] * vIn[k]
			}
		}
		// through the input-attention bottleneck (ReLU down-projection)
		du := make([]float64, nnIP)
		for k := 0; k < nnIP; k++ {
			var acc float64
			for i := 0; i < nnP; i++ {
				acc += dz[i] * m.WA2[i*nnIP+k]
			}
			du[k] = acc
			if vIn[k] <= 0 {
				du[k] = 0
			}
			g.bA1[k] += du[k]
			row := g.A1[k*nnP : (k+1)*nnP]
			for j := 0; j < nnP; j++ {
				row[j] += du[k] * x[j]
			}
		}
	}
}

// step applies one Adam update using the mean gradient over n samples. l2 is
// the weight-decay rate (0 disables): each weight is shrunk by (1-lr*l2) per
// step, decoupled from the gradient magnitude (AdamW-style).
func (a *adamState) step(m *MLP, g *grads, n int, lr float64, l2 float64) {
	a.t++
	t := float64(a.t)
	b1m, b2m := 0.9, 0.999
	eps := 1e-8
	inv := 1 / float64(n)
	decay := 1.0
	if l2 > 0 {
		decay = 1 - lr*l2
	}

	adamVec := func(w, mw, vw, gw []float64) {
		for i := range w {
			gi := gw[i] * inv
			mw[i] = b1m*mw[i] + (1-b1m)*gi
			vw[i] = b2m*vw[i] + (1-b2m)*gi*gi
			mh := mw[i] / (1 - math.Pow(b1m, t))
			vh := vw[i] / (1 - math.Pow(b2m, t))
			if decay != 1 {
				w[i] *= decay
			}
			w[i] -= lr * mh / (math.Sqrt(vh) + eps)
		}
	}
	adamVec(m.W1, a.mW1, a.vW1, g.W1)
	adamVec(m.B1, a.mb1, a.vb1, g.b1)
	adamVec(m.W2, a.mW2, a.vW2, g.W2)
	adamVec(m.B2, a.mb2, a.vb2, g.b2)
	adamVec(m.W3, a.mW3, a.vW3, g.W3)
	adamVec(m.B3, a.mb3, a.vb3, g.b3)
	if m.WSE1 != nil {
		adamVec(m.WSE1, a.mS1, a.vS1, g.S1)
		adamVec(m.BSE1, a.mbs1, a.vbs1, g.bs1)
		adamVec(m.WSE2, a.mS2, a.vS2, g.S2)
		adamVec(m.BSE2, a.mbs2, a.vbs2, g.bs2)
	}
	if m.WA1 != nil {
		adamVec(m.WA1, a.mA1, a.vA1, g.A1)
		adamVec(m.BA1, a.mba1, a.vba1, g.bA1)
		adamVec(m.WA2, a.mA2, a.vA2, g.A2)
		adamVec(m.BA2, a.mba2, a.vba2, g.bA2)
	}
}
