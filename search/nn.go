package main

import (
	"encoding/gob"
	"math"
	"math/rand"
	"os"
)

// ---- learning-to-rank attention fusion ----
//
// Instead of hand-tuning (or per-query heuristically adapting) the fusion
// weights of the hand-crafted similarities, a small attention network is
// trained to predict, for a (query, ref) pair, how likely the ref is the true
// source of the query. The input is the per-pair similarities (the 7 from
// scores() plus the rotation-aligned mask score) augmented with per-query
// distribution statistics (best / second-best gap per feature), so the model
// can condition on query difficulty the way compositeAdaptive does by hand.
//
// The fusion is a single-head attention over semantically-grouped feature
// tokens:
//   - the pair block is split by feature family — base similarity scores,
//     gate/mono bias, per-ring shape detail, and the sczl expert sub-signals —
//     plus the two query-statistic blocks (best / gap), giving 6 tokens;
//   - the query is data-dependent: it is generated from the base-scores token
//     via WQ, so the attention weights adapt to each (query, ref) sample;
//   - each token has its own key/value projection (no weight sharing);
//   - the attended context passes through an FFN with a residual connection,
//     then a linear head to a sigmoid output.

const (
	// attnKey is the key/value (and query) dimension of the attention head.
	attnKey = 96
	// nFFN is the hidden width of the post-attention feed-forward block.
	nFFN = 96
)

// nnInput is the per-pair feature count: pair features + 2*query statistics
// (best / runner-up gap per feature) + the query's 32x32 color grid. It is a
// var (not a const) because the feature set is chosen at runtime via env flags
// (SCZLSPLIT); the network weight arrays are sized from it at construction.
var nnInput = nnNewInputDim()

// nnBaseScores is the fixed leading part of the pair block: the 8 base
// similarity scores.
const nnBaseScores = 8

// nnBiasDims is the gate+mono inductive-bias token width.
const nnBiasDims = 2

// nnColorDims is the width of the query's color projection histogram block
// (32 rows + 32 cols, each mean RGB).
const nnColorDims = 32 * 3 * 2

// sczlDim is the number of sczl sub-signals per ref in the current feature
// layout.
func sczlDim() int {
	if nnSczlSplit() {
		return 6
	}
	return 1
}

// nnPairDim returns the feature count of the pair block: 8 base pair features
// + 2 (gate/mono) + 16 per-ring polar-shape detail + sczl (1 fused, or 6
// sub-signals with SCZLSPLIT=1).
func nnPairDim() int {
	return nnBaseScores + nnBiasDims + 16 + sczlDim()
}

// nnNewInputDim computes the total input width: pair + best + gap (each pf)
// plus the query's color projection block (nnColorDims).
func nnNewInputDim() int {
	return nnPairDim() + 2*nnPairDim() + nnColorDims
}

// nnSczlSplit reports whether the split sczl sub-signal feature layout is
// active. It must match the flag used at training time.
func nnSczlSplit() bool {
	return os.Getenv("SCZLSPLIT") == "1"
}

// attnLayout describes the semantic token split of the input vector x.
// x = [pair | best | gap | color], pair = [base scores 8 | gate/mono 2 |
// ring detail 16 | sczl sczlDim()]. Tokens:
//
//	0. base   — the 8 base similarity scores
//	1. bias   — gate + mono inductive biases
//	2. detail — 16 per-ring polar-shape-detail cosines
//	3. sczl   — the sczl color-agnostic expert sub-signals
//	4. best   — per-query best statistics (dim pf)
//	5. gap    — per-query best-to-runner-up gap statistics (dim pf)
//	6. color  — the query's color projection histogram (nnColorDims = 192)
//
// The query (used by the attention head) is generated from the pair + best +
// gap region (3*pf dims), so the color token only feeds the key and value
// projections.
func attnLayout() (dims, offs []int) {
	pf := nnPairDim()
	dims = []int{nnBaseScores, nnBiasDims, 16, sczlDim(), pf, pf, nnColorDims}
	offs = make([]int, len(dims))
	for i := 1; i < len(dims); i++ {
		offs[i] = offs[i-1] + dims[i-1]
	}
	return dims, offs
}

// AttnNet is a single-head attention fusion network over semantically-grouped
// feature tokens with a sigmoid relevance output in [0,1].
//
//	tokens t = 0..7 as in attnLayout(); base = token 0
//	q    = WQ·x[0:3*pf]                                  (pair+best+gap region)
//	k_t  = WK[block_t]·block_t,  v_t = WV[block_t]·block_t
//	alpha_t = softmax(q·k_t / sqrt(attnKey))
//	ctx     = sum_t alpha_t · v_t
//	ctx     = ctx + FFN(ctx)                              (residual feed-forward)
//	p       = sigmoid(WO·ctx + BO)
//
// WK/WV are flattened per token: WK holds sum_t attnKey*dims[t] weights,
// token t at offset attnKey*sum_{s<t} dims[s]. All fields are exported so gob
// persist/restore works.
type AttnNet struct {
	PF   int   // query projection width (3*nnPairDim)
	Dims []int // per-token input dims (attnLayout)
	Offs []int // per-token offset in x (attnLayout)

	WQ []float64 // query projection from the pair+best+gap region: attnKey x PF

	WK []float64 // key projections, flattened: attnKey x dims[t] per token
	WV []float64 // value projections, flattened: attnKey x dims[t] per token

	W1, B1 []float64 // FFN hidden: nFFN x attnKey
	W2, B2 []float64 // FFN output: attnKey x nFFN

	WO []float64 // output head: attnKey
	BO []float64 // output bias: 1
}

func newAttnNet(seed int64) *AttnNet {
	rng := rand.New(rand.NewSource(seed))
	dims, offs := attnLayout()
	qd := 3 * nnPairDim()
	m := &AttnNet{
		PF:   qd,
		Dims: dims,
		Offs: offs,
		WQ:   randVec(rng, attnKey*qd, 0.6),
		W1:   randVec(rng, nFFN*attnKey, 0.3),
		B1:   make([]float64, nFFN),
		W2:   randVec(rng, attnKey*nFFN, 0.3),
		B2:   make([]float64, attnKey),
		WO:   randVec(rng, attnKey, 0.6),
		BO:   make([]float64, 1),
	}
	var wSize int
	for _, d := range dims {
		wSize += attnKey * d
	}
	m.WK = randVec(rng, wSize, 0.6)
	m.WV = randVec(rng, wSize, 0.6)
	return m
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

func softmax(v []float64) []float64 {
	mx := v[0]
	for _, x := range v[1:] {
		if x > mx {
			mx = x
		}
	}
	var s float64
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = math.Exp(x - mx)
		s += out[i]
	}
	for i := range out {
		out[i] /= s
	}
	return out
}

// attnActs holds the intermediate activations of a forward pass, needed for
// backprop.
type attnActs struct {
	q     []float64   // attnKey
	keys  [][]float64 // nTok x attnKey
	vals  [][]float64 // nTok x attnKey
	alpha []float64   // nTok
	ctx   []float64   // attnKey
	h     []float64   // nFFN (ReLU activations)
	ffn   []float64   // attnKey
	ctx2  []float64   // attnKey (post-residual)
}

// forward runs the network and returns the output probability plus the
// activations needed for backprop.
func (m *AttnNet) forward(x []float64) (float64, *attnActs) {
	a := &attnActs{}
	scale := 1 / math.Sqrt(float64(attnKey))
	nT := len(m.Dims)

	// data-dependent query from the full pair block
	a.q = make([]float64, attnKey)
	pf := m.PF
	for j := 0; j < attnKey; j++ {
		row := m.WQ[j*pf : (j+1)*pf]
		var v float64
		for i := 0; i < pf; i++ {
			v += row[i] * x[i]
		}
		a.q[j] = v
	}

	// per-token key/value projections + attention logits
	a.keys = make([][]float64, nT)
	a.vals = make([][]float64, nT)
	logits := make([]float64, nT)
	wOff := 0
	for t := 0; t < nT; t++ {
		d := m.Dims[t]
		tok := x[m.Offs[t] : m.Offs[t]+d]
		k := make([]float64, attnKey)
		v := make([]float64, attnKey)
		for j := 0; j < attnKey; j++ {
			krow := m.WK[wOff+j*d : wOff+(j+1)*d]
			vrow := m.WV[wOff+j*d : wOff+(j+1)*d]
			var kv, vv float64
			for i := 0; i < d; i++ {
				kv += krow[i] * tok[i]
				vv += vrow[i] * tok[i]
			}
			k[j] = kv
			v[j] = vv
		}
		a.keys[t] = k
		a.vals[t] = v
		var s float64
		for j := 0; j < attnKey; j++ {
			s += a.q[j] * k[j]
		}
		logits[t] = s * scale
		wOff += attnKey * d
	}
	a.alpha = softmax(logits)

	// attended context
	a.ctx = make([]float64, attnKey)
	for t := 0; t < nT; t++ {
		for j := 0; j < attnKey; j++ {
			a.ctx[j] += a.alpha[t] * a.vals[t][j]
		}
	}

	// residual feed-forward
	a.h = make([]float64, nFFN)
	for j := 0; j < nFFN; j++ {
		v := m.B1[j]
		row := m.W1[j*attnKey : (j+1)*attnKey]
		for i := 0; i < attnKey; i++ {
			v += row[i] * a.ctx[i]
		}
		a.h[j] = relu(v)
	}
	a.ffn = make([]float64, attnKey)
	for j := 0; j < attnKey; j++ {
		v := m.B2[j]
		row := m.W2[j*nFFN : (j+1)*nFFN]
		for i := 0; i < nFFN; i++ {
			v += row[i] * a.h[i]
		}
		a.ffn[j] = v
	}
	a.ctx2 = make([]float64, attnKey)
	for j := 0; j < attnKey; j++ {
		a.ctx2[j] = a.ctx[j] + a.ffn[j]
	}

	var z float64
	for j := 0; j < attnKey; j++ {
		z += m.WO[j] * a.ctx2[j]
	}
	z += m.BO[0]
	return sigmoid(z), a
}

func (m *AttnNet) predict(x []float64) float64 {
	p, _ := m.forward(x)
	return p
}

// valid reports whether the loaded weights match the expected shapes.
func (m *AttnNet) valid() bool {
	if m == nil || m.WQ == nil || m.WK == nil || m.WV == nil || m.WO == nil || m.BO == nil ||
		m.W1 == nil || m.B1 == nil || m.W2 == nil || m.B2 == nil {
		return false
	}
	if m.PF != 3*nnPairDim() {
		return false
	}
	dims, _ := attnLayout()
	if len(m.Dims) != len(dims) {
		return false
	}
	var wSize int
	for i, d := range dims {
		if m.Dims[i] != d {
			return false
		}
		wSize += attnKey * d
	}
	return len(m.WK) == wSize && len(m.WV) == wSize &&
		len(m.WQ) == attnKey*m.PF &&
		len(m.W1) == nFFN*attnKey && len(m.B1) == nFFN &&
		len(m.W2) == attnKey*nFFN && len(m.B2) == attnKey &&
		len(m.WO) == attnKey && len(m.BO) == 1
}

// save writes the weights with gob; loadAttnNet reads them back.
func (m *AttnNet) save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(m)
}

func loadAttnNet(path string) (*AttnNet, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	m := &AttnNet{}
	if err := gob.NewDecoder(f).Decode(m); err != nil {
		return nil, err
	}
	return m, nil
}

// ---- training (Adam) ----

type adamState struct {
	mWQ, vWQ []float64
	mWK, vWK []float64
	mWV, vWV []float64
	mW1, vW1 []float64
	mB1, vB1 []float64
	mW2, vW2 []float64
	mB2, vB2 []float64
	mWO, vWO []float64
	mBO, vBO []float64
	t        int
}

func newAdam(m *AttnNet) *adamState {
	return &adamState{
		mWQ: make([]float64, len(m.WQ)), vWQ: make([]float64, len(m.WQ)),
		mWK: make([]float64, len(m.WK)), vWK: make([]float64, len(m.WK)),
		mWV: make([]float64, len(m.WV)), vWV: make([]float64, len(m.WV)),
		mW1: make([]float64, len(m.W1)), vW1: make([]float64, len(m.W1)),
		mB1: make([]float64, len(m.B1)), vB1: make([]float64, len(m.B1)),
		mW2: make([]float64, len(m.W2)), vW2: make([]float64, len(m.W2)),
		mB2: make([]float64, len(m.B2)), vB2: make([]float64, len(m.B2)),
		mWO: make([]float64, len(m.WO)), vWO: make([]float64, len(m.WO)),
		mBO: make([]float64, len(m.BO)), vBO: make([]float64, len(m.BO)),
	}
}

// grads holds the accumulated gradient of the mean BCE loss over a mini-batch.
type grads struct {
	WQ []float64
	WK []float64
	WV []float64
	W1, B1 []float64
	W2, B2 []float64
	WO, BO []float64
}

func newGrads(m *AttnNet) *grads {
	return &grads{
		WQ: make([]float64, len(m.WQ)),
		WK: make([]float64, len(m.WK)),
		WV: make([]float64, len(m.WV)),
		W1: make([]float64, len(m.W1)), B1: make([]float64, len(m.B1)),
		W2: make([]float64, len(m.W2)), B2: make([]float64, len(m.B2)),
		WO: make([]float64, len(m.WO)), BO: make([]float64, len(m.BO)),
	}
}

// backprop accumulates the (weighted) BCE gradient for one (x, y) sample into
// g. w is a per-sample loss weight used to balance positive/negative classes.
func (m *AttnNet) backprop(g *grads, x []float64, y float64, w float64) {
	p, a := m.forward(x)
	dO := (p - y) * w // dL/dz for BCE with sigmoid
	scale := 1 / math.Sqrt(float64(attnKey))
	nT := len(m.Dims)

	// output head
	for j := 0; j < attnKey; j++ {
		g.WO[j] += dO * a.ctx2[j]
	}
	g.BO[0] += dO
	dctx2 := make([]float64, attnKey)
	for j := 0; j < attnKey; j++ {
		dctx2[j] = dO * m.WO[j]
	}

	// FFN backprop: f = relu(W1·ctx+B1) -> W2
	dh := make([]float64, nFFN)
	dctxFFN := make([]float64, attnKey)
	for j := 0; j < attnKey; j++ {
		g.B2[j] += dctx2[j]
		w2row := m.W2[j*nFFN : (j+1)*nFFN]
		g2row := g.W2[j*nFFN : (j+1)*nFFN]
		for i := 0; i < nFFN; i++ {
			g2row[i] += dctx2[j] * a.h[i]
			dh[i] += dctx2[j] * w2row[i]
		}
	}
	for i := 0; i < nFFN; i++ {
		if a.h[i] <= 0 {
			dh[i] = 0 // ReLU derivative
		}
		g.B1[i] += dh[i]
		w1row := m.W1[i*attnKey : (i+1)*attnKey]
		g1row := g.W1[i*attnKey : (i+1)*attnKey]
		for k := 0; k < attnKey; k++ {
			g1row[k] += dh[i] * a.ctx[k]
			dctxFFN[k] += dh[i] * w1row[k]
		}
	}
	// residual: dctx = dctx2 + dctxFFN
	dctx := make([]float64, attnKey)
	for k := 0; k < attnKey; k++ {
		dctx[k] = dctx2[k] + dctxFFN[k]
	}

	// dL/dalpha[t] = dctx · v_t
	dalpha := make([]float64, nT)
	for t := 0; t < nT; t++ {
		var s float64
		for j := 0; j < attnKey; j++ {
			s += dctx[j] * a.vals[t][j]
		}
		dalpha[t] = s
	}

	// softmax backprop: dL/dlogits[t] = alpha[t]*(dalpha[t] - sum_s alpha[s]*dalpha[s])
	var dot float64
	for t := 0; t < nT; t++ {
		dot += a.alpha[t] * dalpha[t]
	}
	dlogits := make([]float64, nT)
	for t := 0; t < nT; t++ {
		dlogits[t] = a.alpha[t] * (dalpha[t] - dot)
	}

	// dL/dv_t = alpha[t]*dctx  →  WV grads
	wOff := 0
	for t := 0; t < nT; t++ {
		d := m.Dims[t]
		tok := x[m.Offs[t] : m.Offs[t]+d]
		for j := 0; j < attnKey; j++ {
			gv := a.alpha[t] * dctx[j]
			vrow := g.WV[wOff+j*d : wOff+(j+1)*d]
			for i := 0; i < d; i++ {
				vrow[i] += gv * tok[i]
			}
		}
		wOff += attnKey * d
	}

	// dL/dk_t = dlogits[t]*scale*q  →  WK grads
	// dL/dq    = sum_t dlogits[t]*scale*k_t
	dq := make([]float64, attnKey)
	wOff = 0
	for t := 0; t < nT; t++ {
		d := m.Dims[t]
		tok := x[m.Offs[t] : m.Offs[t]+d]
		for j := 0; j < attnKey; j++ {
			dk := dlogits[t] * scale * a.q[j]
			krow := g.WK[wOff+j*d : wOff+(j+1)*d]
			for i := 0; i < d; i++ {
				krow[i] += dk * tok[i]
			}
		}
		for j := 0; j < attnKey; j++ {
			dq[j] += dlogits[t] * scale * a.keys[t][j]
		}
		wOff += attnKey * d
	}

	// q = WQ·pair  →  WQ grads
	pf := m.PF
	for j := 0; j < attnKey; j++ {
		qrow := g.WQ[j*pf : (j+1)*pf]
		for i := 0; i < pf; i++ {
			qrow[i] += dq[j] * x[i]
		}
	}
}

// step applies one Adam update using the mean gradient over n samples.
func (a *adamState) step(m *AttnNet, g *grads, n int, lr float64) {
	a.t++
	t := float64(a.t)
	b1m, b2m := 0.9, 0.999
	eps := 1e-8
	inv := 1 / float64(n)

	adamVec := func(w, mw, vw, gw []float64) {
		for i := range w {
			gi := gw[i] * inv
			mw[i] = b1m*mw[i] + (1-b1m)*gi
			vw[i] = b2m*vw[i] + (1-b2m)*gi*gi
			mh := mw[i] / (1 - math.Pow(b1m, t))
			vh := vw[i] / (1 - math.Pow(b2m, t))
			w[i] -= lr * mh / (math.Sqrt(vh) + eps)
		}
	}
	adamVec(m.WQ, a.mWQ, a.vWQ, g.WQ)
	adamVec(m.WK, a.mWK, a.vWK, g.WK)
	adamVec(m.WV, a.mWV, a.vWV, g.WV)
	adamVec(m.W1, a.mW1, a.vW1, g.W1)
	adamVec(m.B1, a.mB1, a.vB1, g.B1)
	adamVec(m.W2, a.mW2, a.vW2, g.W2)
	adamVec(m.B2, a.mB2, a.vB2, g.B2)
	adamVec(m.WO, a.mWO, a.vWO, g.WO)
	adamVec(m.BO, a.mBO, a.vBO, g.BO)
}