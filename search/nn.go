package main

import (
	"encoding/gob"
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
	nnInput = 81 // 27 pair features + 2*27 query statistics
	nnH1    = 96
	nnH2    = 48
)

// MLP is a 3-layer feed-forward network with ReLU hidden units and a sigmoid
// output (relevance in [0,1]).
type MLP struct {
	W1, B1 []float64 // H1 x input  (row-major: W1[j*nnInput+i])
	W2, B2 []float64 // H2 x H1
	W3, B3 []float64 // 1  x H2
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

// forward runs the network and returns the output probability and the two
// hidden activation vectors (needed for backprop).
func (m *MLP) forward(x []float64) (float64, []float64, []float64) {
	h1 := make([]float64, nnH1)
	for j := 0; j < nnH1; j++ {
		v := m.B1[j]
		row := m.W1[j*nnInput : (j+1)*nnInput]
		for i := 0; i < nnInput; i++ {
			v += x[i] * row[i]
		}
		h1[j] = relu(v)
	}
	h2 := make([]float64, nnH2)
	for j := 0; j < nnH2; j++ {
		v := m.B2[j]
		row := m.W2[j*nnH1 : (j+1)*nnH1]
		for i := 0; i < nnH1; i++ {
			v += h1[i] * row[i]
		}
		h2[j] = relu(v)
	}
	out := m.B3[0]
	for i := 0; i < nnH2; i++ {
		out += h2[i] * m.W3[i]
	}
	return sigmoid(out), h1, h2
}

func (m *MLP) predict(x []float64) float64 {
	p, _, _ := m.forward(x)
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

// ---- training (Adam) ----

type adamState struct {
	mW1, vW1 []float64
	mW2, vW2 []float64
	mW3, vW3 []float64
	mb1, vb1 []float64
	mb2, vb2 []float64
	mb3, vb3 []float64
	t        int
}

func newAdam(m *MLP) *adamState {
	return &adamState{
		mW1: make([]float64, len(m.W1)), vW1: make([]float64, len(m.W1)),
		mW2: make([]float64, len(m.W2)), vW2: make([]float64, len(m.W2)),
		mW3: make([]float64, len(m.W3)), vW3: make([]float64, len(m.W3)),
		mb1: make([]float64, len(m.B1)), vb1: make([]float64, len(m.B1)),
		mb2: make([]float64, len(m.B2)), vb2: make([]float64, len(m.B2)),
		mb3: make([]float64, len(m.B3)), vb3: make([]float64, len(m.B3)),
	}
}

// grads holds the accumulated gradient of the mean BCE loss over a mini-batch.
type grads struct {
	W1, b1 []float64
	W2, b2 []float64
	W3, b3 []float64
}

func newGrads(m *MLP) *grads {
	return &grads{
		W1: make([]float64, len(m.W1)), b1: make([]float64, len(m.B1)),
		W2: make([]float64, len(m.W2)), b2: make([]float64, len(m.B2)),
		W3: make([]float64, len(m.W3)), b3: make([]float64, len(m.B3)),
	}
}

// backprop accumulates the (weighted) BCE gradient for one (x, y) sample into
// g. w is a per-sample loss weight used to balance positive/negative classes.
func (m *MLP) backprop(g *grads, x []float64, y float64, w float64) {
	p, h1, h2 := m.forward(x)
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
			grow[i] += dh2[j] * h1[i]
		}
	}

	// hidden 1 -> input
	dh1 := make([]float64, nnH1)
	for i := 0; i < nnH1; i++ {
		var v float64
		for j := 0; j < nnH2; j++ {
			v += dh2[j] * m.W2[j*nnH1+i]
		}
		dh1[i] = v
		if h1[i] <= 0 {
			dh1[i] = 0
		}
	}
	for j := 0; j < nnH1; j++ {
		g.b1[j] += dh1[j]
		grow := g.W1[j*nnInput : (j+1)*nnInput]
		for i := 0; i < nnInput; i++ {
			grow[i] += dh1[j] * x[i]
		}
	}
}

// step applies one Adam update using the mean gradient over n samples.
func (a *adamState) step(m *MLP, g *grads, n int, lr float64) {
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
	adamVec(m.W1, a.mW1, a.vW1, g.W1)
	adamVec(m.B1, a.mb1, a.vb1, g.b1)
	adamVec(m.W2, a.mW2, a.vW2, g.W2)
	adamVec(m.B2, a.mb2, a.vb2, g.b2)
	adamVec(m.W3, a.mW3, a.vW3, g.W3)
	adamVec(m.B3, a.mb3, a.vb3, g.b3)
}
