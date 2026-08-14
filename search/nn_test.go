package main

import (
	"math"
	"math/rand"
	"testing"
)

func bceLoss(p, y float64) float64 {
	if y == 1 {
		return -math.Log(p)
	}
	return -math.Log(1 - p)
}

// finite-difference check of backprop across every weight array (strided
// samples so the head-boundary indices are covered cheaply).
func TestGrad(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	m := newAttnNet(5)
	x := make([]float64, nnInput)
	for i := range x {
		x[i] = rng.Float64()
	}
	y := 1.0
	h := 1e-5

	check := func(name string, w []float64, g []float64) {
		for i := 0; i < len(w); i += 64 {
			// always probe array boundaries
			for _, probeIdx := range []int{i, len(w) - 1} {
				if probeIdx < 0 || probeIdx >= len(w) {
					continue
				}
				sav := w[probeIdx]
				w[probeIdx] = sav + h
				p, _ := m.forward(x)
				lp := bceLoss(p, y)
				w[probeIdx] = sav - h
				p, _ = m.forward(x)
				lm := bceLoss(p, y)
				w[probeIdx] = sav
				num := (lp - lm) / (2 * h)
				anal := g[probeIdx]
				if math.Abs(num-anal) > 1e-4*(1+math.Abs(num)+math.Abs(anal)) {
					t.Fatalf("%s[%d]: num=%.8g anal=%.8g", name, probeIdx, num, anal)
				}
			}
		}
	}

	g := newGrads(m)
	m.backprop(g, x, y, 1)
	check("WQ", m.WQ, g.WQ)
	check("WK", m.WK, g.WK)
	check("WV", m.WV, g.WV)
	check("W1", m.W1, g.W1)
	check("B1", m.B1, g.B1)
	check("W2", m.W2, g.W2)
	check("B2", m.B2, g.B2)
	check("WO", m.WO, g.WO)
	check("BO", m.BO, g.BO)
	for pi, p := range m.Proj {
		check("ProjW1", p.W1, g.Proj[pi].W1)
		check("ProjB1", p.B1, g.Proj[pi].B1)
		check("ProjW2", p.W2, g.Proj[pi].W2)
		check("ProjB2", p.B2, g.Proj[pi].B2)
	}
	t.Log("gradient check OK")
}

func randArr(rng *rand.Rand, n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = rng.Float64()
	}
	return v
}

// TestSNFullChain verifies the gradient from the BCE loss through the attention
// net, the structure block's overlapDiff view, and the rotation-pooled
// StructNet CNN/MLP (including the query rotation max-pool) against finite
// differences.
func TestSNFullChain(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	m := newAttnNet(7)
	sn := m.SN

	mkFeat := func() *Feat {
		return &Feat{
			Zern:    randArr(rng, len(zernModes)),
			Hist:    randArr(rng, hBins*sBins*vBins),
			Radial:  randArr(rng, nRing),
			AngMag:  randArr(rng, nRing*nFreq),
			Thumb16: randArr(rng, snGrid*snGrid),
		}
	}
	q, r := mkFeat(), mkFeat()

	lossAt := func() float64 {
		// recompute q/r embeddings from the (possibly perturbed) CNN so the
		// finite difference sees weight changes, but keep snCur inactive so the
		// struct block is written from these fresh embeddings without backprop.
		q.Emb = sn.embedEmbRot(q.Thumb16)
		r.Emb = sn.embedEmb(r.Thumb16)
		xb := make([]float64, nnInput)
		saved := snCur.active
		snCur.active = false
		nnVecInto(xb, q, r)
		snCur.active = saved
		p := m.predict(xb)
		return bceLoss(p, 1)
	}

	// analytic gradient wrt the SN weights through the deferred path:
	// nnVecInto snapshots q/r embeddings, backprop calls snSplitGrad which
	// accumulates into per-image qAcc/rAcc (indices 0/0), then we run the CNN
	// backward on those accumulations.
	q.Emb = sn.embedEmbRot(q.Thumb16)
	r.Emb = sn.embedEmb(r.Thumb16)
	setupSN(sn, 1, 1)
	xb := make([]float64, nnInput)
	snCur.qIdx, snCur.rIdx = 0, 0
	nnVecInto(xb, q, r)
	g := newGrads(m)
	m.backprop(g, xb, 1, 1)
	sg := newSNGrads(sn)
	qFs, qmhs, _, wins := sn.embedRot(q.Thumb16)
	rF, rmh, _ := sn.embed(r.Thumb16)
	sn.backwardRot(snCur.qAcc, q.Thumb16, qFs, qmhs, wins, sg)
	sn.backward(snCur.rAcc, r.Thumb16, rF, rmh, sg)
	snCur.active = false

	h := 1e-5
	check := func(name string, w []float64, gw []float64) {
		probes := map[int]bool{0: true, len(w) - 1: true}
		for i := 0; i < 6 && i < len(w); i++ {
			probes[rng.Intn(len(w))] = true
		}
		for idx := range probes {
			sav := w[idx]
			w[idx] = sav + h
			lp := lossAt()
			w[idx] = sav - h
			lm := lossAt()
			w[idx] = sav
			num := (lp - lm) / (2 * h)
			anal := gw[idx]
			if math.Abs(num-anal) > 1e-3*(1+math.Abs(num)+math.Abs(anal)) {
				t.Fatalf("%s[%d]: num=%.8g anal=%.8g", name, idx, num, anal)
			}
		}
	}
	check("SN.W1", sn.W1, sg.W1)
	check("SN.B1", sn.B1, sg.B1)
	check("SN.W2", sn.W2, sg.W2)
	check("SN.B2", sn.B2, sg.B2)
	check("SN.W3", sn.W3, sg.W3)
	check("SN.B3", sn.B3, sg.B3)
	check("SN.W4", sn.W4, sg.W4)
	check("SN.B4", sn.B4, sg.B4)
	t.Log("structure-CNN chain gradient OK")
}