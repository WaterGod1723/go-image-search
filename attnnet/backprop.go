package attnnet

import "math"

// grads mirrors every parameter slice of Model; backward accumulates into it.
type grads struct {
	wPatch, bPatch []float64
	wPos           []float64
	layers         []encGrad
	outW, outB     []float64
	gateW          []float64
	gateB          []float64
}

type encGrad struct {
	wQ, bQ []float64
	wK, bK []float64
	wV, bV []float64
	wO, bO []float64
	ln1G, ln1B []float64
	ln2G, ln2B []float64
	w1, b1 []float64
	w2, b2 []float64
}

func newGrads(m *Model) *grads {
	g := &grads{
		wPatch: make([]float64, len(m.WPatch)),
		bPatch: make([]float64, len(m.BPatch)),
		wPos:   make([]float64, len(m.WPos)),
		outW:   make([]float64, len(m.OutW)),
		outB:   make([]float64, len(m.OutB)),
		gateW:  make([]float64, len(m.GateW)),
		gateB:  make([]float64, len(m.GateB)),
	}
	for _, L := range m.Layers {
		g.layers = append(g.layers, encGrad{
			wQ: make([]float64, len(L.WQ)), bQ: make([]float64, len(L.BQ)),
			wK: make([]float64, len(L.WK)), bK: make([]float64, len(L.BK)),
			wV: make([]float64, len(L.WV)), bV: make([]float64, len(L.BV)),
			wO: make([]float64, len(L.WO)), bO: make([]float64, len(L.BO)),
			ln1G: make([]float64, len(L.LN1G)), ln1B: make([]float64, len(L.LN1B)),
			ln2G: make([]float64, len(L.LN2G)), ln2B: make([]float64, len(L.LN2B)),
			w1: make([]float64, len(L.W1)), b1: make([]float64, len(L.B1)),
			w2: make([]float64, len(L.W2)), b2: make([]float64, len(L.B2)),
		})
	}
	return g
}

// params walks the model's trainable slices; grads walks the same order.
func (m *Model) params() []*[]float64 {
	var p []*[]float64
	add := func(s *[]float64) { p = append(p, s) }
	add(&m.WPatch)
	add(&m.BPatch)
	add(&m.WPos)
	for i := range m.Layers {
		L := &m.Layers[i]
		add(&L.WQ)
		add(&L.BQ)
		add(&L.WK)
		add(&L.BK)
		add(&L.WV)
		add(&L.BV)
		add(&L.WO)
		add(&L.BO)
		add(&L.LN1G)
		add(&L.LN1B)
		add(&L.LN2G)
		add(&L.LN2B)
		add(&L.W1)
		add(&L.B1)
		add(&L.W2)
		add(&L.B2)
	}
	add(&m.OutW)
	add(&m.OutB)
	add(&m.GateW)
	add(&m.GateB)
	return p
}

func (g *grads) grads() []*[]float64 {
	var p []*[]float64
	add := func(s *[]float64) { p = append(p, s) }
	add(&g.wPatch)
	add(&g.bPatch)
	add(&g.wPos)
	for i := range g.layers {
		LG := &g.layers[i]
		add(&LG.wQ)
		add(&LG.bQ)
		add(&LG.wK)
		add(&LG.bK)
		add(&LG.wV)
		add(&LG.bV)
		add(&LG.wO)
		add(&LG.bO)
		add(&LG.ln1G)
		add(&LG.ln1B)
		add(&LG.ln2G)
		add(&LG.ln2B)
		add(&LG.w1)
		add(&LG.b1)
		add(&LG.w2)
		add(&LG.b2)
	}
	add(&g.outW)
	add(&g.outB)
	add(&g.gateW)
	add(&g.gateB)
	return p
}

// backward computes the gradients for one (x, mask) sample and accumulates them
// into g. It returns the CE loss and the gate BCE loss (0 for ModeGlobal).
func (m *Model) backward(a *acts, x []uint8, mask []uint8, g *grads) (float64, float64) {
	cfg := m.Cfg
	n := m.NTok
	D := cfg.D
	dx := cfg.patchDim()

	// ---- output head: dL/dlogits = p - y ----
	var cnt float64
	for j := 0; j < n; j++ {
		if mask[j] != 0 {
			cnt++
		}
	}
	invCnt := 0.0
	if cnt > 0 {
		invCnt = 1 / cnt
	}
	p := softmaxVec(a.logits)
	dlogits := make([]float64, n)
	for j := 0; j < n; j++ {
		t := 0.0
		if mask[j] != 0 {
			t = invCnt
		}
		dlogits[j] = p[j] - t
	}
	dpool := make([]float64, D)
	for k := 0; k < n; k++ {
		row := m.OutW[k*D : (k+1)*D]
		g.outB[k] += dlogits[k]
		for d := 0; d < D; d++ {
			g.outW[k*D+d] += dlogits[k] * a.pool[d]
			dpool[d] += dlogits[k] * row[d]
		}
	}
	dh := make([][]float64, n)
	for i := 0; i < n; i++ {
		dh[i] = make([]float64, D)
		for d := 0; d < D; d++ {
			dh[i][d] = dpool[d] / float64(n)
		}
	}

	for li := cfg.Depth - 1; li >= 0; li-- {
		L := &m.Layers[li]
		LG := &g.layers[li]
		hIn := a.h[li]
		n1 := a.xhat1[li]
		q, k, v := a.q[li], a.k[li], a.v[li]
		gv := a.g[li]

		// MLP branch first: it defines dL/dh1 = dh + dxH2, which the attention
		// branch (h1 = hIn + z) needs as its incoming gradient.
		dxRelu := make([][]float64, n)
		for i := 0; i < n; i++ {
			dxRelu[i] = make([]float64, cfg.MLPHidden)
		}
		for i := 0; i < n; i++ {
			r := a.relu[li][i]
			for o := 0; o < D; o++ {
				LG.b2[o] += dh[i][o]
				for j := 0; j < cfg.MLPHidden; j++ {
					LG.w2[o*cfg.MLPHidden+j] += dh[i][o] * r[j]
					if r[j] > 0 {
						dxRelu[i][j] += dh[i][o] * L.W2[o*cfg.MLPHidden+j]
					}
				}
			}
		}
		x2 := a.xhat2[li]
		dxLN2 := make([][]float64, n)
		for i := 0; i < n; i++ {
			dxLN2[i] = make([]float64, D)
		}
		for i := 0; i < n; i++ {
			for j := 0; j < cfg.MLPHidden; j++ {
				LG.b1[j] += dxRelu[i][j]
				for d := 0; d < D; d++ {
					LG.w1[j*D+d] += dxRelu[i][j] * x2[i][d]
					dxLN2[i][d] += dxRelu[i][j] * L.W1[j*D+d]
				}
			}
		}
		dH1 := make([][]float64, n)
		for i := 0; i < n; i++ {
			dH1[i] = make([]float64, D)
			dxr := layernormBackward(a.h1[li][i], dxLN2[i], L.LN2G, L.LN2B, LG.ln2G, LG.ln2B)
			for d := 0; d < D; d++ {
				dH1[i][d] = dh[i][d] + dxr[d]
			}
		}

		// attention branch (dy = dH1)
		dAttnIn, dWO, dBO := linearBackward(L.WO, L.BO, a.attnO[li], dH1, D)
		for o := 0; o < D; o++ {
			LG.bO[o] += dBO[o]
			for i := 0; i < D; i++ {
				LG.wO[o*D+i] += dWO[o*D+i]
			}
		}
		dq, dk, dv, dlgAttn := attentionBackward(dAttnIn, q, k, v, gv, a.attn[li], cfg)
		dxN1 := make([][]float64, n)
		for i := 0; i < n; i++ {
			dxN1[i] = make([]float64, D)
		}
		backLinearAccum(L.WQ, L.BQ, n1, dq, dxN1, LG.wQ, LG.bQ, D)
		backLinearAccum(L.WK, L.BK, n1, dk, dxN1, LG.wK, LG.bK, D)
		backLinearAccum(L.WV, L.BV, n1, dv, dxN1, LG.wV, LG.bV, D)

		// region gate backward (from attention value-gating + BCE loss).
		// dL/dlogit = dL_attn/dg * g*(1-g) + (g - mask)*GateLambda/(Depth*n)
		dGateIn := make([][]float64, n)
		for i := 0; i < n; i++ {
			dGateIn[i] = make([]float64, D)
		}
		if cfg.Mode == ModeIcon {
			for i := 0; i < n; i++ {
				gi := gv[i]
				dlogit := dlgAttn[i]*gi*(1-gi) + (gi-float64(mask[i]))*cfg.GateLambda/float64(cfg.Depth*n)
				g.gateB[0] += dlogit
				for d := 0; d < D; d++ {
					g.gateW[d] += dlogit * hIn[i][d]
					dGateIn[i][d] += dlogit * m.GateW[d]
				}
			}
		}
		// LayerNorm1 backward (+ residual through h1 = hIn + z)
		for i := 0; i < n; i++ {
			dxr := layernormBackward(hIn[i], dxN1[i], L.LN1G, L.LN1B, LG.ln1G, LG.ln1B)
			for d := 0; d < D; d++ {
				dH1[i][d] = dH1[i][d] + dxr[d] + dGateIn[i][d]
			}
		}
		dh = dH1
	}

	// ---- patch embed + positional embedding ----
	for i := 0; i < n; i++ {
		for d := 0; d < D; d++ {
			g.wPos[i*D+d] += dh[i][d]
			g.bPatch[d] += dh[i][d]
			for j := 0; j < dx; j++ {
				g.wPatch[d*dx+j] += dh[i][d] * float64(x[i*dx+j]) / 255
			}
		}
	}

	// ---- losses ----
	ce := ceMask(a.logits, mask)
	gateLoss := 0.0
	if cfg.Mode == ModeIcon {
		for li := 0; li < cfg.Depth; li++ {
			for i := 0; i < n; i++ {
				gateLoss += bce(a.g[li][i], float64(mask[i]))
			}
		}
		gateLoss *= cfg.GateLambda / float64(cfg.Depth*n)
	}
	return ce, gateLoss
}

// linearBackward backpropagates dy (n x out) through y = W*x + b, returning
// dx (n x in), dW (out x in), db (out). W is [out][in] row-major.
func linearBackward(w, b []float64, x, dy [][]float64, in int) ([][]float64, []float64, []float64) {
	out := len(b)
	n := len(x)
	dx := make([][]float64, n)
	for i := 0; i < n; i++ {
		dx[i] = make([]float64, in)
	}
	dW := make([]float64, out*in)
	db := make([]float64, out)
	for o := 0; o < out; o++ {
		row := w[o*in : (o+1)*in]
		for i := 0; i < n; i++ {
			db[o] += dy[i][o]
			for j := 0; j < in; j++ {
				dW[o*in+j] += dy[i][o] * x[i][j]
				dx[i][j] += dy[i][o] * row[j]
			}
		}
	}
	return dx, dW, db
}

// backLinearAccum backpropagates dy through a linear layer and accumulates both
// the parameter gradients and the input gradient dxOut.
func backLinearAccum(w, b []float64, x, dy, dxOut [][]float64, dW, dB []float64, in int) {
	out := len(b)
	n := len(x)
	for o := 0; o < out; o++ {
		row := w[o*in : (o+1)*in]
		for i := 0; i < n; i++ {
			dB[o] += dy[i][o]
			for j := 0; j < in; j++ {
				dW[o*in+j] += dy[i][o] * x[i][j]
				dxOut[i][j] += dy[i][o] * row[j]
			}
		}
	}
}

// attentionBackward backpropagates dOut through multi-head self-attention,
// returning dq/dk/dv and the dL/dg contribution of the value gate (ModeIcon).
func attentionBackward(dOut, q, k, v [][]float64, g []float64, attn [][][]float64, cfg Config) (dq, dk, dv [][]float64, dlogitG []float64) {
	n := len(q)
	D := cfg.D
	hd := cfg.headDim()
	heads := cfg.Heads
	scale := 1 / math.Sqrt(float64(hd))

	dq = make([][]float64, n)
	dk = make([][]float64, n)
	dv = make([][]float64, n)
	dlogitG = make([]float64, n)
	for i := 0; i < n; i++ {
		dq[i] = make([]float64, D)
		dk[i] = make([]float64, D)
		dv[i] = make([]float64, D)
	}

	for h := 0; h < heads; h++ {
		lo, hi := h*hd, (h+1)*hd
		probs := attn[h]
		// dA[i][j] = sum_d dOut[i][d] * vg[j][d]
		dA := make([][]float64, n)
		for i := 0; i < n; i++ {
			dA[i] = make([]float64, n)
			for j := 0; j < n; j++ {
				var acc float64
				for d := lo; d < hi; d++ {
					val := v[j][d]
					if cfg.Mode == ModeIcon {
						val *= g[j]
					}
					acc += dOut[i][d] * val
				}
				dA[i][j] = acc
			}
		}
		// softmax backward
		dS := make([][]float64, n)
		for i := 0; i < n; i++ {
			dS[i] = make([]float64, n)
			var dot float64
			for m := 0; m < n; m++ {
				dot += probs[i][m] * dA[i][m]
			}
			for j := 0; j < n; j++ {
				dS[i][j] = probs[i][j] * (dA[i][j] - dot)
			}
		}
		for i := 0; i < n; i++ {
			for d := lo; d < hi; d++ {
				var aq, ak float64
				for j := 0; j < n; j++ {
					aq += dS[i][j] * k[j][d]
					ak += dS[j][i] * q[j][d]
				}
				dq[i][d] += aq * scale
				dk[i][d] += ak * scale
			}
		}
		// dv and gate gradient
		for j := 0; j < n; j++ {
			var dg float64
			for d := lo; d < hi; d++ {
				var acc float64
				for i := 0; i < n; i++ {
					acc += probs[i][j] * dOut[i][d]
				}
				if cfg.Mode == ModeIcon {
					dg += acc * v[j][d]
					dv[j][d] += acc * g[j]
				} else {
					dv[j][d] += acc
				}
			}
			if cfg.Mode == ModeIcon {
				dlogitG[j] += dg
			}
		}
	}
	return dq, dk, dv, dlogitG
}

// layernormBackward backpropagates dy through a LayerNorm row (over the D dim),
// accumulating gamma/beta gradients; it returns the dx row.
func layernormBackward(x, dy, gamma, beta, dG, dB []float64) []float64 {
	D := len(x)
	dx := make([]float64, D)
	var mean float64
	for _, v := range x {
		mean += v
	}
	mean /= float64(D)
	var m2 float64
	for _, v := range x {
		d := v - mean
		m2 += d * d
	}
	std := math.Sqrt(m2/float64(D) + 1e-5)
	xh := make([]float64, D)
	for d := 0; d < D; d++ {
		xh[d] = (x[d] - mean) / std
		dG[d] += dy[d] * xh[d]
		dB[d] += dy[d]
	}
	var s1, s2 float64
	for d := 0; d < D; d++ {
		s1 += gamma[d] * dy[d]
		s2 += gamma[d] * dy[d] * xh[d]
	}
	for d := 0; d < D; d++ {
		dx[d] = (float64(D)*gamma[d]*dy[d] - s1 - xh[d]*s2) / (float64(D) * std)
	}
	return dx
}

func cloneRows(x [][]float64) [][]float64 {
	out := make([][]float64, len(x))
	for i := range x {
		out[i] = append([]float64(nil), x[i]...)
	}
	return out
}

// ---- Adam ----

type adamState struct {
	params []*[]float64
	gs     *grads
	grads  []*[]float64
	m, v   []float64
	t      int
}

func newAdam(m *Model) *adamState {
	params := m.params()
	g := newGrads(m)
	var total int
	for _, p := range params {
		total += len(*p)
	}
	return &adamState{
		params: params,
		gs:     g,
		grads:  g.grads(),
		m:      make([]float64, total),
		v:      make([]float64, total),
	}
}

// reset zeroes the accumulated gradient buffers.
func (a *adamState) reset() {
	for _, g := range a.grads {
		for i := range *g {
			(*g)[i] = 0
		}
	}
}

// apply does one Adam update: gradient buffers already hold the summed
// mini-batch gradients, invN is 1/batchSize, lr the learning rate.
func (a *adamState) apply(invN, lr float64) {
	a.t++
	t := float64(a.t)
	b1, b2 := 0.9, 0.999
	eps := 1e-8
	off := 0
	for pi := range a.params {
		w := *a.params[pi]
		gw := *a.grads[pi]
		for i := range w {
			gi := gw[i] * invN
			a.m[off] = b1*a.m[off] + (1-b1)*gi
			a.v[off] = b2*a.v[off] + (1-b2)*gi*gi
			mh := a.m[off] / (1 - math.Pow(b1, t))
			vh := a.v[off] / (1 - math.Pow(b2, t))
			w[i] -= lr * mh / (math.Sqrt(vh) + eps)
			off++
		}
	}
}
