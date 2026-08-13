package main

import (
	"encoding/gob"
	"math"
	"math/rand"
	"os"
)

// ---- learning-to-rank attention fusion over learned similarity tokens ----
//
// The hand-crafted per-pair similarity scores (hist/radial/angmag/polarShift/
// polcol/roundTrip/zernike/mask), the gate/mono biases, the per-ring detail,
// the sczl expert signals and the color projection histogram are REMOVED.
// Instead the model receives, for each (query, ref) pair, four raw per-mode
// DUAL-VIEW blocks of the rotation-invariant descriptors — per component both
// the overlap `min(q,r)` (how much both have) and the directed difference
// `|q-r|` (who is missing what):
//   - Zernike magnitudes (len(zernModes) dims),
//   - HSV histogram    (hBins*sBins*vBins dims),
//   - radial density    (nRing dims),
//   - angular FFT mags  (nRing*nFreq dims).
//
// Each block is compressed by a small jointly-trained projection MLP
// (In -> Hid -> Out, ReLU hidden) into a 12-dim token; the query (used by the
// attention head) is generated from those projected tokens via WQ, so the
// network learns its own notion of "which descriptors, which sub-bands matter
// for this pair" instead of receiving hand-tuned similarities. Attention
// weights adapt per sample; tokens have per-token key/value projections (no
// weight sharing); the attended context passes a residual FFN then a linear
// head to a sigmoid output. The projection MLPs are trained joint with the
// network (the raw blocks are carried at the end of the input vector,
// invisible to attention, and patched into their token slots by forward).

const (
	// attnKey is the key/value (and query) dimension of the attention head.
	attnKey = 96
	// nFFN is the hidden width of the post-attention feed-forward block.
	nFFN = 96
)

// nnInput is the per-pair input width: the projected-token slots
// (nnPairDim) plus the raw blocks carried for the input projectors. It is a
// var (not a const) because the descriptor dims are fixed at package init.
var nnInput = nnNewInputDim()

// ---- joint-trainable input projectors ----
//
// InputProjector maps a raw per-pair feature block (In dims, carried at the
// end of the input vector, invisible to the attention) to a compact token (Out
// dims) that the attention consumes. It is a two-layer MLP (In -> Hid -> Out,
// ReLU hidden) trained jointly with the network: backprop computes the loss
// gradient wrt the projected token (WK/WV contributions, plus WQ since the
// token lies in the query region) and flows it through the MLP. This lets
// high-dimensional raw features (e.g. the 576-dim histogram overlap+diff
// block) enter the fusion as learned embeddings instead of hand-collapsed
// scalar similarities. Add future jointly trained feature blocks by declaring
// them in attnProjConfigs and appending their raw block to the input vector.

// nnProjHid is the hidden width of every input projector.
const nnProjHid = 8

// nnProjOut is the projected token width of every input projector.
const nnProjOut = 12

// InputProjector is a small jointly-trained projection MLP. All fields are
// exported so gob persist/restore works.
type InputProjector struct {
	In, Hid, Out int
	W1, B1       []float64 // Hid x In
	W2, B2       []float64 // Out x Hid
}

type projCfg struct{ in, hid, out int }

// attnProjConfigs declares the input projectors in layout order. Each entry
// adds a projected token of Out dims to the attention layout and consumes In
// carry dims appended at the very end of the input vector. Every raw block
// carries BOTH views (overlap min then directed diff), so In is 2x the
// descriptor dims. The raw blocks built by buildPairVec must be concatenated
// in exactly this order.
func attnProjConfigs() []projCfg {
	return []projCfg{
		{in: 2 * len(zernModes), hid: nnProjHid, out: nnProjOut},         // zern   min+diff
		{in: 2 * hBins * sBins * vBins, hid: nnProjHid, out: nnProjOut},  // hist   min+diff
		{in: 2 * nRing, hid: nnProjHid, out: nnProjOut},                  // radial min+diff
		{in: 2 * nRing * nFreq, hid: nnProjHid, out: nnProjOut},          // angmag min+diff
	}
}

// nnProjRawIn returns the total carry dims consumed by all input projectors.
func nnProjRawIn() int {
	n := 0
	for _, c := range attnProjConfigs() {
		n += c.in
	}
	return n
}

// nnPairDim returns the projected-token block width: the sum of every input
// projector's Out dims.
func nnPairDim() int {
	d := 0
	for _, c := range attnProjConfigs() {
		d += c.out
	}
	return d
}

// nnNewInputDim computes the total input width: the projected-token slots plus
// the raw carry blocks of every input projector.
func nnNewInputDim() int {
	return nnPairDim() + nnProjRawIn()
}

// attnLayout describes the semantic token split of the input vector x.
// x = [token slots (patched by the projectors in forward) | raw carry blocks].
// Tokens:
//
//	0. zern   — projected Zernike min+diff (nnProjOut)
//	1. hist   — projected HSV min+diff (nnProjOut)
//	2. radial — projected per-ring min+diff (nnProjOut)
//	3. angmag — projected angular-FFT min+diff (nnProjOut)
//
// The query (used by the attention head) is generated from the first
// nnPairDim() dims (the projected token slots) via WQ. The raw carry blocks
// are invisible to the attention; forward projects them into the slots.
func attnLayout() (dims, offs []int) {
	for _, c := range attnProjConfigs() {
		dims = append(dims, c.out)
	}
	offs = make([]int, len(dims))
	for i := 1; i < len(dims); i++ {
		offs[i] = offs[i-1] + dims[i-1]
	}
	return dims, offs
}

// AttnNet is a single-head attention fusion network over semantically-grouped
// feature tokens with a sigmoid relevance output in [0,1].
//
//	tokens t = 0..3 (projected similarity tokens, see attnLayout())
//	q    = WQ·x[0:PF]                                   (projected tokens region)
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
	PF   int   // query projection width (nnPairDim)
	Dims []int // per-token input dims (attnLayout)
	Offs []int // per-token offset in x (attnLayout)

	WQ []float64 // query projection from the projected-token region: attnKey x PF

	WK []float64 // key projections, flattened: attnKey x dims[t] per token
	WV []float64 // value projections, flattened: attnKey x dims[t] per token

	W1, B1 []float64 // FFN hidden: nFFN x attnKey
	W2, B2 []float64 // FFN output: attnKey x nFFN

	WO []float64 // output head: attnKey
	BO []float64 // output bias: 1

	Proj []*InputProjector // joint-trainable input projectors (nil-safe)
}

func newAttnNet(seed int64) *AttnNet {
	rng := rand.New(rand.NewSource(seed))
	dims, offs := attnLayout()
	qd := nnPairDim()
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
	for _, c := range attnProjConfigs() {
		m.Proj = append(m.Proj, newInputProjector(rng, c.in, c.hid, c.out))
	}
	return m
}

func newInputProjector(rng *rand.Rand, in, hid, out int) *InputProjector {
	return &InputProjector{
		In: in, Hid: hid, Out: out,
		W1: randVec(rng, hid*in, 0.3),
		B1: make([]float64, hid),
		W2: randVec(rng, out*hid, 0.3),
		B2: make([]float64, out),
	}
}

// apply runs the projector over a raw block and returns the hidden activations
// (needed by backward) and the projected token.
func (p *InputProjector) apply(in []float64) (h, out []float64) {
	h = make([]float64, p.Hid)
	for j := 0; j < p.Hid; j++ {
		v := p.B1[j]
		row := p.W1[j*p.In : (j+1)*p.In]
		for i := 0; i < p.In; i++ {
			v += row[i] * in[i]
		}
		h[j] = relu(v)
	}
	out = make([]float64, p.Out)
	for j := 0; j < p.Out; j++ {
		v := p.B2[j]
		row := p.W2[j*p.Hid : (j+1)*p.Hid]
		for i := 0; i < p.Hid; i++ {
			v += row[i] * h[i]
		}
		out[j] = v
	}
	return h, out
}

// backward accumulates the projector's weight gradients given the forward
// hidden activations h, raw input in, and the loss gradient dz wrt its output.
func (p *InputProjector) backward(g *projGrad, h, in, dz []float64) {
	// z = W2·h + B2
	for j := 0; j < p.Out; j++ {
		g.B2[j] += dz[j]
		g2row := g.W2[j*p.Hid : (j+1)*p.Hid]
		for i := 0; i < p.Hid; i++ {
			g2row[i] += dz[j] * h[i]
		}
	}
	// h = relu(W1·in + B1)
	dh := make([]float64, p.Hid)
	for i := 0; i < p.Hid; i++ {
		if h[i] <= 0 {
			continue
		}
		for j := 0; j < p.Out; j++ {
			dh[i] += dz[j] * p.W2[j*p.Hid+i]
		}
	}
	for i := 0; i < p.Hid; i++ {
		g.B1[i] += dh[i]
		g1row := g.W1[i*p.In : (i+1)*p.In]
		for k := 0; k < p.In; k++ {
			g1row[k] += dh[i] * in[k]
		}
	}
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
	xeff  []float64    // input with projected tokens patched in
	q     []float64    // attnKey
	keys  [][]float64  // nTok x attnKey
	vals  [][]float64  // nTok x attnKey
	alpha []float64    // nTok
	ctx   []float64    // attnKey
	h     []float64    // nFFN (ReLU activations)
	ffn   []float64    // attnKey
	ctx2  []float64    // attnKey (post-residual)
	pin   [][]float64  // per-projector raw input blocks
	ph    [][]float64  // per-projector hidden activations
}

// forward runs the network and returns the output probability plus the
// activations needed for backprop.
func (m *AttnNet) forward(x []float64) (float64, *attnActs) {
	a := &attnActs{}
	scale := 1 / math.Sqrt(float64(attnKey))
	nT := len(m.Dims)

	// Patch the projected tokens: replace each projector's token slot with the
	// MLP output over its raw carry block.
	a.xeff = x
	if len(m.Proj) > 0 {
		a.xeff = make([]float64, len(x))
		copy(a.xeff, x)
		rawOff := len(x) - nnProjRawIn()
		for pi, p := range m.Proj {
			tokIdx := len(m.Dims) - len(m.Proj) + pi
			tokOff := m.Offs[tokIdx]
			in := x[rawOff : rawOff+p.In]
			h, z := p.apply(in)
			copy(a.xeff[tokOff:tokOff+p.Out], z)
			a.pin = append(a.pin, in)
			a.ph = append(a.ph, h)
			rawOff += p.In
		}
	}

	// data-dependent query from the full pair block
	a.q = make([]float64, attnKey)
	pf := m.PF
	for j := 0; j < attnKey; j++ {
		row := m.WQ[j*pf : (j+1)*pf]
		var v float64
		for i := 0; i < pf; i++ {
			v += row[i] * a.xeff[i]
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
		tok := a.xeff[m.Offs[t] : m.Offs[t]+d]
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
	if m.PF != nnPairDim() {
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
		len(m.WO) == attnKey && len(m.BO) == 1 &&
		validProjectors(m)
}

// validProjectors checks the input-projector shapes against attnProjConfigs.
func validProjectors(m *AttnNet) bool {
	cfgs := attnProjConfigs()
	if len(m.Proj) != len(cfgs) {
		return false
	}
	for i, p := range m.Proj {
		c := cfgs[i]
		if p == nil || p.In != c.in || p.Hid != c.hid || p.Out != c.out {
			return false
		}
		if len(p.W1) != c.hid*c.in || len(p.B1) != c.hid ||
			len(p.W2) != c.out*c.hid || len(p.B2) != c.out {
			return false
		}
	}
	return true
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

type adamProj struct {
	mW1, vW1, mB1, vB1 []float64
	mW2, vW2, mB2, vB2 []float64
}

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
	Proj     []*adamProj
	t        int
}

func newAdam(m *AttnNet) *adamState {
	a := &adamState{
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
	for _, p := range m.Proj {
		a.Proj = append(a.Proj, &adamProj{
			mW1: make([]float64, len(p.W1)), vW1: make([]float64, len(p.W1)),
			mB1: make([]float64, len(p.B1)), vB1: make([]float64, len(p.B1)),
			mW2: make([]float64, len(p.W2)), vW2: make([]float64, len(p.W2)),
			mB2: make([]float64, len(p.B2)), vB2: make([]float64, len(p.B2)),
		})
	}
	return a
}

// grads holds the accumulated gradient of the mean BCE loss over a mini-batch.
type projGrad struct {
	W1, B1 []float64
	W2, B2 []float64
}

type grads struct {
	WQ []float64
	WK []float64
	WV []float64
	W1, B1 []float64
	W2, B2 []float64
	WO, BO []float64
	Proj []*projGrad
}

func newGrads(m *AttnNet) *grads {
	g := &grads{
		WQ: make([]float64, len(m.WQ)),
		WK: make([]float64, len(m.WK)),
		WV: make([]float64, len(m.WV)),
		W1: make([]float64, len(m.W1)), B1: make([]float64, len(m.B1)),
		W2: make([]float64, len(m.W2)), B2: make([]float64, len(m.B2)),
		WO: make([]float64, len(m.WO)), BO: make([]float64, len(m.BO)),
	}
	for _, p := range m.Proj {
		g.Proj = append(g.Proj, &projGrad{
			W1: make([]float64, len(p.W1)), B1: make([]float64, len(p.B1)),
			W2: make([]float64, len(p.W2)), B2: make([]float64, len(p.B2)),
		})
	}
	return g
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
		tok := a.xeff[m.Offs[t] : m.Offs[t]+d]
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
		tok := a.xeff[m.Offs[t] : m.Offs[t]+d]
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
			qrow[i] += dq[j] * a.xeff[i]
		}
	}

	// Joint-trainable input projectors: the loss gradient wrt each projected
	// token (WK + WV contributions, plus WQ since the token lies in the query
	// region) is back-propagated through the projector MLP.
	for pi, p := range m.Proj {
		tokIdx := len(m.Dims) - len(m.Proj) + pi
		d := m.Dims[tokIdx]
		off := m.Offs[tokIdx]
		wOff := 0
		for t := 0; t < tokIdx; t++ {
			wOff += attnKey * m.Dims[t]
		}
		dzd := make([]float64, d)
		for j := 0; j < attnKey; j++ {
			dk := dlogits[tokIdx] * scale * a.q[j]
			gv := a.alpha[tokIdx] * dctx[j]
			krow := m.WK[wOff+j*d : wOff+(j+1)*d]
			vrow := m.WV[wOff+j*d : wOff+(j+1)*d]
			for i := 0; i < d; i++ {
				dzd[i] += dk*krow[i] + gv*vrow[i]
			}
		}
		if off+d <= nnPairDim() {
			for i := 0; i < d; i++ {
				var s float64
				for j := 0; j < attnKey; j++ {
					s += dq[j] * m.WQ[j*pf+off+i]
				}
				dzd[i] += s
			}
		}
		p.backward(g.Proj[pi], a.ph[pi], a.pin[pi], dzd)
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
	for pi, p := range m.Proj {
		ap := a.Proj[pi]
		gp := g.Proj[pi]
		adamVec(p.W1, ap.mW1, ap.vW1, gp.W1)
		adamVec(p.B1, ap.mB1, ap.vB1, gp.B1)
		adamVec(p.W2, ap.mW2, ap.vW2, gp.W2)
		adamVec(p.B2, ap.mB2, ap.vB2, gp.B2)
	}
}