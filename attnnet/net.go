// Package attnnet trains a small attention-only network that localizes an icon
// on a synthetic canvas. It is fully standalone: its own go.mod, its own data
// generator and training loop, and no imports from the parent repo.
//
// The network tokenizes the input image into non-overlapping patches, embeds
// them with a learned linear projection plus a learned positional encoding,
// and runs a two-layer transformer encoder. Two attention modes are supported:
//
//   - ModeGlobal: vanilla multi-head self-attention over every patch, so the
//     network may attend anywhere on the canvas.
//   - ModeIcon: a learned per-patch region gate (sigmoid) is trained to cover
//     only the icon; the attention values are multiplied by the gate, so the
//     network can only "see" the icon region while encoding.
//
// The output is a per-patch attention distribution whose target is uniform over
// the icon patches, i.e. the model learns to produce an attention map over the
// icon. Metrics: attention mass inside the icon box and top-k IoU vs the ground
// truth mask.
package attnnet

import (
	"encoding/gob"
	"math"
	"math/rand"
	"os"
)

// Mode selects how self-attention is allowed to look at the canvas.
type Mode int

const (
	// ModeGlobal attends to every patch.
	ModeGlobal Mode = iota
	// ModeIcon gates attention so it can only look at the icon region.
	ModeIcon
)

// Config holds the network hyper-parameters.
type Config struct {
	ImageSize int // input image side in px (multiple of PatchSize)
	PatchSize int // patch side in px
	D         int // token embedding dim
	Heads     int // attention heads (D must be divisible by Heads)
	Depth     int // transformer encoder layers
	MLPHidden int // hidden dim of the per-token MLP
	Mode      Mode
	// GateLambda is the weight of the icon-region gate BCE loss (ModeIcon only).
	GateLambda float64
}

func (c Config) nTok() int { g := c.ImageSize / c.PatchSize; return g * g }

// NTok returns the number of patches/tokens for this config.
func (c Config) NTok() int { return c.nTok() }

// Grid returns the number of patches per image side.
func (c Config) Grid() int { return c.ImageSize / c.PatchSize }

func (c Config) patchDim() int { return c.PatchSize * c.PatchSize * 3 }

func (c Config) headDim() int { return c.D / c.Heads }

// Default returns the configuration used by the training command.
func Default() Config {
	return Config{
		ImageSize:  48,
		PatchSize:  6, // 8x8 = 64 tokens
		D:          32,
		Heads:      4,
		Depth:      2,
		MLPHidden:  64,
		Mode:       ModeIcon,
		GateLambda: 1.0,
	}
}

// encLayer is one transformer encoder block: LayerNorm -> attention (+residual)
// then LayerNorm -> MLP (+residual). Weights are [out][in] row-major. Fields are
// exported so the whole model serializes with encoding/gob.
type encLayer struct {
	WQ, BQ []float64 // D x D
	WK, BK []float64 // D x D
	WV, BV []float64 // D x D
	WO, BO []float64 // D x D
	LN1G, LN1B []float64 // layernorm before attention
	LN2G, LN2B []float64 // layernorm before MLP
	W1, B1 []float64     // D -> MLPHidden
	W2, B2 []float64     // MLPHidden -> D
}

// Model is the whole attention network.
type Model struct {
	Cfg    Config
	NTok   int
	Dim    int
	WPatch []float64 // D x patchDim
	BPatch []float64 // D
	WPos   []float64 // nTok x D
	Layers []encLayer
	OutW   []float64 // nTok x D (output head from mean-pooled tokens)
	OutB   []float64 // nTok
	// region gate head (ModeIcon): per-token scalar from its embedding.
	GateW []float64 // D
	GateB []float64
}

// New builds a randomly initialized model from cfg.
func New(cfg Config, seed int64) *Model {
	rng := rand.New(rand.NewSource(seed))
	m := &Model{Cfg: cfg, NTok: cfg.nTok(), Dim: cfg.D}
	m.WPatch = randN(rng, cfg.D*cfg.patchDim(), 0.1)
	m.BPatch = make([]float64, cfg.D)
	m.WPos = randN(rng, m.NTok*cfg.D, 0.02)
	for i := 0; i < cfg.Depth; i++ {
		L := encLayer{
			WQ:    randN(rng, cfg.D*cfg.D, 1/math.Sqrt(float64(cfg.D))),
			BQ:    make([]float64, cfg.D),
			WK:    randN(rng, cfg.D*cfg.D, 1/math.Sqrt(float64(cfg.D))),
			BK:    make([]float64, cfg.D),
			WV:    randN(rng, cfg.D*cfg.D, 1/math.Sqrt(float64(cfg.D))),
			BV:    make([]float64, cfg.D),
			WO:    randN(rng, cfg.D*cfg.D, 0.1),
			BO:    make([]float64, cfg.D),
			LN1G:  ones(cfg.D),
			LN1B:  make([]float64, cfg.D),
			LN2G:  ones(cfg.D),
			LN2B:  make([]float64, cfg.D),
			W1:    randN(rng, cfg.D*cfg.MLPHidden, 1/math.Sqrt(float64(cfg.D))),
			B1:    make([]float64, cfg.MLPHidden),
			W2:    randN(rng, cfg.MLPHidden*cfg.D, 1/math.Sqrt(float64(cfg.MLPHidden))),
			B2:    make([]float64, cfg.D),
		}
		m.Layers = append(m.Layers, L)
	}
	m.OutW = randN(rng, m.NTok*cfg.D, 0.1)
	m.OutB = make([]float64, m.NTok)
	m.GateW = randN(rng, cfg.D, 0.1)
	m.GateB = make([]float64, 1)
	return m
}

func randN(rng *rand.Rand, n int, scale float64) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = (rng.Float64()*2 - 1) * scale
	}
	return out
}

func ones(n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

// ---- forward ----

// acts stores every intermediate value needed by backward.
type acts struct {
	h     [][][]float64   // input tokens per layer [depth][nTok][D]
	h1    [][][]float64   // pre-LayerNorm2 residual input [depth][nTok][D]
	xhat1 [][][]float64   // LayerNorm1 output [depth][nTok][D]
	q, k, v [][][]float64 // QKV projections [depth][nTok][D]
	g     [][]float64     // gate values [depth][nTok] (ModeIcon)
	gl    [][]float64     // gate logits [depth][nTok]
	attnO [][][]float64   // attention output (pre-O-proj) [depth][nTok][D]
	attn  [][][][]float64 // attention probs [depth][heads][nTok][nTok]
	xhat2 [][][]float64   // LayerNorm2 output [depth][nTok][D]
	relu  [][][]float64   // MLP hidden after ReLU [depth][nTok][MLPHidden]
	pool  []float64       // mean-pooled last tokens [D]
	logits []float64      // output head logits [nTok]
}

// forward runs the network for one sample whose RGB pixel block is x
// (len = nTok*patchDim, layout: token-major then R,G,B per pixel). mask is the
// ground-truth per-patch icon mask (0/1). It returns the activations, the CE
// loss and, for ModeIcon, the mean gate BCE loss (0 for ModeGlobal).
func (m *Model) forward(x []uint8, mask []uint8) (*acts, float64, float64) {
	cfg := m.Cfg
	n := m.NTok
	D := cfg.D
	a := &acts{}
		h := make([][]float64, n)
	for i := 0; i < n; i++ {
		h[i] = make([]float64, D)
		for d := 0; d < D; d++ {
			var v float64
			wrow := m.WPatch[d*cfg.patchDim() : (d+1)*cfg.patchDim()]
			for j := 0; j < cfg.patchDim(); j++ {
				v += float64(x[i*cfg.patchDim()+j]) / 255 * wrow[j]
			}
			h[i][d] = v + m.BPatch[d] + m.WPos[i*D+d]
		}
	}

	cur := h
	for li := 0; li < cfg.Depth; li++ {
		L := &m.Layers[li]
		a.h = append(a.h, cur)
		// LayerNorm1
		n1 := make([][]float64, n)
		for i := 0; i < n; i++ {
			n1[i] = layernorm(cur[i], L.LN1G, L.LN1B)
		}
		a.xhat1 = append(a.xhat1, n1)
		// QKV projections
		q := matVecRows(L.WQ, L.BQ, n1, D)
		k := matVecRows(L.WK, L.BK, n1, D)
		v := matVecRows(L.WV, L.BV, n1, D)
		a.q = append(a.q, q)
		a.k = append(a.k, k)
		a.v = append(a.v, v)
		// region gate (ModeIcon) from the pre-LN residual input
		gl := make([]float64, n)
		g := make([]float64, n)
		if cfg.Mode == ModeIcon {
			for i := 0; i < n; i++ {
				z := m.GateB[0]
				for d := 0; d < D; d++ {
					z += m.GateW[d] * cur[i][d]
				}
				gl[i] = z
				g[i] = sigmoid(z)
			}
		}
		a.gl = append(a.gl, gl)
		a.g = append(a.g, g)
		// self-attention (multi-head)
		attnO, attn := selfAttn(q, k, v, g, cfg, cfg.Mode)
		a.attnO = append(a.attnO, attnO)
		a.attn = append(a.attn, attn)
		// O projection + residual
		z := matVecRows(L.WO, L.BO, attnO, D)
		h1 := make([][]float64, n)
		for i := 0; i < n; i++ {
			h1[i] = make([]float64, D)
			for d := 0; d < D; d++ {
				h1[i][d] = cur[i][d] + z[i][d]
			}
		}
		// LayerNorm2 + MLP + residual
		n2 := make([][]float64, n)
		mlpIn := make([][]float64, n)
		for i := 0; i < n; i++ {
			n2[i] = layernorm(h1[i], L.LN2G, L.LN2B)
			mlpIn[i] = n2[i]
		}
		a.h1 = append(a.h1, h1)
		a.xhat2 = append(a.xhat2, mlpIn)
		r := make([][]float64, n)
		for i := 0; i < n; i++ {
			r[i] = reluVec(matVec(L.W1, L.B1, n2[i], D))
		}
		a.relu = append(a.relu, r)
		out := make([][]float64, n)
		for i := 0; i < n; i++ {
			out[i] = matVec(L.W2, L.B2, r[i], cfg.MLPHidden)
			for d := 0; d < D; d++ {
				out[i][d] += h1[i][d]
			}
		}
		cur = out
	}

	// output head over mean-pooled tokens
	pool := make([]float64, D)
	for i := 0; i < n; i++ {
		for d := 0; d < D; d++ {
			pool[d] += cur[i][d] / float64(n)
		}
	}
	a.pool = pool
	logits := make([]float64, m.NTok)
	for i := 0; i < m.NTok; i++ {
		row := m.OutW[i*D : (i+1)*D]
		var v float64
		for d := 0; d < D; d++ {
			v += pool[d] * row[d]
		}
		logits[i] = v + m.OutB[i]
	}
	a.logits = logits

	// cross-entropy vs uniform target over the icon mask
	ce := ceMask(logits, mask)
	gateLoss := 0.0
	if cfg.Mode == ModeIcon && mask != nil {
		for li := 0; li < cfg.Depth; li++ {
			for i := 0; i < n; i++ {
				gi := a.g[li][i]
				t := float64(mask[i])
				gateLoss += bce(gi, t)
			}
		}
		gateLoss /= float64(cfg.Depth * n)
	}
	return a, ce, gateLoss
}

// Predict returns the softmax attention distribution over patches.
func (m *Model) Predict(x []uint8) []float64 {
	a, _, _ := m.forward(x, nil)
	return softmaxVec(a.logits)
}

// Save writes the model weights with gob; Load reads them back.
func (m *Model) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(m)
}

// Load reads a model written by Save.
func Load(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	m := &Model{}
	if err := gob.NewDecoder(f).Decode(m); err != nil {
		return nil, err
	}
	m.NTok = m.Cfg.nTok()
	m.Dim = m.Cfg.D
	return m, nil
}

// ---- primitives ----

func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

func reluVec(v []float64) []float64 {
	for i := range v {
		if v[i] < 0 {
			v[i] = 0
		}
	}
	return v
}

// matVec computes w*v + b where w is [out][in] row-major.
func matVec(w, b, v []float64, in int) []float64 {
	out := make([]float64, len(b))
	for o := 0; o < len(b); o++ {
		var acc float64
		row := w[o*in : (o+1)*in]
		for i := 0; i < in; i++ {
			acc += row[i] * v[i]
		}
		out[o] = acc + b[o]
	}
	return out
}

// matVecRows applies the same linear layer to every row of x (n x in).
func matVecRows(w, b []float64, x [][]float64, in int) [][]float64 {
	out := make([][]float64, len(x))
	for i := range x {
		out[i] = matVec(w, b, x[i], in)
	}
	return out
}

// layernorm normalizes v over its D dims with per-dim gamma/beta.
func layernorm(v, g, b []float64) []float64 {
	D := len(v)
	var mean, m2 float64
	for _, x := range v {
		mean += x
	}
	mean /= float64(D)
	for _, x := range v {
		d := x - mean
		m2 += d * d
	}
	std := math.Sqrt(m2/float64(D) + 1e-5)
	out := make([]float64, D)
	for i := 0; i < D; i++ {
		out[i] = (v[i]-mean)/std*g[i] + b[i]
	}
	return out
}

// selfAttn computes multi-head softmax attention. In ModeIcon, values are gated
// by g (per-token scalar), so a token can only gather content from patches the
// gate has opened.
func selfAttn(q, k, v [][]float64, g []float64, cfg Config, mode Mode) ([][]float64, [][][]float64) {
	n := len(q)
	D := cfg.D
	hd := cfg.headDim()
	heads := cfg.Heads
	scale := 1 / math.Sqrt(float64(hd))

	attn := make([][][]float64, heads)
	for h := 0; h < heads; h++ {
		attn[h] = make([][]float64, n)
	}
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float64, D)
	}

	for h := 0; h < heads; h++ {
		lo, hi := h*hd, (h+1)*hd
		// scores + softmax
		probs := make([][]float64, n)
		for i := 0; i < n; i++ {
			row := make([]float64, n)
			var mx float64 = -1e30
			for j := 0; j < n; j++ {
				var s float64
				for d := lo; d < hi; d++ {
					s += q[i][d] * k[j][d]
				}
				row[j] = s * scale
				if row[j] > mx {
					mx = row[j]
				}
			}
			var sum float64
			for j := 0; j < n; j++ {
				row[j] = math.Exp(row[j] - mx)
				sum += row[j]
			}
			for j := 0; j < n; j++ {
				row[j] /= sum
			}
			probs[i] = row
		}
		attn[h] = probs
		// weighted sum of (gated) values
		for i := 0; i < n; i++ {
			for d := lo; d < hi; d++ {
				var acc float64
				for j := 0; j < n; j++ {
					val := v[j][d]
					if mode == ModeIcon {
						val *= g[j]
					}
					acc += probs[i][j] * val
				}
				out[i][d] = acc
			}
		}
	}
	return out, attn
}

func softmaxVec(v []float64) []float64 {
	var mx float64 = -1e30
	for _, x := range v {
		if x > mx {
			mx = x
		}
	}
	var sum float64
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = math.Exp(x - mx)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// ceMask is the cross-entropy of softmax(logits) against a uniform target over
// the set of patches with mask=1.
func ceMask(logits []float64, mask []uint8) float64 {
	p := softmaxVec(logits)
	var cnt float64
	var ce float64
	for j := 0; j < len(mask); j++ {
		if mask[j] != 0 {
			cnt++
			ce -= math.Log(p[j])
		}
	}
	if cnt == 0 {
		return 0
	}
	return ce / cnt
}

func bce(p, t float64) float64 {
	p = math.Max(1e-7, math.Min(1-1e-7, p))
	return -(t*math.Log(p) + (1-t)*math.Log(1-p))
}
