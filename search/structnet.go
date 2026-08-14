package main

import "math/rand"

// ---- rotation-pooled structure CNN ----
//
// The descriptors fed to the attention net are all per-sprite statistics that
// discard spatial layout; StructNet recovers a coarse OVERALL-STRUCTURE signal.
// It is a tiny CNN over the 16x16 normalized occupancy thumbnail:
//
//	conv 3x3 (pad 1, 8 ch) -> relu -> 2x2 pool -> 16x16x8
//	conv 3x3 (pad 1, 8 ch) -> relu -> 2x2 pool -> 8x8x8
//	flatten 4x4x8 = 128 -> MLP(32) -> emb (16 dims)
//
// Queries are arbitrarily-rotated renders, so the EMBEDDING of a query is max-
// pooled over the 4 multiples-of-90° rotations of its thumbnail: the pooled
// embedding is invariant under 90° steps and degrades gracefully in between
// (the rotation-invariant descriptors carry the continuum). The ref uses its
// single absolute orientation. The structure pair block then feeds the fused
// attention the rotation-aware [min(qEmb,rEmb)|diff(qEmb,rEmb)] view.
//
// The weights are joint-trained, but with deferred backward: at each epoch the
// embeddings of every query/ref are recomputed once, the per-sample head/block
// backprop accumulates gradients w.r.t. those embeddings (snQAcc/snRAcc), and
// at the end of the epoch the CNN/MLP backward runs once per image with the
// accumulated gradients, so conv work happens O(images) instead of O(pairs).

const (
	snGrid = thumbN // 16
	snC1   = 8
	snC2   = 8
	snPool = 4           // after two 2x2 pools: 16 -> 8 -> 4
	snFMap = snPool * snPool * snC2
	snHid  = 32
	snEmb  = 16
	snRot  = 4 // multiples of 90°
)

// StructNet is the structure-CNN weight set. All fields are exported so gob
// persist/restore works.
type StructNet struct {
	W1, B1 []float64 // conv1: snC1 x 1 x 3 x 3
	W2, B2 []float64 // conv2: snC2 x snC1 x 3 x 3
	W3, B3 []float64 // MLP1: snHid x snFMap
	W4, B4 []float64 // MLP2: snEmb x snHid
}

func newStructNet(rng *rand.Rand) *StructNet {
	return &StructNet{
		W1: randVec(rng, snC1*9, 0.3), B1: make([]float64, snC1),
		W2: randVec(rng, snC2*snC1*9, 0.3), B2: make([]float64, snC2),
		W3: randVec(rng, snHid*snFMap, 0.3), B3: make([]float64, snHid),
		W4: randVec(rng, snEmb*snHid, 0.3), B4: make([]float64, snEmb),
	}
}

// ensureStructEmb fills f.Emb for every ref (single orientation) when missing.
func ensureStructEmb(refs []*Feat, sn *StructNet) {
	if sn == nil {
		return
	}
	for _, r := range refs {
		if r == nil || len(r.Emb) == snEmb {
			continue
		}
		_, _, r.Emb = sn.embed(r.Thumb16)
	}
}

// embedEmbed runs embed and returns only the embedding.
func (s *StructNet) embedEmb(x []float64) []float64 {
	_, _, e := s.embed(x)
	return e
}

// embedEmbRot runs the rotation-pooled query embedding and returns only it.
func (s *StructNet) embedEmbRot(x []float64) []float64 {
	_, _, e, _ := s.embedRot(x)
	return e
}

// embed computes the single-orientation structure embedding of the 16x16 grid
// x. It returns the conv-pool2 features F, the MLP hidden activations mh and
// the final embedding emb; backward() recomputes the conv path from x, so no
// intermediate feature maps need to be retained.
func (s *StructNet) embed(x []float64) (F, mh, emb []float64) {
	z1 := conv3(s.W1, s.B1, x, 1, snGrid, snC1)
	reluArr(z1)
	p1 := pool2x2(z1, snGrid, snC1)
	z2 := conv3(s.W2, s.B2, p1, snC1, 8, snC2)
	reluArr(z2)
	F = pool2x2(z2, 8, snC2)
	mh = make([]float64, snHid)
	for j := 0; j < snHid; j++ {
		v := s.B3[j]
		row := s.W3[j*snFMap : (j+1)*snFMap]
		for i := 0; i < snFMap; i++ {
			v += row[i] * F[i]
		}
		mh[j] = relu(v)
	}
	emb = make([]float64, snEmb)
	for j := 0; j < snEmb; j++ {
		v := s.B4[j]
		row := s.W4[j*snHid : (j+1)*snHid]
		for i := 0; i < snHid; i++ {
			v += row[i] * mh[i]
		}
		emb[j] = v
	}
	return F, mh, emb
}

// embedRot computes the rotation-pooled query embedding: pooled[d] = max over
// the 4 orientations of the per-orientation embedding. Returns Fs/mhs per
// orientation (for backward) and the winning orientation per embedding dim.
func (s *StructNet) embedRot(x []float64) (Fs, mhs [][]float64, pooled []float64, wins []int) {
	Fs = make([][]float64, snRot)
	mhs = make([][]float64, snRot)
	perEmb := make([][]float64, snRot)
	for k := 0; k < snRot; k++ {
		Fs[k], mhs[k], perEmb[k] = s.embed(rot16(x, k))
	}
	pooled = make([]float64, snEmb)
	wins = make([]int, snEmb)
	for d := 0; d < snEmb; d++ {
		m := perEmb[0][d]
		w := 0
		for k := 1; k < snRot; k++ {
			if perEmb[k][d] > m {
				m = perEmb[k][d]
				w = k
			}
		}
		pooled[d] = m
		wins[d] = w
	}
	return Fs, mhs, pooled, wins
}

// rot16 rotates the 16x16 grid k*90° clockwise.
func rot16(x []float64, k int) []float64 {
	out := make([]float64, snGrid*snGrid)
	for r := 0; r < snGrid; r++ {
		for c := 0; c < snGrid; c++ {
			var sr, sc int
			switch k {
			case 0:
				sr, sc = r, c
			case 1:
				sr, sc = c, snGrid-1-r
			case 2:
				sr, sc = snGrid-1-r, snGrid-1-c
			default:
				sr, sc = snGrid-1-c, r
			}
			out[r*snGrid+c] = x[sr*snGrid+sc]
		}
	}
	return out
}

// conv3 runs a 3x3 stride-1 pad-1 convolution (no activation).
// in: chIn x H x H (ch-major); out: chOut x H x H. Weight W is chOut x chIn x 3 x 3.
func conv3(W, B []float64, in []float64, chIn, H, chOut int) []float64 {
	out := make([]float64, chOut*H*H)
	for fo := 0; fo < chOut; fo++ {
		for y := 0; y < H; y++ {
			for xx := 0; xx < H; xx++ {
				var v float64
				for fi := 0; fi < chIn; fi++ {
					for dy := 0; dy < 3; dy++ {
						iy := y - 1 + dy
						if iy < 0 || iy >= H {
							continue
						}
						for dx := 0; dx < 3; dx++ {
							ix := xx - 1 + dx
							if ix < 0 || ix >= H {
								continue
							}
							v += W[(fo*chIn+fi)*9+dy*3+dx] * in[fi*H*H+iy*H+ix]
						}
					}
				}
				out[fo*H*H+y*H+xx] = v + B[fo]
			}
		}
	}
	return out
}

// dConv3 accumulates conv weight gradients given the output gradient dOut and
// the pre-activation input in; returns the input gradient dIn.
func dConv3(W, gW, gB []float64, in []float64, chIn, H, chOut int, dOut []float64) []float64 {
	dIn := make([]float64, len(in))
	for fo := 0; fo < chOut; fo++ {
		for y := 0; y < H; y++ {
			for xx := 0; xx < H; xx++ {
				d := dOut[fo*H*H+y*H+xx]
				gB[fo] += d
				for fi := 0; fi < chIn; fi++ {
					for dy := 0; dy < 3; dy++ {
						iy := y - 1 + dy
						if iy < 0 || iy >= H {
							continue
						}
						for dx := 0; dx < 3; dx++ {
							ix := xx - 1 + dx
							if ix < 0 || ix >= H {
								continue
							}
							wIdx := (fo*chIn+fi)*9 + dy*3 + dx
							gW[wIdx] += d * in[fi*H*H+iy*H+ix]
							dIn[fi*H*H+iy*H+ix] += d * W[wIdx]
						}
					}
				}
			}
		}
	}
	return dIn
}

// pool2x2 max-pools a ch-major H x H map non-overlapping, yielding ph x ph.
func pool2x2(in []float64, H, ch int) []float64 {
	ph := H / 2
	out := make([]float64, ch*ph*ph)
	for c := 0; c < ch; c++ {
		base := c * H * H
		for y := 0; y < ph; y++ {
			for xx := 0; xx < ph; xx++ {
				o := base + 2*y*H + 2*xx
				m := in[o]
				if in[o+1] > m {
					m = in[o+1]
				}
				if in[o+H] > m {
					m = in[o+H]
				}
				if in[o+H+1] > m {
					m = in[o+H+1]
				}
				out[c*ph*ph+y*ph+xx] = m
			}
		}
	}
	return out
}

// dPool2x2 back-propagates dOut through a max pool whose input values are in.
func dPool2x2(in, dOut []float64, H, ch int) []float64 {
	ph := H / 2
	dIn := make([]float64, H*H*ch)
	for c := 0; c < ch; c++ {
		base := c * H * H
		for y := 0; y < ph; y++ {
			for xx := 0; xx < ph; xx++ {
				o := base + 2*y*H + 2*xx
				offs := [4]int{0, 1, H, H + 1}
				mi, mv := 0, in[o]
				for k := 1; k < 4; k++ {
					if in[o+offs[k]] > mv {
						mv = in[o+offs[k]]
						mi = k
					}
				}
				dIn[o+offs[mi]] += dOut[c*ph*ph+y*ph+xx]
			}
		}
	}
	return dIn
}

func reluArr(v []float64) {
	for i := range v {
		if v[i] < 0 {
			v[i] = 0
		}
	}
}

// backward accumulates gradient wrt the CNN/MLP weights for one grid given the
// embedding gradient demb and the cached F/mh (the conv path is recomputed from
// x).
func (s *StructNet) backward(demb []float64, x []float64, F, mh []float64, g *snGrads) {
	// MLP: emb = W4·mh + B4, mh = relu(W3·F + B3)
	dh := make([]float64, snHid)
	for j := 0; j < snHid; j++ {
		if mh[j] <= 0 {
			continue
		}
		for i := 0; i < snEmb; i++ {
			dh[j] += demb[i] * s.W4[i*snHid+j]
		}
	}
	dF := make([]float64, snFMap)
	for j := 0; j < snHid; j++ {
		g.B3[j] += dh[j]
		row := s.W3[j*snFMap : (j+1)*snFMap]
		for k := 0; k < snFMap; k++ {
			g.W3[j*snFMap+k] += dh[j] * F[k]
			dF[k] += dh[j] * row[k]
		}
	}
	for i := 0; i < snEmb; i++ {
		g.B4[i] += demb[i]
		for j := 0; j < snHid; j++ {
			g.W4[i*snHid+j] += demb[i] * mh[j]
		}
	}

	// conv path (recomputed): conv1 -> relu -> pool -> conv2 -> relu -> pool -> F
	z1r := conv3(s.W1, s.B1, x, 1, snGrid, snC1)
	reluArr(z1r)
	p1 := pool2x2(z1r, snGrid, snC1)
	z2r := conv3(s.W2, s.B2, p1, snC1, 8, snC2)
	reluArr(z2r)

	dz2 := dPool2x2(z2r, dF, 8, snC2)
	for i := range dz2 {
		if z2r[i] == 0 {
			dz2[i] = 0
		}
	}
	dp1 := dConv3(s.W2, g.W2, g.B2, p1, snC1, 8, snC2, dz2)
	dz1 := dPool2x2(z1r, dp1, snGrid, snC1)
	for i := range dz1 {
		if z1r[i] == 0 {
			dz1[i] = 0
		}
	}
	dConv3(s.W1, g.W1, g.B1, x, 1, snGrid, snC1, dz1)
}

// backwardRot back-propagates the pooled embedding gradient demb through the
// rotation pool into the winning orientations' conv paths.
func (s *StructNet) backwardRot(demb, x []float64, Fs, mhs [][]float64, wins []int, g *snGrads) {
	byOrient := make([][]int, snRot)
	for d, w := range wins {
		byOrient[w] = append(byOrient[w], d)
	}
	for k := 0; k < snRot; k++ {
		if len(byOrient[k]) == 0 {
			continue
		}
		d1 := make([]float64, snEmb)
		for _, d := range byOrient[k] {
			d1[d] = demb[d]
		}
		s.backward(d1, rot16(x, k), Fs[k], mhs[k], g)
	}
}

// snGrads holds the accumulated CNN/MLP gradient.
type snGrads struct {
	W1, B1 []float64
	W2, B2 []float64
	W3, B3 []float64
	W4, B4 []float64
}

func newSNGrads(s *StructNet) *snGrads {
	return &snGrads{
		W1: make([]float64, len(s.W1)), B1: make([]float64, len(s.B1)),
		W2: make([]float64, len(s.W2)), B2: make([]float64, len(s.B2)),
		W3: make([]float64, len(s.W3)), B3: make([]float64, len(s.B3)),
		W4: make([]float64, len(s.W4)), B4: make([]float64, len(s.B4)),
	}
}