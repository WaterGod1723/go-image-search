package main

import (
	"math"
	"math/cmplx"
	"sort"
	"sync"
)

const (
	hBins = 16
	sBins = 6
	vBins = 3
	nRing = 16
	nAng  = 24
	nFreq = 12
	znMax = 12
	maskN = 64
)

// zernModes collects all (n, m) pairs used: n in 0..znMax, m >= 0 step 2
// (|Z_{n,m}| == |Z_{n,-m}| so keep non-negative orders only).
var zernModes = buildZernModes()
var zernTerms = makeZernTerms(zernModes)

type zernMode struct{ n, m int }

func buildZernModes() []zernMode {
	var out []zernMode
	for n := 0; n <= znMax; n++ {
		for m := 0; m <= n; m += 2 {
			out = append(out, zernMode{n, m})
		}
	}
	return out
}

// zernTerm is one term c*r^e of a Zernike radial polynomial.
type zernTerm struct {
	c float64
	e int
}

// makeZernTerms returns the radial polynomial terms of every mode. Evaluation
// avoids math.Pow per term: the caller precomputes the powers r^0..r^znMax once
// per pixel, so each term costs a single multiply-add instead of a full exp/log
// (math.Pow is ~100-300ns each; this was the dominant cost of buildFeat).
func makeZernTerms(modes []zernMode) [][]zernTerm {
	out := make([][]zernTerm, len(modes))
	for i, md := range modes {
		n, m := md.n, md.m
		am := m
		if am < 0 {
			am = -am
		}
		// P_nm(r) = sum_{s=0}^{(n-|m|)/2} (-1)^s (n-s)!/(s! ((n+|m|)/2-s)! ((n-|m|)/2-s)!) r^(n-2s)
		var terms []zernTerm
		for s := 0; s <= (n-am)/2; s++ {
			c := float64(fact(n-s))
			c /= float64(fact(s) * fact((n+am)/2-s) * fact((n-am)/2-s))
			if s%2 == 1 {
				c = -c
			}
			terms = append(terms, zernTerm{c, n - 2*s})
		}
		out[i] = terms
	}
	return out
}

func fact(k int) int {
	f := 1
	for z := 2; z <= k; z++ {
		f *= z
	}
	return f
}

// Feat holds the descriptor set for one sprite.
type Feat struct {
	Hist       []float64 // HSV histogram [H][S][V] L1-norm
	Radial     []float64 // density per normalized ring, L1
	AngMag     []float64 // per-ring angular FFT magnitudes
	PolarShape []float64 // ring x sector counts, per-ring L1
	PolarCol   []float64 // ring x sector mean RGB
	Zern       []float64 // |Z_nm| Zernike magnitudes, rotation-invariant, L2
	Mask48     []float64 // 48x48 binary mask, centroid-centered, RMS-scaled
	N          int
	Mono       bool
	ColorProj  []float64 // 32x32 color projection histogram (row+col mean RGB)
	Sx, Sy     []float64 // sprite points scaled to Mask48 grid (centroid orgin)
	RMS        float64   // root-mean-square radius of sprite pixels
	MinX, MinY int
	MaxX, MaxY int

	// lazy, query-independent shape-context data for references (see refSC);
	// unexported so gob skips them when the server index is cached.
	scOnce sync.Once
	scPts  []scPoint
	scHist []float64
}

func hsv(r, g, b uint8) (h, s, v float64) {
	fr, fg, fb := float64(r)/255, float64(g)/255, float64(b)/255
	mx := math.Max(fr, math.Max(fg, fb))
	mn := math.Min(fr, math.Min(fg, fb))
	d := mx - mn
	v = mx
	if mx > 0 {
		s = d / mx
	}
	if d > 0 {
		switch {
		case mx == fr:
			h = math.Mod((fg-fb)/d, 6)
		case mx == fg:
			h = (fb-fr)/d + 2
		default:
			h = (fr-fg)/d + 4
		}
		h *= 60
		if h < 0 {
			h += 360
		}
	}
	return h, s, v
}

func l1norm(v []float64) {
	var s float64
	for _, x := range v {
		s += x
	}
	if s <= 0 {
		return
	}
	for i := range v {
		v[i] /= s
	}
}

func l2norm(v []float64) {
	var s float64
	for _, x := range v {
		s += x * x
	}
	if s <= 0 {
		return
	}
	n := 1 / math.Sqrt(s)
	for i := range v {
		v[i] *= n
	}
}

// buildFeat computes the descriptor for a sprite pixel set (already isolated).
func buildFeat(px []Px) *Feat {
	f := &Feat{
		Hist:       make([]float64, hBins*sBins*vBins),
		Radial:     make([]float64, nRing),
		AngMag:     make([]float64, nRing*nFreq),
		PolarShape: make([]float64, nRing*nAng),
		PolarCol:   make([]float64, nRing*nAng*3),
		Zern:       make([]float64, len(zernModes)),
	}
	f.N = len(px)
	if f.N == 0 {
		return f
	}

	// centroid + rms + maxR
	var sx, sy float64
	f.MinX, f.MinY = 1<<30, 1<<30
	f.MaxX, f.MaxY = -1, -1
	for _, p := range px {
		sx += float64(p.X)
		sy += float64(p.Y)
		if p.X < f.MinX {
			f.MinX = p.X
		}
		if p.X > f.MaxX {
			f.MaxX = p.X
		}
		if p.Y < f.MinY {
			f.MinY = p.Y
		}
		if p.Y > f.MaxY {
			f.MaxY = p.Y
		}
	}
	cx := sx / float64(f.N)
	cy := sy / float64(f.N)
	var r2sum, rmax float64
	for _, p := range px {
		dx, dy := float64(p.X)-cx, float64(p.Y)-cy
		r := math.Hypot(dx, dy)
		r2sum += dx*dx + dy*dy
		if r > rmax {
			rmax = r
		}
	}
	rms := math.Sqrt(r2sum / float64(f.N))
	f.RMS = rms
	if rms <= 0 {
		return f
	}

	// 1) HSV histogram
	var satPix float64
	for _, p := range px {
		h, s, v := hsv(p.R, p.G, p.B)
		hi := int(h / 360 * hBins)
		if hi >= hBins {
			hi = hBins - 1
		}
		si := int(s * sBins)
		if si >= sBins {
			si = sBins - 1
		}
		vi := int(v * vBins)
		if vi >= vBins {
			vi = vBins - 1
		}
		f.Hist[(hi*sBins+si)*vBins+vi]++
		if s > 0.22 && v > 0.15 {
			satPix++
		}
	}
	l1norm(f.Hist)
	if float64(f.N) > 0 && satPix < 0.05*float64(f.N) {
		f.Mono = true
	}

	// 2) polar: ring x sector histograms and their mean colors
	cellCnt := make([]int, nRing*nAng)
	polarR := make([]float64, nRing*nAng)   // per-ring normalized counts
	colAcc := make([]float64, nRing*nAng*3) // RGB sums per cell
	for _, p := range px {
		dx, dy := float64(p.X)-cx, float64(p.Y)-cy
		r := math.Hypot(dx, dy)
		ri := int(r / rms * nRing)
		if ri >= nRing {
			ri = nRing - 1
		}
		ai := int((math.Atan2(dy, dx)*(180/math.Pi) + 180) / 360 * nAng)
		if ai >= nAng {
			ai = nAng - 1
		}
		cell := ri*nAng + ai
		polarR[cell]++
		cellCnt[cell]++
		f.Radial[ri]++
		colAcc[cell*3] += float64(p.R) / 255
		colAcc[cell*3+1] += float64(p.G) / 255
		colAcc[cell*3+2] += float64(p.B) / 255
	}
	l1norm(f.Radial)

	// per-ring normalize polar shape
	for ri := 0; ri < nRing; ri++ {
		var s float64
		for a := 0; a < nAng; a++ {
			s += polarR[ri*nAng+a]
		}
		if s > 0 {
			for a := 0; a < nAng; a++ {
				polarR[ri*nAng+a] /= s
			}
		}
	}
	copy(f.PolarShape, polarR)

	// mean colors
	for c := 0; c < nRing*nAng; c++ {
		if cellCnt[c] == 0 {
			continue
		}
		f.PolarCol[c*3] = colAcc[c*3] / float64(cellCnt[c])
		f.PolarCol[c*3+1] = colAcc[c*3+1] / float64(cellCnt[c])
		f.PolarCol[c*3+2] = colAcc[c*3+2] / float64(cellCnt[c])
	}

	// 3) per-ring angular FFT magnitudes
	for ri := 0; ri < nRing; ri++ {
		var s float64
		for a := 0; a < nAng; a++ {
			s += f.PolarShape[ri*nAng+a]
		}
		if s <= 0 {
			continue
		}
		for k := 1; k <= nFreq; k++ {
			var acc complex128
			for a := 0; a < nAng; a++ {
				acc += complex(f.PolarShape[ri*nAng+a], 0) *
					cmplx.Exp(complex(0, -2*math.Pi*float64(k)*float64(a)/nAng))
			}
			f.AngMag[ri*nFreq+k-1] = cmplx.Abs(acc)
		}
		l1norm(f.AngMag[ri*nFreq : (ri+1)*nFreq])
	}
	l1norm(f.AngMag)

	// 4) Zernike moments on binary mask. Per pixel the powers of r are
	// precomputed once (cheap multiply chain) and cos(m*t)/sin(m*t) are built
	// from cos(2t)/sin(2t) by angle addition, avoiding per-term math.Pow and
	// per-mode trig — this loop dominated buildFeat cost before.
	zacc := make([]complex128, len(zernModes))
	scale := 1.0 / rmax
	var pow [znMax + 1]float64
	for _, p := range px {
		dx, dy := float64(p.X)-cx, float64(p.Y)-cy
		r := math.Hypot(dx, dy) * scale
		t := math.Atan2(dy, dx)
		pow[0] = 1
		for e := 1; e <= znMax; e++ {
			pow[e] = pow[e-1] * r
		}
		ct, st := math.Cos(t), math.Sin(t)
		c2t := 2*ct*ct - 1
		s2t := 2 * st * ct
		var cosK, sinK [znMax/2 + 1]float64
		cosK[0], sinK[0] = 1, 0
		for k := 1; k <= znMax/2; k++ {
			cosK[k] = cosK[k-1]*c2t - sinK[k-1]*s2t
			sinK[k] = sinK[k-1]*c2t + cosK[k-1]*s2t
		}
		for i, md := range zernModes {
			terms := zernTerms[i]
			var poly float64
			for _, tm := range terms {
				poly += tm.c * pow[tm.e]
			}
			k := md.m / 2
			zacc[i] += complex(poly*cosK[k], poly*sinK[k])
		}
	}
	for i := range f.Zern {
		f.Zern[i] = cmplx.Abs(zacc[i])
	}
	l2norm(f.Zern)

	// 5) 64x64 soft mask: centroid -> center. Scale by the 98th percentile radius
	// (rotation-invariant, unlike bbox/rmax) times 0.45*half, so the sprite
	// comfortably fits the grid for all sprites.
	f.Mask48 = make([]float64, maskN*maskN)
	mh := maskN / 2
	rad := make([]float64, f.N)
	for i, p := range px {
		rad[i] = math.Hypot(float64(p.X)-cx, float64(p.Y)-cy)
	}
	sort.Float64s(rad)
	s := 0.45 * float64(mh) / rad[int(0.98*float64(f.N))]
	if s <= 0 || math.IsInf(s, 0) {
		return f
	}
	f.Sx = make([]float64, f.N)
	f.Sy = make([]float64, f.N)
	for i, p := range px {
		dx := float64(p.X) - cx
		dy := float64(p.Y) - cy
		x := int(dx*s) + mh
		y := int(dy*s) + mh
		if x >= 0 && x < maskN && y >= 0 && y < maskN {
			if p.A > f.Mask48[y*maskN+x] {
				f.Mask48[y*maskN+x] = p.A
			}
		}
		f.Sx[i] = dx * s
		f.Sy[i] = dy * s
	}
	f.ColorProj = buildColorProj(px)
	return f
}

// buildColorProj builds the 32x32 color projection histogram of a sprite:
// the sprite's bounding box is padded to a square (letterboxed), resized to
// 32x32, then the mean RGB of each row and each column is recorded. Output is
// 32*3 + 32*3 = 192 floats: [row0.R row0.G row0.B ... row31] then
// [col0.R col0.G col0.B ... col31]. This is the query's static spatial color
// layout — a target occupying only part of a composite image leaves a
// distinctive row/column color signature.
func buildColorProj(px []Px) []float64 {
	if len(px) == 0 {
		return nil
	}
	minX, minY := px[0].X, px[0].Y
	maxX, maxY := px[0].X, px[0].Y
	for _, p := range px {
		if p.X < minX {
			minX = p.X
		}
		if p.X > maxX {
			maxX = p.X
		}
		if p.Y < minY {
			minY = p.Y
		}
		if p.Y > maxY {
			maxY = p.Y
		}
	}
	w := maxX - minX + 1
	h := maxY - minY + 1
	side := w
	if h > side {
		side = h
	}
	if side <= 0 {
		return nil
	}
	// map each pixel into the padded square's 32x32 grid
	cell := make([][3]float64, 32*32)
	cnt := make([]float64, 32*32)
	offX := float64(side-w) / 2
	offY := float64(side-h) / 2
	for _, p := range px {
		fx := (float64(p.X-minX) + offX) / float64(side) * 32
		fy := (float64(p.Y-minY) + offY) / float64(side) * 32
		gx := int(fx)
		gy := int(fy)
		if gx < 0 {
			gx = 0
		}
		if gx > 31 {
			gx = 31
		}
		if gy < 0 {
			gy = 0
		}
		if gy > 31 {
			gy = 31
		}
		idx := gy*32 + gx
		cell[idx][0] += float64(p.R) / 255
		cell[idx][1] += float64(p.G) / 255
		cell[idx][2] += float64(p.B) / 255
		cnt[idx]++
	}
	out := make([]float64, 0, 32*3*2)
	// row projections: mean RGB of occupied cells in each row
	for gy := 0; gy < 32; gy++ {
		var r, g, b, n float64
		for gx := 0; gx < 32; gx++ {
			idx := gy*32 + gx
			if cnt[idx] == 0 {
				continue
			}
			r += cell[idx][0] / cnt[idx]
			g += cell[idx][1] / cnt[idx]
			b += cell[idx][2] / cnt[idx]
			n++
		}
		if n == 0 {
			out = append(out, 0, 0, 0)
			continue
		}
		out = append(out, r/n, g/n, b/n)
	}
	// column projections
	for gx := 0; gx < 32; gx++ {
		var r, g, b, n float64
		for gy := 0; gy < 32; gy++ {
			idx := gy*32 + gx
			if cnt[idx] == 0 {
				continue
			}
			r += cell[idx][0] / cnt[idx]
			g += cell[idx][1] / cnt[idx]
			b += cell[idx][2] / cnt[idx]
			n++
		}
		if n == 0 {
			out = append(out, 0, 0, 0)
			continue
		}
		out = append(out, r/n, g/n, b/n)
	}
	return out
}