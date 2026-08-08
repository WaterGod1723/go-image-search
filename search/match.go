package main

import (
	"fmt"
	"math"
	"os"
)

func cosSim(a, b []float64) float64 {
	var d1, d2, dp float64
	for i := range a {
		d1 += a[i] * a[i]
		d2 += b[i] * b[i]
		dp += a[i] * b[i]
	}
	if d1 <= 0 || d2 <= 0 {
		return 0
	}
	return dp / math.Sqrt(d1*d2)
}

func histSim(a, b []float64) float64 {
	var s float64
	for i := range a {
		if a[i] < b[i] {
			s += a[i]
		} else {
			s += b[i]
		}
	}
	return s
}

// polarShiftSim: global circular-shift search over all rings; rotation maps to
// the same integer shift on every ring, so report the best mean cosine.
func polarShiftSim(q, r *Feat) float64 {
	best := -1E18
	for k := -nAng; k <= nAng; k++ {
		var acc float64
		n := 0
		for ri := 0; ri < nRing; ri++ {
			qb := q.PolarShape[ri*nAng : (ri+1)*nAng]
			rb := r.PolarShape[ri*nAng : (ri+1)*nAng]
			var dq, dr, dp float64
			for a := 0; a < nAng; a++ {
				src := (a - k%nAng + nAng) % nAng
				dq += qb[a] * qb[a]
				dr += rb[src] * rb[src]
				dp += qb[a] * rb[src]
			}
			if dq <= 0 || dr <= 0 {
				continue
			}
			acc += dp / math.Sqrt(dq*dr)
			n++
		}
		if n > 0 {
			v := acc / float64(n)
			if v > best {
				best = v
			}
		}
	}
	return math.Max(0, best)
}

// polarColSim matches mean colors ring-by-ring after the best global shift.
func polarColSim(q, r *Feat) float64 {
	best := -1E18
	for k := -nAng; k <= nAng; k++ {
		acc := 0.0
		n := 0
		for ri := 0; ri < nRing; ri++ {
			qb := q.PolarCol[ri*nAng*3 : (ri+1)*nAng*3]
			rb := r.PolarCol[ri*nAng*3 : (ri+1)*nAng*3]
			var dq, dr, dp float64
			for a := 0; a < nAng; a++ {
				src := (a - k%nAng + nAng) % nAng
				for c := 0; c < 3; c++ {
					qv := qb[a*3+c]
					rv := rb[src*3+c]
					dq += qv * qv
					dr += rv * rv
					dp += qv * rv
				}
			}
			if dq <= 0 || dr <= 0 {
				continue
			}
			acc += dp / math.Sqrt(dq*dr)
			n++
		}
		if n > 0 {
			v := acc / float64(n)
			if v > best {
				best = v
			}
		}
	}
	return math.Max(0, best)
}

func geomSim(q, r *Feat) float64 {
	// fallback: Zernike n=0 magnitude encodes fill-level
	return cosSim(q.Zern, r.Zern)
}

const nSpaces = 7

// scores returns the 7 feature similarities for query q and ref r:
//  0 color hist intersection
//  1 radial cosine
//  2 angular FFT magnitudes cosine
//  3 polar shape (best shift)
//  4 polar color layout (best shift)
//  5 geometry (Zern n=0 ratio)
//  6 Zernike magnitudes cosine
func scores(q, r *Feat) []float64 {
	return []float64{
		histSim(q.Hist, r.Hist),
		cosSim(q.Radial, r.Radial),
		cosSim(q.AngMag, r.AngMag),
		polarShiftSim(q, r),
		polarColSim(q, r),
		roundTripFactor(q, r),
		cosSim(q.Zern, r.Zern),
	}
}

// roundTripFactor grades overall stickiness: mean color + shape together.
func roundTripFactor(q, r *Feat) float64 {
	return 0.5*cosSim(q.Radial, r.Radial) + 0.5*cosSim(q.AngMag, r.AngMag)
}

// composite ranks refs for query q.
//
// Stage 1: histogram groups that every ref whose normalized histogram differs
// from the query by less than eps is "in the running". Stage 2 resolves the
// group by the strongest discriminators (polar shape for gray sprites;
// color-layout for colorful), while refs clearly outside the group stay low.
func composite(q *Feat, refs []*Feat, weights []float64) []int {
	s := make([][]float64, len(refs))
	for i, r := range refs {
		s[i] = scores(q, r)
	}
	w := weights // [hist, radial, angmag, shape, color, geom]

	// find best histogram score
	bestHist := -1.0
	for i := range refs {
		if s[i][0] > bestHist {
			bestHist = s[i][0]
		}
	}

	// Group = refs within a soft margin of the best histogram. For gray/gray
	// families the histogram run hits the same value and the whole mono set
	// stays in group, letting shape decide; for colorful sprites the group is
	// small (unique color).
	groupEps := math.Max(0.015, (bestHist-0.7)*0.15)
	inGroup := make([]bool, len(refs))
	for i := range refs {
		inGroup[i] = s[i][0] >= bestHist-groupEps
	}

	// within the group, weighted blend (hist veto + discriminator fields) and
// then refine by brute-force mask rotation IoU
	score := make([]float64, len(refs))
	for i := range refs {
		if !inGroup[i] {
			score[i] = -1
			continue
		}
		v := w[0]*s[i][0] + w[1]*s[i][1] + w[2]*s[i][2] +
			w[3]*s[i][3] + w[4]*s[i][4] + w[5]*s[i][5]
		score[i] = v
	}
	// but ensure any in-group candidate beats all out-of-group
	for i := range refs {
		if score[i] < 0 && inGroup[i] {
			score[i] = 0
		}
	}

	// refine: re-rank the in-group set by exact mask rotation match.
	// This breaks gray-glyph histogram ties that shape/color features cannot.
	if len(inGroup) > 0 {
		refine := make([]float64, len(refs))
		top := make([]int, 0, len(refs))
		for i := range refs {
			if inGroup[i] {
				top = append(top, i)
			}
		}
		// pick the strongest histogram candidates for the expensive sweep
		insertionSort(top, func(a, b int) bool { return s[a][0] > s[b][0] })
		if len(top) > 6 {
			top = top[:6]
		}
		for _, i := range top {
			refine[i] = maskScore(q, refs[i])
		}
		// blend: histogram + mask; mask is decisive
		for _, i := range top {
			score[i] = 0.25*score[i] + 0.75*refine[i]
		}
	}

	out := make([]int, len(refs))
	for i := range out {
		out[i] = i
	}
	insertionSort(out, func(a, b int) bool {
		// histogram "family" first (anything in group > anything outside)
		if inGroup[a] != inGroup[b] {
			return inGroup[a]
		}
		return score[a] > score[b]
	})
	return out
}

// compositeAdaptive ranks refs for query q using score-distribution-driven
// fusion instead of fixed weights. For each feature dimension we estimate its
// discriminative power for THIS query as how far the best score separates from
// the rest of the pack; dimensions that are tied (e.g. identical gray
// histograms) get near-zero weight while dimensions that clearly separate the
// best candidate dominate. This replaces hand-tuned defaultWeights with a
// principled, per-query adaptive scheme.
func compositeAdaptive(q *Feat, refs []*Feat) []int {
	s := make([][]float64, len(refs))
	for i, r := range refs {
		s[i] = scores(q, r)
	}
	nD := nSpaces
	w := make([]float64, nD)

	// Per-dimension discrimination: sorted descending scores; weight = how far
	// the winner is above the runner-up, scaled by the median absolute spread.
	idx := make([]int, len(refs))
	for d := 0; d < nD; d++ {
		for i := range idx {
			idx[i] = i
		}
		insertionSort(idx, func(a, b int) bool { return s[a][d] > s[b][d] })
		if len(refs) == 0 {
			continue
		}
		// robust "separation": (best - runnerUp) / (spread among top few)
		best := s[idx[0]][d]
		var others float64
		n := 0
		for _, i := range idx[1:] {
			others += s[i][d]
			n++
		}
		if n == 0 {
			w[d] = 1
			continue
		}
		meanOther := others / float64(n)
		// discrimination = gap between best and the pack, in units of the
		// pack's spread so dimensions with wide innate ranges are comparable.
		gap := best - meanOther
		var spread float64
		for _, i := range idx {
			spread += (s[i][d] - meanOther) * (s[i][d] - meanOther)
		}
		spread = math.Sqrt(spread / float64(len(idx)))
		// shift so tied dimensions (gap~0, spread~0) drop to ~0
		if spread > 0 && gap > 0 {
			w[d] = gap / (spread + 1e-9)
		} else if gap > 0 {
			w[d] = gap
		} else {
			w[d] = 0
		}
	}

	// Normalize weights to sum to 1; a flat all-zero profile (every dim tied)
	// falls back to uniform weighting.
	var wsum float64
	for d := 0; d < nD; d++ {
		wsum += w[d]
	}
	if wsum <= 1e-9 {
		for d := 0; d < nD; d++ {
			w[d] = 1 / float64(nD)
		}
	} else {
		for d := 0; d < nD; d++ {
			w[d] /= wsum
		}
	}

	score := make([]float64, len(refs))
	for i := range refs {
		var v float64
		for d := 0; d < nD; d++ {
			v += w[d] * s[i][d]
		}
		score[i] = v
	}

	// Optional histogram-family gate: keep the adaptive dimension weighting
	// but never let a wrong-color family outrank the histogram winner's family.
	bestHist := -1.0
	for i := range refs {
		if s[i][0] > bestHist {
			bestHist = s[i][0]
		}
	}
	groupEps := math.Max(0.015, (bestHist-0.7)*0.15)
	inGroup := make([]bool, len(refs))
	for i := range refs {
		inGroup[i] = s[i][0] >= bestHist-groupEps
	}

	// Refine the in-group winners with the exact mask rotation match (breaks
	// gray-glyph histogram ties that even the adaptive shape dims struggle with).
	refineW := 0.0
	if v := os.Getenv("REFINE"); v != "" {
		fmt.Sscanf(v, "%f", &refineW)
	}
	if refineW > 0 {
		top := make([]int, 0, len(refs))
		for i := range refs {
			if inGroup[i] {
				top = append(top, i)
			}
		}
		insertionSort(top, func(a, b int) bool { return s[a][0] > s[b][0] })
		if len(top) > 6 {
			top = top[:6]
		}
		for _, i := range top {
			score[i] += refineW * maskScore(q, refs[i])
		}
	}

	// Shape-context refinement: among the in-group candidates, re-rank by
	// point-based shape matching which survives thin-stroke degradation better
	// than grid occupancy. Default weight 0.5; override with SCREFINE env.
	scW := 0.5
	if v := os.Getenv("SCREFINE"); v != "" {
		fmt.Sscanf(v, "%f", &scW)
	}
	if scW > 0 {
		top := make([]int, 0, len(refs))
		for i := range refs {
			if inGroup[i] {
				top = append(top, i)
			}
		}
		insertionSort(top, func(a, b int) bool { return s[a][0] > s[b][0] })
		if len(top) > 8 {
			top = top[:8]
		}
		for _, i := range top {
			score[i] = (1-scW)*score[i] + scW*shapeContextSim(q, refs[i])
		}
	}

	// tiebreak mode: keep adaptive ordering but use maskScore only to reorder
	// candidates whose adaptive scores are within a small band (ties).
	tiebreak := os.Getenv("TIEBREAK") == "1"
	if tiebreak {
		band := 0.02
		top := make([]int, 0, len(refs))
		for i := range refs {
			if inGroup[i] {
				top = append(top, i)
			}
		}
		insertionSort(top, func(a, b int) bool { return score[a] > score[b] })
		if len(top) > 8 {
			top = top[:8]
		}
		if len(top) > 0 {
			ref := score[top[0]]
			var inBand []int
			for _, i := range top {
				if ref-score[i] <= band {
					inBand = append(inBand, i)
				}
			}
			if len(inBand) > 1 {
				insertionSort(inBand, func(a, b int) bool { return maskScore(q, refs[a]) > maskScore(q, refs[b]) })
				// splice reordered band back into top
				for j, i := range inBand {
					top[j] = i
				}
				// apply the band ordering over the full score list by giving a tiny
				// rank bonus proportional to mask, so relative band order survives.
				for j, i := range inBand {
					_ = j
					score[i] += float64(len(inBand)-j) * 1e-6
				}
			}
		}
	}

	out := make([]int, len(refs))
	for i := range out {
		out[i] = i
	}
	insertionSort(out, func(a, b int) bool {
		if inGroup[a] != inGroup[b] {
			return inGroup[a]
		}
		return score[a] > score[b]
	})
	return out
}

func insertionSort(arr []int, less func(a, b int) bool) {
	for i := 1; i < len(arr); i++ {
		for j := i; j > 0 && less(arr[j], arr[j-1]); j-- {
			arr[j], arr[j-1] = arr[j-1], arr[j]
		}
	}
}

// rotateMask resamples a maskN x maskN mask rotated by theta degrees
// (positive = CCW), bilinear interpolation, precomputed lookup-free inline.
func rotateMask(src []float64, theta float64, dst []float64) {
	c := 0.5 * (maskN - 1)
	cs, sn := math.Cos(theta*math.Pi/180), math.Sin(theta*math.Pi/180)
	for y := 0; y < maskN; y++ {
		dy := float64(y) - c
		for x := 0; x < maskN; x++ {
			dx := float64(x) - c
			sx := dx*cs + dy*sn + c
			sy := -dx*sn + dy*cs + c
			x0 := int(sx)
			y0 := int(sy)
			if x0 < 0 || y0 < 0 || x0 >= maskN-1 || y0 >= maskN-1 {
				dst[y*maskN+x] = 0
				continue
			}
			fx := sx - float64(x0)
			fy := sy - float64(y0)
			a := src[y0*maskN+x0]*(1-fx) + src[y0*maskN+x0+1]*fx
			bb := src[(y0+1)*maskN+x0]*(1-fx) + src[(y0+1)*maskN+x0+1]*fx
			dst[y*maskN+x] = a*(1-fy) + bb*fy
		}
	}
}

// maskScoreGrid returns best Dice overlap over a rotation sweep at grid size g
// (16/32/64). Coarse grids ignore sub-grid stroke detail, so under-extracted
// thin strokes in queries (checkmark, receiver) hurt less.
func maskScoreGrid(q, r *Feat, g int) float64 {
	if g != maskN {
		// downsample both masks to g x g by block-averaging
		qa := make([]float64, g*g)
		rb := make([]float64, g*g)
		s := maskN / g
		for y := 0; y < g; y++ {
			for x := 0; x < g; x++ {
				var a, b float64
				var n int
				for dy := 0; dy < s; dy++ {
					for dx := 0; dx < s; dx++ {
						ya, xa := y*s+dy, x*s+dx
						a += q.Mask48[ya*maskN+xa]
						b += r.Mask48[ya*maskN+xa]
						n++
					}
				}
				qa[y*g+x] = a / float64(n)
				rb[y*g+x] = b / float64(n)
			}
		}
		return maskScoreOn(qa, rb, g)
	}
	return maskScoreOn(q.Mask48, r.Mask48, maskN)
}

// maskScoreOn is maskScore for arbitrary grids.
func maskScoreOn(qa, rb []float64, g int) float64 {
	best := 0.0
	rot := make([]float64, g*g)
	qsum := 0.0
	for _, v := range qa {
		qsum += v
	}
	rsum := 0.0
	for _, v := range rb {
		rsum += v
	}
	if qsum <= 1e-9 || rsum <= 1e-9 {
		return 0
	}
	c := 0.5 * float64(g-1)
	for th := 0; th < 360; th += 4 {
		cs, sn := math.Cos(float64(th)*math.Pi/180), math.Sin(float64(th)*math.Pi/180)
		for y := 0; y < g; y++ {
			dy := float64(y) - c
			for x := 0; x < g; x++ {
				dx := float64(x) - c
				sx := dx*cs + dy*sn + c
				sy := -dx*sn + dy*cs + c
				x0, y0 := int(sx), int(sy)
				if x0 < 0 || y0 < 0 || x0 >= g-1 || y0 >= g-1 {
					rot[y*g+x] = 0
					continue
				}
				fx := sx - float64(x0)
				fy := sy - float64(y0)
				a := qa[y0*g+x0]*(1-fx) + qa[y0*g+x0+1]*fx
				bb := qa[(y0+1)*g+x0]*(1-fx) + qa[(y0+1)*g+x0+1]*fx
				rot[y*g+x] = a*(1-fy) + bb*fy
			}
		}
		var inter float64
		for i := range rot {
			a, b := rot[i], rb[i]
			if a < b {
				inter += a
			} else {
				inter += b
			}
		}
		v := 2 * inter / (qsum + rsum)
		if v > best {
			best = v
		}
		if best > 0.99 {
			break
		}
	}
	return best
}

// maskScore returns the best weighted overlap between the query and reference
// masks over a fine 2-degree rotation sweep. The masks carry coverage weight
// (alpha for refs, bg-distance for queries) so thin interior strokes survive.
func maskScore(q, r *Feat) float64 {
	return maskScoreGrid(q, r, maskN)
}

func maskSum(m []float64) float64 {
	var s float64
	for _, v := range m {
		s += v
	}
	return s
}

func maskCount(m []float64) float64 {
	return maskSum(m)
}

func defaultWeights(mono bool) []float64 {
	if mono {
		// gray: histogram families all equal -> Zernike/most shape features
		return []float64{0.03, 0.12, 0.15, 0.25, 0.05, 0.05, 0.35}
	}
	// colorful: histogram dominates, Zernike/color layout refine tight cases
	return []float64{0.62, 0.04, 0.06, 0.10, 0.08, 0.02, 0.08}
}
