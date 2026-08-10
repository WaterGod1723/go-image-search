package nnengine

import (
	"math"
	"os"
	"strconv"

	"go-image-search/internal/nnengine/sczl"
)

// polarShiftBoth computes, in a single shift scan, both the best global-rotation
// mean cosine (what polarShiftSim reports) and the per-ring detail vector (what
// polarShiftDetail reports). They previously scanned the same k range twice.
func polarShiftBoth(q, r *Feat) (float64, []float64) {
	bestK := 0
	best := -1e18
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
			if dq > 0 && dr > 0 {
				acc += dp / math.Sqrt(dq*dr)
				n++
			}
		}
		if n > 0 {
			if v := acc / float64(n); v > best {
				best, bestK = v, k
			}
		}
	}
	out := make([]float64, nRing)
	for ri := 0; ri < nRing; ri++ {
		qb := q.PolarShape[ri*nAng : (ri+1)*nAng]
		rb := r.PolarShape[ri*nAng : (ri+1)*nAng]
		var dq, dr, dp float64
		for a := 0; a < nAng; a++ {
			src := (a - bestK%nAng + nAng) % nAng
			dq += qb[a] * qb[a]
			dr += rb[src] * rb[src]
			dp += qb[a] * rb[src]
		}
		if dq > 0 && dr > 0 {
			out[ri] = dp / math.Sqrt(dq*dr)
		}
	}
	if best < 0 {
		best = 0
	}
	return best, out
}

// queryMasks precomputes the 90 rotated copies of a query's 64x64 mask once,
// shared across every reference (maskScore would otherwise re-rotate per ref).
type queryMasks struct {
	rot  [90][]float64
	qsum float64
}

func newQueryMasks(qa []float64) *queryMasks {
	qm := &queryMasks{}
	for _, v := range qa {
		qm.qsum += v
	}
	if qm.qsum <= 1e-9 {
		return qm
	}
	k := 0
	for th := 0; th < 360; th += 4 {
		rot := make([]float64, maskN*maskN)
		rotateMask(qa, float64(th), rot)
		qm.rot[k] = rot
		k++
	}
	return qm
}

func (qm *queryMasks) score(rb []float64) float64 {
	if qm.qsum <= 1e-9 {
		return 0
	}
	rsum := 0.0
	for _, v := range rb {
		rsum += v
	}
	if rsum <= 1e-9 {
		return 0
	}
	best := 0.0
	denom := qm.qsum + rsum
	for _, rot := range qm.rot {
		var inter float64
		for i := range rot {
			a, b := rot[i], rb[i]
			if a < b {
				inter += a
			} else {
				inter += b
			}
		}
		v := 2 * inter / denom
		if v > best {
			best = v
		}
		if best > 0.99 {
			break
		}
	}
	return best
}

// computePairs builds the full pair vector per ref: base similarities, the
// histogram-family gate, the mono flag, the 16 per-ring shape details, and the
// sczl color-agnostic global similarity (a second, independent expert whose
// error set differs — the network learns when to trust it, i.e. soft neural
// routing). With the split sczl layout each row holds the 6 sub-signals
// [occ, ncc, region, fourier, radial, hog] instead of a single fused score.
func computePairs(q *Feat, refs []*Feat, qm *queryMasks, sczlRows [][]float64) [][]float64 {
	out := make([][]float64, len(refs))
	hists := make([]float64, len(refs))
	details := make([][]float64, len(refs))
	for i, r := range refs {
		// s[3] (best-shift polar shape) and the per-ring detail come from the
		// SAME scan (polarShiftBoth) — previously polarShiftSim and
		// polarShiftDetail each scanned the full k range.
		s := make([]float64, 7)
		s[0] = histSim(q.Hist, r.Hist)
		s[1] = cosSim(q.Radial, r.Radial)
		s[2] = cosSim(q.AngMag, r.AngMag)
		s[3], details[i] = polarShiftBoth(q, r)
		s[4] = polarColSim(q, r)
		s[5] = 0.5*s[1] + 0.5*s[2] // roundTripFactor
		s[6] = cosSim(q.Zern, r.Zern)
		hists[i] = s[0]
		out[i] = []float64{s[0], s[1], s[2], s[3], s[4], s[5], s[6], qm.score(r.Mask48)}
	}
	bestHist := -1.0
	for _, h := range hists {
		if h > bestHist {
			bestHist = h
		}
	}
	eps := math.Max(0.015, (bestHist-0.7)*0.15)
	mono := 0.0
	if q.Mono {
		mono = 1
	}
	for i := range refs {
		gate := 0.0
		if hists[i] >= bestHist-eps {
			gate = 1
		}
		out[i] = append(out[i], gate, mono)
		out[i] = append(out[i], details[i]...)
		if sczlRows != nil {
			out[i] = append(out[i], sczlRows[i]...)
		} else {
			out[i] = append(out[i], 0)
		}
	}
	return out
}

// queryStats summarizes, for this query, the best and best-to-runner-up gap of
// every pair feature across the reference set (the adaptivity signal).
func queryStats(pairs [][]float64) (best, gap []float64) {
	if len(pairs) == 0 {
		return nil, nil
	}
	nF := len(pairs[0])
	best = make([]float64, nF)
	gap = make([]float64, nF)
	for d := 0; d < nF; d++ {
		b1, b2 := -1.0, -1.0
		for i := range pairs {
			v := pairs[i][d]
			if v > b1 {
				b2, b1 = b1, v
			} else if v > b2 {
				b2 = v
			}
		}
		if b1 < 0 {
			b1 = 0
		}
		if b2 < 0 {
			b2 = 0
		}
		best[d] = b1
		gap[d] = b1 - b2
	}
	return best, gap
}

func nnVec(pair, best, gap []float64) []float64 {
	x := make([]float64, 0, nnInput)
	x = append(x, pair...)
	x = append(x, best...)
	x = append(x, gap...)
	return x
}

func buildPairDataScored(q *Feat, refs []*Feat, qm *queryMasks, sczlRows [][]float64) ([][]float64, []float64, []float64) {
	pairs := computePairs(q, refs, qm, sczlRows)
	best, gap := queryStats(pairs)
	return pairs, best, gap
}

// cheapPref is the rotation-invariant, scan-free pre-filter score
// (0.4*hist + 0.2*zern + 0.2*radial + 0.2*angmag).
func cheapPref(q, r *Feat) float64 {
	return 0.4*histSim(q.Hist, r.Hist) + 0.2*cosSim(q.Zern, r.Zern) +
		0.2*cosSim(q.Radial, r.Radial) + 0.2*cosSim(q.AngMag, r.AngMag)
}

// prefilterN returns the coarse pre-filter shortlist size (env PREFILTER_N).
func prefilterN() int {
	n := 50
	if v := os.Getenv("PREFILTER_N"); v != "" {
		if x, err := strconv.Atoi(v); err == nil && x > 0 {
			n = x
		}
	}
	return n
}

// rankNN ranks refs for query q by the trained MLP, two-stage: cheap
// rotation-invariant pre-filter over all refs, then the expensive features
// (mask rotation, sczl expert, polar detail, NN, SC refinement) on the top-N
// shortlist only.
func rankNN(q *Feat, refs []*Feat, m *MLP, sczlIx *sczl.Index, qd sczl.Descriptor) ([]int, []float64) {
	prefN := prefilterN()

	pref := make([]float64, len(refs))
	for i, r := range refs {
		pref[i] = cheapPref(q, r)
	}
	order := make([]int, len(refs))
	for i := range order {
		order[i] = i
	}
	insertionSort(order, func(a, b int) bool { return pref[a] > pref[b] })
	nKeep := min(prefN, len(refs))
	keep := order[:nKeep]

	qm := newQueryMasks(q.Mask48)
	var sczlRows [][]float64
	if sczlIx != nil && qd.Valid {
		pq := sczl.PrepareQuery(qd, 36)
		sczlRows = make([][]float64, nKeep)
		for j, ki := range keep {
			sczlRows[j] = pq.GlobalScoresRow(sczlIx.Entries[ki])
		}
	}
	keptRefs := make([]*Feat, nKeep)
	for j, ki := range keep {
		keptRefs[j] = refs[ki]
	}
	pairs, best, gap := buildPairDataScored(q, keptRefs, qm, sczlRows)
	score := make([]float64, nKeep)
	for j := range keptRefs {
		score[j] = m.predict(nnVec(pairs[j], best, gap))
	}
	outKeep := make([]int, nKeep)
	for j := range outKeep {
		outKeep[j] = j
	}
	// 与训练/评测一致的排序：先按颜色直方图族（hist）分组，再按 gate、
	// 最后按神经网络相关度，保证展示与训练时的排序语义一致。
	sortKept := func(a, b int) bool {
		if pairs[a][0] != pairs[b][0] {
			return pairs[a][0] > pairs[b][0]
		}
		ga, gb := pairs[a][8], pairs[b][8]
		if ga != gb {
			return ga > gb
		}
		return score[a] > score[b]
	}
	insertionSort(outKeep, sortKept)

	scTop := 12
	scBlend := 0.7
	if v := os.Getenv("SCTOP"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			scTop = n
		}
	}
	if v := os.Getenv("SCBLEND"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			scBlend = f
		}
	}
	if scBlend > 0 && scTop > 0 {
		qHists := scQueryHists(q)
		for i := 0; i < scTop && i < len(outKeep); i++ {
			j := outKeep[i]
			score[j] = (1-scBlend)*score[j] + scBlend*scSimFromHists(qHists, keptRefs[j])
		}
		insertionSort(outKeep, sortKept)
	}

	out := make([]int, 0, len(refs))
	scores := make([]float64, 0, len(refs))
	for _, j := range outKeep {
		out = append(out, keep[j])
		scores = append(scores, score[j])
	}
	for _, i := range order[nKeep:] {
		out = append(out, i)
		scores = append(scores, cheapPref(q, refs[i]))
	}
	return out, scores
}
