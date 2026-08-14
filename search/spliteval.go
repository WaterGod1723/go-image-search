package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
)

// SplitConfig controls the source-level train/val/test split.
type SplitConfig struct {
	TrainRatio float64
	ValRatio   float64
	TestRatio  float64
	Seed       int64
}

// DefaultSplitConfig returns a 70/15/15 source-level split.
func DefaultSplitConfig() SplitConfig {
	return SplitConfig{TrainRatio: 0.70, ValRatio: 0.15, TestRatio: 0.15, Seed: 42}
}

// negTier describes the difficulty level of a negative sample.
type negTier int

const (
	negEasy negTier = iota
	negMedium
	negHard
)

func (t negTier) String() string {
	switch t {
	case negEasy:
		return "easy"
	case negMedium:
		return "medium"
	default:
		return "hard"
	}
}

// negCounts holds per-tier negative counts per query.
type negCounts struct {
	easy, medium, hard int
}

// defaultNegCounts returns the standard negative composition:
// 1 positive : 2 easy : 2 medium : 1 hard (5 negatives total).
func defaultNegCounts() negCounts {
	return negCounts{easy: 2, medium: 2, hard: 1}
}

// sourceGroup bundles all ref indices sharing the same sourceID.
type sourceGroup struct {
	sourceID string
	refIdxs  []int
}

// groupRefsBySource partitions ref indices into source groups. The returned
// slice is sorted by sourceID for deterministic splitting.
func groupRefsBySource(refNames []string) []sourceGroup {
	idxByID := map[string][]int{}
	var ids []string
	for i, n := range refNames {
		sid := sourceIDOf(n)
		if _, ok := idxByID[sid]; !ok {
			ids = append(ids, sid)
		}
		idxByID[sid] = append(idxByID[sid], i)
	}
	sort.Strings(ids)
	out := make([]sourceGroup, len(ids))
	for i, sid := range ids {
		out[i] = sourceGroup{sourceID: sid, refIdxs: idxByID[sid]}
	}
	return out
}

// splitSources divides source groups into train/val/test by sourceID. All refs
// from the same source always land in the same split — no leakage.
func splitSources(groups []sourceGroup, cfg SplitConfig) (trainSrcs, valSrcs, testSrcs map[string]bool) {
	rng := rand.New(rand.NewSource(cfg.Seed))
	perm := rng.Perm(len(groups))
	n := len(groups)
	nTrain := int(math.Round(cfg.TrainRatio * float64(n)))
	nVal := int(math.Round(cfg.ValRatio * float64(n)))
	if nTrain+nVal >= n {
		nVal = n - nTrain - 1
	}
	if nTrain < 1 {
		nTrain = 1
	}
	if nVal < 1 && n > nTrain {
		nVal = 1
	}
	trainSrcs = make(map[string]bool, nTrain)
	valSrcs = make(map[string]bool, nVal)
	testSrcs = make(map[string]bool, n-nTrain-nVal)
	for i, pi := range perm {
		sid := groups[pi].sourceID
		switch {
		case i < nTrain:
			trainSrcs[sid] = true
		case i < nTrain+nVal:
			valSrcs[sid] = true
		default:
			testSrcs[sid] = true
		}
	}
	return
}

// collectSplitRefs gathers ref indices whose sourceID belongs to srcSet.
func collectSplitRefs(groups []sourceGroup, srcSet map[string]bool) []int {
	var out []int
	for _, g := range groups {
		if srcSet[g.sourceID] {
			out = append(out, g.refIdxs...)
		}
	}
	return out
}

// makeGalleryMap maps a ref's global index to its position in a gallery
// subset (used to locate the true ref inside a per-split gallery).
func makeGalleryMap(galleryIdxs []int) map[int]int {
	m := make(map[int]int, len(galleryIdxs))
	for gi, ri := range galleryIdxs {
		m[ri] = gi
	}
	return m
}

// makeGalleryRefs builds the []*Feat gallery from global refs by index.
func makeGalleryRefs(refs []*Feat, galleryIdxs []int) []*Feat {
	out := make([]*Feat, len(galleryIdxs))
	for gi, ri := range galleryIdxs {
		out[gi] = refs[ri]
	}
	return out
}

// makeGallerySrcIDs extracts source IDs for a gallery subset.
func makeGallerySrcIDs(refSrcIDs []string, galleryIdxs []int) []string {
	out := make([]string, len(galleryIdxs))
	for gi, ri := range galleryIdxs {
		out[gi] = refSrcIDs[ri]
	}
	return out
}

// querySplitOf returns "train"/"val"/"test" based on which split the query's
// source belongs to.
func querySplitOf(sourceID string, trainS, valS, testS map[string]bool) string {
	switch {
	case trainS[sourceID]:
		return "train"
	case valS[sourceID]:
		return "val"
	case testS[sourceID]:
		return "test"
	default:
		return "unknown"
	}
}

// tieredHardNegs returns negative ref indices for a query, split into easy /
// medium / hard tiers. All negatives come from refs in the same split whose
// sourceID differs from the query's source, so no same-source variant is ever
// used as a negative.
//
//   - easy:   random refs from different sources
//   - medium: refs with high color-histogram overlap (similar palette)
//   - hard:   refs with highest combined color+shape overlap (most confusable)
func tieredHardNegs(q *Feat, refs []*Feat, refSrcIDs []string, candidateIdxs []int, trueIdx int, qSourceID string, nc negCounts, seed int) (easy, medium, hard []int) {
	type cand struct {
		idx int
		s   float64
	}
	cands := make([]cand, 0, len(candidateIdxs))
	for _, ci := range candidateIdxs {
		if ci == trueIdx {
			continue
		}
		if refSrcIDs[ci] == qSourceID {
			continue
		}
		cands = append(cands, cand{idx: ci, s: refOverlap(q, refs[ci])})
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].s > cands[b].s })

	rng := rand.New(rand.NewSource(int64(seed)*7919 + 13))
	seen := make(map[int]bool, nc.easy+nc.medium+nc.hard)

	// hard: top-N by overlap
	for i := 0; i < len(cands) && len(hard) < nc.hard; i++ {
		idx := cands[i].idx
		if !seen[idx] {
			hard = append(hard, idx)
			seen[idx] = true
		}
	}

	// medium: from the upper-middle band (overlap around the 50th percentile)
	medStart := len(cands) / 4
	if medStart > len(cands) {
		medStart = len(cands)
	}
	medPool := cands[medStart:]
	rng.Shuffle(len(medPool), func(i, j int) { medPool[i], medPool[j] = medPool[j], medPool[i] })
	for _, c := range medPool {
		if len(medium) >= nc.medium {
			break
		}
		if !seen[c.idx] {
			medium = append(medium, c.idx)
			seen[c.idx] = true
		}
	}

	// easy: random from the lower-overlap tail + random from the whole pool
	for len(easy) < nc.easy {
		if len(cands) == 0 {
			break
		}
		idx := cands[rng.Intn(len(cands))].idx
		if !seen[idx] {
			easy = append(easy, idx)
			seen[idx] = true
		}
	}
	return
}

// rankMetrics holds retrieval metrics for one split.
type rankMetrics struct {
	refRecall1, refRecall5, refRecall10 float64
	srcRecall1, srcRecall5, srcRecall10 float64
	mrr                                  float64
	n                                    int
}

// computeRankMetrics evaluates the model on a set of queries against a gallery
// of refs. For each query, it ranks all gallery refs by the model's predicted
// score, then computes:
//   - refRecall@K: fraction where the true ref is in the top-K
//   - srcRecall@K: fraction where any ref from the same source is in the top-K
//   - MRR: mean reciprocal rank of the true ref
func computeRankMetrics(m *AttnNet, queries []nnQuery, galleryRefs []*Feat, gallerySrcIDs []string, refIdxInGallery map[int]int) rankMetrics {
	var mt rankMetrics
	mt.n = len(queries)
	if mt.n == 0 {
		return mt
	}
	xbuf := make([]float64, nnInput)
	var sumRR float64
	counted := 0
	for i := range queries {
		qu := &queries[i]
		gIdx, ok := refIdxInGallery[qu.trueIdx]
		if !ok {
			continue
		}
		counted++
		qSrc := gallerySrcIDs[gIdx]
		bestScore := -1e18
		scores := make([]float64, len(galleryRefs))
		for j := range galleryRefs {
			nnVecInto(xbuf, qu.q, galleryRefs[j])
			s := m.predict(xbuf)
			scores[j] = s
			if s > bestScore {
				bestScore = s
			}
		}
		// rank: count refs with higher score than true ref
		trueScore := scores[gIdx]
		refRank := 1
		srcRank := 1
		for j := range galleryRefs {
			if j == gIdx {
				continue
			}
			if scores[j] > trueScore {
				refRank++
				if gallerySrcIDs[j] != qSrc {
					srcRank++
				}
			}
		}
		if refRank <= 1 {
			mt.refRecall1++
		}
		if refRank <= 5 {
			mt.refRecall5++
		}
		if refRank <= 10 {
			mt.refRecall10++
		}
		if srcRank <= 1 {
			mt.srcRecall1++
		}
		if srcRank <= 5 {
			mt.srcRecall5++
		}
		if srcRank <= 10 {
			mt.srcRecall10++
		}
		sumRR += 1.0 / float64(refRank)
	}
	mt.n = counted
	if mt.n == 0 {
		return mt
	}
	mt.refRecall1 /= float64(mt.n)
	mt.refRecall5 /= float64(mt.n)
	mt.refRecall10 /= float64(mt.n)
	mt.srcRecall1 /= float64(mt.n)
	mt.srcRecall5 /= float64(mt.n)
	mt.srcRecall10 /= float64(mt.n)
	mt.mrr = sumRR / float64(mt.n)
	return mt
}

// computeRankMetricsFast uses the two-stage prefilter+NN pipeline (rankNN)
// instead of exhaustive scoring, so large galleries remain fast.
func computeRankMetricsFast(m *AttnNet, queries []nnQuery, galleryRefs []*Feat, gallerySrcIDs []string, refIdxInGallery map[int]int) rankMetrics {
	var mt rankMetrics
	mt.n = len(queries)
	if mt.n == 0 {
		return mt
	}
	var sumRR float64
	counted := 0
	for i := range queries {
		qu := &queries[i]
		gIdx, ok := refIdxInGallery[qu.trueIdx]
		if !ok {
			continue
		}
		counted++
		qSrc := gallerySrcIDs[gIdx]
		ranked := rankNN(qu.q, galleryRefs, m)
		refRank, srcRank := 1, 1
		for _, rIdx := range ranked {
			if rIdx == gIdx {
				break
			}
			refRank++
			if gallerySrcIDs[rIdx] != qSrc {
				srcRank++
			}
		}
		if refRank <= 1 {
			mt.refRecall1++
		}
		if refRank <= 5 {
			mt.refRecall5++
		}
		if refRank <= 10 {
			mt.refRecall10++
		}
		if srcRank <= 1 {
			mt.srcRecall1++
		}
		if srcRank <= 5 {
			mt.srcRecall5++
		}
		if srcRank <= 10 {
			mt.srcRecall10++
		}
		sumRR += 1.0 / float64(refRank)
	}
	mt.n = counted
	if mt.n == 0 {
		return mt
	}
	mt.refRecall1 /= float64(mt.n)
	mt.refRecall5 /= float64(mt.n)
	mt.refRecall10 /= float64(mt.n)
	mt.srcRecall1 /= float64(mt.n)
	mt.srcRecall5 /= float64(mt.n)
	mt.srcRecall10 /= float64(mt.n)
	mt.mrr = sumRR / float64(mt.n)
	return mt
}

// embDistStats holds similarity statistics for diagnostics.
type embDistStats struct {
	posMean, posMin, posMax float64
	negMean, negMin, negMax float64
	hardMean, hardMin, hardMax float64
}

// computeEmbDistStats samples a subset of queries and computes the model's
// predicted similarity for positive, easy-negative, and hard-negative pairs,
// so we can verify the embedding actually separates the classes.
func computeEmbDistStats(m *AttnNet, queries []nnQuery, refs []*Feat, samples []nnSample, refSrcIDs []string, maxN int) embDistStats {
	var st embDistStats
	if len(samples) == 0 {
		return st
	}
	xbuf := make([]float64, nnInput)
	var pos, neg, hard []float64
	for _, s := range samples {
		if s.y == 1 {
			nnVecInto(xbuf, queries[s.qIdx].q, refs[s.rIdx])
			pos = append(pos, m.predict(xbuf))
		} else {
			nnVecInto(xbuf, queries[s.qIdx].q, refs[s.rIdx])
			v := m.predict(xbuf)
			neg = append(neg, v)
			// classify hard by high negative score
			if v > 0.3 {
				hard = append(hard, v)
			}
		}
		if len(pos)+len(neg) >= maxN {
			break
		}
	}
	if len(pos) > 0 {
		st.posMean, st.posMin, st.posMax = meanMinMax(pos)
	}
	if len(neg) > 0 {
		st.negMean, st.negMin, st.negMax = meanMinMax(neg)
	}
	if len(hard) > 0 {
		st.hardMean, st.hardMin, st.hardMax = meanMinMax(hard)
	}
	return st
}

func meanMinMax(v []float64) (float64, float64, float64) {
	if len(v) == 0 {
		return 0, 0, 0
	}
	var sum, mn, mx float64
	mn, mx = v[0], v[0]
	for _, x := range v {
		sum += x
		if x < mn {
			mn = x
		}
		if x > mx {
			mx = x
		}
	}
	return sum / float64(len(v)), mn, mx
}

// printMetrics formats and prints metrics for one split.
func printMetrics(label string, mt rankMetrics) {
	fmt.Printf("  %-12s  ref@1=%.1f%%  ref@5=%.1f%%  ref@10=%.1f%%  src@1=%.1f%%  src@5=%.1f%%  src@10=%.1f%%  mrr=%.3f  (n=%d)\n",
		label,
		100*mt.refRecall1, 100*mt.refRecall5, 100*mt.refRecall10,
		100*mt.srcRecall1, 100*mt.srcRecall5, 100*mt.srcRecall10,
		mt.mrr, mt.n)
}

// parseSplitConfig reads split ratios from env vars (TRAIN_RATIO, VAL_RATIO,
// TEST_RATIO, SPLIT_SEED), falling back to 70/15/15.
func parseSplitConfig() SplitConfig {
	cfg := DefaultSplitConfig()
	if v := os.Getenv("TRAIN_RATIO"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			cfg.TrainRatio = f
		}
	}
	if v := os.Getenv("VAL_RATIO"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			cfg.ValRatio = f
		}
	}
	if v := os.Getenv("TEST_RATIO"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			cfg.TestRatio = f
		}
	}
	if v := os.Getenv("SPLIT_SEED"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			cfg.Seed = n
		}
	}
	return cfg
}
