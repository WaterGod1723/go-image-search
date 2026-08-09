package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"time"

	"image-search-test/search/sczl"
)

// buildRefIndex loads the reference sprites (test_pngs) exactly like the eval
// path does.
func buildRefIndex(root string) ([]*Feat, []string) {
	return indexRefs(filepath.Join(root, "test_pngs"))
}

// queryMasks precomputes the 90 rotated copies of a query's 64x64 mask once.
// maskScore would otherwise re-rotate the query for every reference; sharing
// the precomputed set turns the per-query index sweep from
// O(rotations*refs*4096) into O(rotations*4096 + refs*rotations*4096) with the
// rotation factor amortized once, i.e. ~66x fewer rotations per query.
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

// score returns the best rotation-aligned Dice of the query mask against a
// reference mask, mirroring maskScore() exactly but reusing the precomputed
// rotations.
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

// pairFeat computes the 8 per-(query,ref) similarities fed to the network: the
// 7 from scores() plus the rotation-aligned mask overlap (cached).
func pairFeat(q, r *Feat, qm *queryMasks) []float64 {
	s := scores(q, r)
	return []float64{s[0], s[1], s[2], s[3], s[4], s[5], s[6], qm.score(r.Mask48)}
}

// polarShiftDetail returns, per ring, the shape cosine at the best global
// rotation (the same shift polarShiftSim finds), giving the network the
// rotation-invariant silhouette profile instead of just its aggregate — this is
// what separates the gray-outline icon families the aggregate scores tie on.
func polarShiftDetail(q, r *Feat) []float64 {
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
	return out
}

// computePairs builds the full pair vector per ref: the base similarities plus
// two learnable inductive biases the hand-written ranker gets for free:
//   - gate: 1 if the ref's color histogram is inside the query's histogram
//     "family" (the same soft margin compositeAdaptive uses), so a wrong-color
//     family can never outrank the true one;
//   - mono: whether the query sprite is monochrome (0/1), letting the model
//     condition on gray vs colorful queries.
//
// plus the 16 per-ring shape-detail cosines and the five sczl color-agnostic
// global sub-scores (occ / ncc / region / Fourier / radial) as separate expert
// dimensions (a second, independent expert whose error set differs — the
// network learns when to trust each one, i.e. soft neural routing).
func computePairs(q *Feat, refs []*Feat, qm *queryMasks, sczlSub [][5]float64) [][]float64 {
	out := make([][]float64, len(refs))
	hists := make([]float64, len(refs))
	for i, r := range refs {
		s := scores(q, r)
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
	for i, r := range refs {
		gate := 0.0
		if hists[i] >= bestHist-eps {
			gate = 1
		}
		out[i] = append(out[i], gate, mono)
		out[i] = append(out[i], polarShiftDetail(q, r)...)
		if sczlSub != nil {
			out[i] = append(out[i], sczlSub[i][:]...)
		} else {
			out[i] = append(out[i], 0, 0, 0, 0, 0)
		}
	}
	return out
}

// buildPairData computes the pair feature vectors, the query-level statistics
// and the sczl sub-scores for a query, in one place shared by training and
// evaluation so the NN never sees inconsistent feature layouts.
func buildPairData(q *Feat, refs []*Feat, qm *queryMasks, sczlIx *sczl.Index, qd sczl.Descriptor) ([][]float64, []float64, []float64) {
	var sczlSub [][5]float64
	if sczlIx != nil && qd.Valid {
		sczlSub = sczlIx.SubScoresAll(qd, 36)
	}
	return buildPairDataScored(q, refs, qm, sczlSub)
}

// buildPairDataScored is buildPairData with the sczl sub-scores supplied by the
// caller (e.g. computed only for a pre-filtered shortlist).
func buildPairDataScored(q *Feat, refs []*Feat, qm *queryMasks, sczlSub [][5]float64) ([][]float64, []float64, []float64) {
	pairs := computePairs(q, refs, qm, sczlSub)
	best, gap := queryStats(pairs)
	return pairs, best, gap
}

// cheapPref is the rotation-invariant, scan-free pre-filter score
// (0.4*hist + 0.2*zern + 0.2*radial + 0.2*angmag, all cheap cosines). It is
// used to shortlist refs before the expensive mask/sczl/NN/SC work.
func cheapPref(q, r *Feat) float64 {
	return 0.4*histSim(q.Hist, r.Hist) + 0.2*cosSim(q.Zern, r.Zern) +
		0.2*cosSim(q.Radial, r.Radial) + 0.2*cosSim(q.AngMag, r.AngMag)
}

// prefilterN returns the coarse pre-filter shortlist size (env PREFILTER_N,
// default 50). For the current test set a size of 40 keeps the true ref in
// 100% of queries; the default leaves margin for unseen data.
func prefilterN() int {
	n := 50
	if v := os.Getenv("PREFILTER_N"); v != "" {
		if x, err := strconv.Atoi(v); err == nil && x > 0 {
			n = x
		}
	}
	return n
}

// queryStats summarizes, for this query, the best and best-to-runner-up gap of
// every pair feature across the whole reference set. This injects the
// "adaptivity" signal compositeAdaptive computes by hand into the pointwise
// model.
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

// rankNN ranks refs for query q by the MLP's predicted relevance, in two
// stages so the cost grows sub-linearly with the library size:
//
//	stage 1 (all refs, cheap): scan-free rotation-invariant pre-filter score;
//	stage 2 (top-N only): the expensive features (mask rotation, sczl expert,
//	polar detail), the NN forward pass, and the shape-context shortlist
//	refinement.
//
// The final order is histogram-primary → gate → NN score within the shortlist
// (with SC refinement), and the pre-filtered-out refs trail by their cheap
// score. Measured on test_set a shortlist of 40 never drops the true ref, so
// recall@1/@3 is preserved while big libraries avoid the per-ref mask/sczl
// sweep for all but N entries.
func rankNN(q *Feat, refs []*Feat, m *MLP, sczlIx *sczl.Index, qd sczl.Descriptor) []int {
	prefN := prefilterN()

	// ---- stage 1: cheap pre-filter over all refs ----
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

	// ---- stage 2: full features + NN on the shortlist ----
	qm := newQueryMasks(q.Mask48)
	var sczlSub [][5]float64
	if sczlIx != nil && qd.Valid {
		pq := sczl.PrepareQuery(qd, 36)
		sczlSub = make([][5]float64, nKeep)
		for j, ki := range keep {
			sczlSub[j] = pq.GlobalSubScoresOf(sczlIx.Entries[ki])
		}
	}
	keptRefs := make([]*Feat, nKeep)
	for j, ki := range keep {
		keptRefs[j] = refs[ki]
	}
	pairs, best, gap := buildPairDataScored(q, keptRefs, qm, sczlSub)
	score := make([]float64, nKeep)
	for j := range keptRefs {
		score[j] = m.predict(nnVec(pairs[j], best, gap))
	}
	outKeep := make([]int, nKeep)
	for j := range outKeep {
		outKeep[j] = j
	}
	insertionSort(outKeep, func(a, b int) bool {
		if pairs[a][0] != pairs[b][0] {
			return pairs[a][0] > pairs[b][0]
		}
		ga, gb := pairs[a][8], pairs[b][8]
		if ga != gb {
			return ga > gb
		}
		return score[a] > score[b]
	})

	// Shortlist shape-context refinement: the neural score handles the broad
	// ranking; a precise (but expensive) point-match re-checks only the top few
	// in-family candidates, mirroring compositeAdaptive's refinement stage.
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
		insertionSort(outKeep, func(a, b int) bool {
			if pairs[a][0] != pairs[b][0] {
				return pairs[a][0] > pairs[b][0]
			}
			ga, gb := pairs[a][8], pairs[b][8]
			if ga != gb {
				return ga > gb
			}
			return score[a] > score[b]
		})
	}

	// ---- final: shortlist first (original indices), then the rest ----
	out := make([]int, 0, len(refs))
	for _, j := range outKeep {
		out = append(out, keep[j])
	}
	for _, i := range order[nKeep:] {
		out = append(out, i)
	}
	return out
}

type nnSample struct {
	x []float64
	y float64
	w float64
}

// nnQuery holds the per-query pair features and stats so training epochs never
// recompute features (the rotation-scan mask match is the expensive part).
type nnQuery struct {
	trueIdx int
	pairs   [][]float64
	best    []float64
	gap     []float64
}

// runTrain builds a labeled dataset from a gentest-generated directory, trains
// the fusion MLP, and persists the weights.
func runTrain(root string, args []string) {
	trainDir := "train_set"
	weightsPath := "weights.gob"
	epochs := 12
	lr := 0.01
	batch := 256
	if len(args) > 0 {
		trainDir = args[0]
	}
	if len(args) > 1 {
		weightsPath = args[1]
	}
	if len(args) > 2 {
		if v, err := strconv.Atoi(args[2]); err == nil {
			epochs = v
		}
	}
	if len(args) > 3 {
		if v, err := strconv.ParseFloat(args[3], 64); err == nil {
			lr = v
		}
	}

	refs, refNames := buildRefIndex(root)
	fmt.Printf("refs: %d\n", len(refs))
	entries := loadManifest(trainDir)

	tPrep := time.Now()
	sczlIx := buildSCZLRefs(root)
	type qPrep struct {
		e       manifestEntry
		trueIdx int
		q       *Feat
		qd      sczl.Descriptor
	}
	prep := make([]qPrep, len(entries))
	valid := make([]bool, len(entries))
	parFor(len(entries), func(i int) {
		e := entries[i]
		img, err := loadPNG(filepath.Join(trainDir, e.Image))
		if err != nil {
			return
		}
		px := extractQuery(img)
		if len(px) == 0 {
			return
		}
		ti := -1
		for j, n := range refNames {
			if n == e.Src {
				ti = j
				break
			}
		}
		if ti < 0 {
			return
		}
		prep[i] = qPrep{e: e, trueIdx: ti, q: buildFeat(px), qd: sczl.Extract(img)}
		valid[i] = true
	})
	used := prep[:0]
	for i := range valid {
		if valid[i] {
			used = append(used, prep[i])
		}
	}
	fmt.Printf("prep: %d queries (%.1fs)\n", len(used), time.Since(tPrep).Seconds())
	prep = used
	if len(prep) == 0 {
		fatal(fmt.Errorf("no usable queries in %s", trainDir))
	}

	// Phase 1 (the expensive one, parallel): per-query rotation-mask match
	// against every reference. The rotation sweeps are the CPU-heavy part; each
	// query is independent so split across all cores.
	start := time.Now()
	queries := make([]nnQuery, len(prep))
	parFor(len(prep), func(i int) {
		queries[i] = buildNNQuery(prep[i].q, refs, sczlIx, prep[i].qd, prep[i].trueIdx)
	})
	fmt.Printf("pair features: %.1fs\n", time.Since(start).Seconds())

	// Phase 2 (serial): build samples. Per query: 1 positive (the true ref) +
	// hard negatives (histogram-closest refs) + a few random negatives, so the
	// model spends its capacity on the confusable gray-outline families.
	var samples []nnSample
	for i := range queries {
		qu := &queries[i]
		samples = append(samples, nnSample{x: nnVec(qu.pairs[qu.trueIdx], qu.best, qu.gap), y: 1, w: 1})
		for _, j := range hardNegs(qu, i) {
			samples = append(samples, nnSample{x: nnVec(qu.pairs[j], qu.best, qu.gap), y: 0, w: 1})
		}
	}
	fmt.Printf("samples: %d\n", len(samples))

	// class-balanced BCE weights
	nPos, nNeg := 0, 0
	for _, s := range samples {
		if s.y == 1 {
			nPos++
		} else {
			nNeg++
		}
	}
	total := float64(len(samples))
	wPos := total / float64(2*nPos)
	wNeg := total / float64(2*nNeg)
	for i := range samples {
		if samples[i].y == 1 {
			samples[i].w = wPos
		} else {
			samples[i].w = wNeg
		}
	}
	fmt.Printf("pos=%d neg=%d wPos=%.2f wNeg=%.2f\n", nPos, nNeg, wPos, wNeg)

	m := newMLP(42)
	adam := newAdam(m)
	rng := rand.New(rand.NewSource(7))
	order := make([]int, len(samples))
	for i := range order {
		order[i] = i
	}
	t0 := time.Now()
	for ep := 0; ep < epochs; ep++ {
		rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
		var loss float64
		for off := 0; off < len(order); off += batch {
			end := min(off+batch, len(order))
			g := newGrads(m)
			n := end - off
			for _, si := range order[off:end] {
				s := samples[si]
				p, _, _, _, _, _, _, _ := m.forward(s.x)
				loss += -s.w * (s.y*mathLog(p) + (1-s.y)*mathLog(1-p))
				m.backprop(g, s.x, s.y, s.w)
			}
			adam.step(m, g, n, lr)
		}
		if ep%4 == 0 || ep == epochs-1 {
			t1 := time.Now()
			ok := trainAt1(m, queries)
			fmt.Printf("epoch %d/%d  loss=%.4f  train@1=%d/%d (%.1f%%)  [%.1fs]\n",
				ep+1, epochs, loss/float64(len(samples)), ok, len(queries),
				100*float64(ok)/float64(len(queries)), t1.Sub(t0).Seconds())
		}
	}
	fmt.Printf("epoch phase: %.1fs\n", time.Since(t0).Seconds())

	if err := m.save(weightsPath); err != nil {
		fatal(err)
	}
	fmt.Printf("weights saved to %s\n", weightsPath)
}

func buildNNQuery(q *Feat, refs []*Feat, sczlIx *sczl.Index, qd sczl.Descriptor, trueIdx int) nnQuery {
	qm := newQueryMasks(q.Mask48)
	pairs, best, gap := buildPairData(q, refs, qm, sczlIx, qd)
	return nnQuery{trueIdx: trueIdx, pairs: pairs, best: best, gap: gap}
}

// buildSCZLRefs extracts the sczl descriptor of every reference sprite, as the
// second expert's index.
func buildSCZLRefs(root string) *sczl.Index {
	srcDir := filepath.Join(root, "test_pngs")
	files, err := listPNG(srcDir)
	if err != nil {
		fatal(err)
	}
	ix := sczl.New()
	for _, fn := range files {
		img, err := loadPNG(filepath.Join(srcDir, fn))
		if err != nil {
			fatal(err)
		}
		if d := sczl.Extract(img); d.Valid {
			ix.AddImage(fn, d)
		}
	}
	return ix
}

// parFor runs fn over [0,n) across runtime.GOMAXPROCS workers.
func parFor(n int, fn func(i int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		for i := 0; i < n; i++ {
			fn(i)
		}
		return
	}
	var wg sync.WaitGroup
	next := make(chan int)
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for i := range next {
				fn(i)
			}
		}()
	}
	for i := 0; i < n; i++ {
		next <- i
	}
	close(next)
	wg.Wait()
}

func trainAt1(m *MLP, queries []nnQuery) int {
	ok := 0
	for i := range queries {
		qu := &queries[i]
		bestScore, bestIdx := -1.0, -1
		for j := range qu.pairs {
			if v := m.predict(nnVec(qu.pairs[j], qu.best, qu.gap)); v > bestScore {
				bestScore, bestIdx = v, j
			}
		}
		if bestIdx == qu.trueIdx {
			ok++
		}
	}
	return ok
}

func mathLog(x float64) float64 {
	if x <= 1e-12 {
		return -36.0
	}
	return math.Log(x)
}

// hardNegs returns negative ref indices for training. Negatives are chosen to
// be confusable along color OR shape: the top refs by a hist+mask blend, plus
// a few random refs (which are usually color-family outsiders), so the model
// learns both within-family shape ties and cross-family rejection.
func hardNegs(qu *nnQuery, seed int) []int {
	type hd struct {
		idx int
		s   float64
	}
	var cands []hd
	for j := range qu.pairs {
		if j == qu.trueIdx {
			continue
		}
		cands = append(cands, hd{idx: j, s: 0.5*qu.pairs[j][0] + 0.5*qu.pairs[j][7]})
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].s > cands[b].s })
	sel := make([]int, 0, 16)
	for _, c := range cands[:min(8, len(cands))] {
		sel = append(sel, c.idx)
	}
	rng := rand.New(rand.NewSource(int64(seed)*7919 + 13))
	for k := 0; k < 3; k++ {
		sel = append(sel, rng.Intn(len(qu.pairs)))
	}
	seen := make(map[int]bool, len(sel))
	out := make([]int, 0, len(sel))
	for _, j := range sel {
		if j == qu.trueIdx || seen[j] {
			continue
		}
		seen[j] = true
		out = append(out, j)
	}
	return out
}

// runSegDump renders the segmented sprite (green overlay on black) for every
// test query into dir, for visual QA of the extraction.
func runSegDump(root string, args []string) {
	dir := "segdump"
	if len(args) > 0 {
		dir = args[0]
	}
	entries := loadManifest(filepath.Join(root, "test_set"))
	ok, empty := 0, 0
	for _, e := range entries {
		img, err := loadPNG(filepath.Join(root, "test_set", e.Image))
		if err != nil {
			fatal(err)
		}
		px := extractQuery(img)
		if len(px) == 0 {
			fmt.Printf("EMPTY %s\n", e.Image)
			empty++
			continue
		}
		dumpSegmentation(dir, e.Image, img, px)
		ok++
	}
	fmt.Printf("segmented %d, empty %d -> %s\n", ok, empty, dir)
}

// runRankReport prints, per query, the rank of the TRUE ref under each ranking
// method, so the gray-query routing can be chosen on evidence. Methods:
// hist, mask64, mask32, zern, polarshape, SC (shape-context), NN (trained),
// baseline compositeAdaptive.
func runRankReport(root string, args []string) {
	m := &MLP{}
	if len(args) > 0 {
		var err error
		if m, err = loadMLP(args[0]); err != nil {
			fatal(err)
		}
	} else {
		m = newMLP(1)
	}
	refs, refNames := buildRefIndex(root)
	entries := loadManifest(filepath.Join(root, "test_set"))

	// sczl descriptors aligned 1:1 with refs, for the cheap coarse-filter
	// signature (Fourier + radial, rotation-invariant, no scan).
	refsSCZL := make([]sczl.Descriptor, len(refs))
	{
		files, err := listPNG(filepath.Join(root, "test_pngs"))
		if err != nil {
			fatal(err)
		}
		byName := map[string]int{}
		for i, nm := range refNames {
			byName[nm] = i
		}
		for _, fn := range files {
			if i, ok := byName[fn]; ok {
				if img, err := loadPNG(filepath.Join(root, "test_pngs", fn)); err == nil {
					refsSCZL[i] = sczl.Extract(img)
				}
			}
		}
	}
	n := len(entries)
	img := make([]string, n)
	want := make([]string, n)
	wi := make([]int, n)
	mono := make([]bool, n)
	gap := make([]float64, n)
	ok := make([]bool, n)
	histRank := make([]int, n)
	zernRank := make([]int, n)
	shapeRank := make([]int, n)
	m64Rank := make([]int, n)
	m32Rank := make([]int, n)
	scRank := make([]int, n)
	nnRank := make([]int, n)
	baseRank := make([]int, n)
	fdrRank := make([]int, n)
	prefRank := make([]int, n)

	parFor(n, func(i int) {
		e := entries[i]
		px := pxFromEntry(root, e.Image)
		if len(px) == 0 {
			return
		}
		q := buildFeat(px)
		idx := -1
		for j, name := range refNames {
			if name == e.Src {
				idx = j
				break
			}
		}
		if idx < 0 {
			return
		}
		img[i], want[i], wi[i], mono[i], ok[i] = e.Image, e.Src, idx, q.Mono, true

		sAll := make([][]float64, len(refs))
		m64 := make([]float64, len(refs))
		m32 := make([]float64, len(refs))
		sc := make([]float64, len(refs))
		pairs, best, g := buildPairData(q, refs, newQueryMasks(q.Mask48), nil, sczl.Descriptor{})
		gap[i] = g[0]
		nn := make([]float64, len(refs))
		for j, r := range refs {
			sAll[j] = scores(q, r)
			m64[j] = maskScore(q, r)
			m32[j] = maskScoreGrid(q, r, 32)
			sc[j] = shapeContextSim(q, r)
			nn[j] = m.predict(nnVec(pairs[j], best, g))
		}
		rank := func(v []float64) int {
			r := 1
			for j := range refs {
				if j != idx && v[j] > v[idx] {
					r++
				}
			}
			return r
		}
		histRank[i] = rank(func() []float64 {
			out := make([]float64, len(refs))
			for j := range refs {
				out[j] = sAll[j][0]
			}
			return out
		}())
		zernRank[i] = rank(func() []float64 {
			out := make([]float64, len(refs))
			for j := range refs {
				out[j] = sAll[j][6]
			}
			return out
		}())
		shapeRank[i] = rank(func() []float64 {
			out := make([]float64, len(refs))
			for j := range refs {
				out[j] = sAll[j][3]
			}
			return out
		}())
		m64Rank[i] = rank(m64)
		m32Rank[i] = rank(m32)
		scRank[i] = rank(sc)
		nnRank[i] = rank(nn)
		// cheap coarse-filter signature: sczl Fourier(32) + radial(16), cosine
		fdr := make([]float64, len(refs))
		if qimg, err := loadPNG(filepath.Join(root, "test_set", e.Image)); err == nil {
			qd := sczl.Extract(qimg)
			qv := sczlFR(qd)
			for j := range refs {
				fdr[j] = cosineF(qv, sczlFR(refsSCZL[j]))
			}
		}
		fdrRank[i] = rank(fdr)
		// cheap fused prefilter: 0.4*hist + 0.2*zern + 0.2*radial + 0.2*angmag
		// (all rotation-invariant, no scan) — how tight can the shortlist be?
		pref := make([]float64, len(refs))
		for j := range refs {
			pref[j] = 0.4*sAll[j][0] + 0.2*sAll[j][6] + 0.2*sAll[j][1] + 0.2*sAll[j][2]
		}
		prefRank[i] = rank(pref)
		br := 1
		for j, r := range compositeAdaptive(q, refs) {
			if r == idx {
				br = j + 1
				break
			}
		}
		baseRank[i] = br
	})

	fmt.Printf("  %-16s %-4s %-5s %-60s %-5s %-5s %-5s %-5s %-5s %-5s %-5s %-5s %-5s %-5s\n",
		"img", "mono", "gap", "want", "hist", "zern", "shp", "m64", "m32", "sc", "NN", "base", "fdr", "pref")
	for i := 0; i < n; i++ {
		if !ok[i] {
			continue
		}
		mark := "c"
		if mono[i] {
			mark = "m"
		}
		fmt.Printf("  %-16s %-4s %-5.3f %-60s %-5d %-5d %-5d %-5d %-5d %-5d %-5d %-5d %-5d %-5d\n",
			img[i], mark, gap[i], want[i], histRank[i], zernRank[i], shapeRank[i],
			m64Rank[i], m32Rank[i], scRank[i], nnRank[i], baseRank[i], fdrRank[i], prefRank[i])
	}
}

// sczlFR concatenates a descriptor's rotation-invariant Fourier + radial
// signature (cheap coarse-filter vector, no rotation scan).
func sczlFR(d sczl.Descriptor) []float64 {
	out := make([]float64, 0, len(d.Fourier)+len(d.Radial))
	out = append(out, d.Fourier...)
	out = append(out, d.Radial...)
	return out
}

// cosineF is the cosine similarity of two vectors.
func cosineF(a, b []float64) float64 {
	var d1, d2, dp float64
	for i := range a {
		d1 += a[i] * a[i]
		d2 += b[i] * b[i]
		dp += a[i] * b[i]
	}
	if d1 <= 0 || d2 <= 0 {
		return 0
	}
	return dp / (math.Sqrt(d1) * math.Sqrt(d2))
}

// pxFromEntry extracts the query sprite pixels for a test-set entry.
func pxFromEntry(root, name string) []Px {
	img, err := loadPNG(filepath.Join(root, "test_set", name))
	if err != nil {
		fatal(err)
	}
	return extractQuery(img)
}

// so a failing query can be inspected feature-by-feature (which features rank
// the true ref highest, which do not).
func dumpNNFeat(name, want string, q *Feat, refs []*Feat, refNames []string) {
	type row struct {
		name                                        string
		hist, radial, ang, shp, col, zern, m64, m32 float64
	}
	rows := make([]row, len(refs))
	wi := -1
	for i, r := range refs {
		if refNames[i] == want {
			wi = i
		}
		s := scores(q, r)
		rows[i] = row{refNames[i], s[0], s[1], s[2], s[3], s[4], s[6], maskScore(q, r), maskScoreGrid(q, r, 32)}
	}
	sort.Slice(rows, func(a, b int) bool { return rows[a].hist > rows[b].hist })
	fmt.Printf("== %s want=%s mono=%v n=%d\n", name, want, q.Mono, q.N)
	fmt.Printf("   %-72s hist   m64    m32    zern   shp\n", "ref")
	for i := range rows {
		tag := " "
		if rows[i].name == want {
			tag = "*"
		}
		fmt.Printf("%s %-72s %.3f %.3f %.3f %.3f %.3f\n", tag, rows[i].name,
			rows[i].hist, rows[i].m64, rows[i].m32, rows[i].zern, rows[i].shp)
		if i >= 14 {
			break
		}
	}
	fmt.Printf("   want(%s) is at row-index %d by hist\n", want, wi)
}

func runNNEval(root string, args []string) {
	weightsPath := "weights.gob"
	if len(args) > 0 {
		weightsPath = args[0]
	}
	m, err := loadMLP(weightsPath)
	if err != nil {
		fatal(err)
	}
	refs, refNames := buildRefIndex(root)
	fmt.Printf("index: %d reference sprites\n", len(refs))
	sczlIx := buildSCZLRefs(root)
	entries := loadManifest(filepath.Join(root, "test_set"))

	nn1, nn3, nn5, base1, base3, base5 := 0, 0, 0, 0, 0, 0
	type miss struct {
		img, want string
		got       []string
	}
	var nnMisses []miss
	type evalRow struct {
		img, want string
		nnRank    []int
		baseRank  []int
		mono      bool
		ok        bool
	}
	rows := make([]evalRow, len(entries))
	parFor(len(entries), func(i int) {
		e := entries[i]
		img, err := loadPNG(filepath.Join(root, "test_set", e.Image))
		if err != nil {
			fatal(err)
		}
		px := extractQuery(img)
		if len(px) == 0 {
			return
		}
		q := buildFeat(px)
		if os.Getenv("NNFEAT") == "1" && e.Image == "sample_00004.png" {
			dumpNNFeat(e.Image, e.Src, q, refs, refNames)
		}
		qd := sczl.Extract(img)
		rows[i] = evalRow{img: e.Image, want: e.Src, mono: q.Mono, nnRank: rankNN(q, refs, m, sczlIx, qd), baseRank: compositeAdaptive(q, refs), ok: true}
	})
	total := 0
	nnMono1, nnColor1 := 0, 0
	nnMonoN, nnColorN := 0, 0
	nnMono5, nnColor5 := 0, 0
	baseMono1, baseColor1 := 0, 0
	for i := range rows {
		r := rows[i]
		if !r.ok {
			continue
		}
		total++
		if refNames[r.nnRank[0]] == r.want {
			nn1++
		}
		in3 := false
		in5 := false
		var got5 []string
		for idx, ridx := range r.nnRank[:min(5, len(r.nnRank))] {
			got5 = append(got5, refNames[ridx])
			if refNames[ridx] == r.want {
				in5 = true
				if idx < 3 {
					in3 = true
				}
			}
		}
		if in3 {
			nn3++
		}
		if in5 {
			nn5++
		}
		if refNames[r.nnRank[0]] != r.want {
			nnMisses = append(nnMisses, miss{r.img, r.want, got5})
		}
		if r.mono {
			nnMonoN++
			if refNames[r.nnRank[0]] == r.want {
				nnMono1++
			}
			if in5 {
				nnMono5++
			}
			if refNames[r.baseRank[0]] == r.want {
				baseMono1++
			}
		} else {
			nnColorN++
			if refNames[r.nnRank[0]] == r.want {
				nnColor1++
			}
			if in5 {
				nnColor5++
			}
			if refNames[r.baseRank[0]] == r.want {
				baseColor1++
			}
		}
		in3 = false
		in5 = false
		for idx, ridx := range r.baseRank[:min(5, len(r.baseRank))] {
			if refNames[ridx] == r.want {
				in5 = true
				if idx < 3 {
					in3 = true
				}
			}
		}
		if in3 {
			base3++
		}
		if in5 {
			base5++
		}
		if refNames[r.baseRank[0]] == r.want {
			base1++
		}
	}

	fmt.Printf("baseline adaptive : recall@1=%d/%d (%.1f%%)  recall@3=%d/%d (%.1f%%)  recall@5=%d/%d (%.1f%%)\n",
		base1, total, 100*float64(base1)/float64(total), base3, total, 100*float64(base3)/float64(total), base5, total, 100*float64(base5)/float64(total))
	fmt.Printf("neural (MLP)      : recall@1=%d/%d (%.1f%%)  recall@3=%d/%d (%.1f%%)  recall@5=%d/%d (%.1f%%)\n",
		nn1, total, 100*float64(nn1)/float64(total), nn3, total, 100*float64(nn3)/float64(total), nn5, total, 100*float64(nn5)/float64(total))
	fmt.Printf("  neural mono : recall@1=%d/%d (%.1f%%)  @5=%d/%d (%.1f%%)   [base @1=%d/%d (%.1f%%)]\n",
		nnMono1, nnMonoN, pct(nnMono1, nnMonoN), nnMono5, nnMonoN, pct(nnMono5, nnMonoN), baseMono1, nnMonoN, pct(baseMono1, nnMonoN))
	fmt.Printf("  neural color: recall@1=%d/%d (%.1f%%)  @5=%d/%d (%.1f%%)   [base @1=%d/%d (%.1f%%)]\n",
		nnColor1, nnColorN, pct(nnColor1, nnColorN), nnColor5, nnColorN, pct(nnColor5, nnColorN), baseColor1, nnColorN, pct(baseColor1, nnColorN))

	if len(nnMisses) > 0 {
		fmt.Printf("\n-- NN misses (%d) --\n", len(nnMisses))
		for _, m2 := range nnMisses {
			fmt.Printf("%-18s want=%-60s got=%v\n", m2.img, m2.want, m2.got)
		}
	}
}
