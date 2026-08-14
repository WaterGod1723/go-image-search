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

// buildPairData computes, for a query, the raw feature blocks of every
// reference. The raw blocks are concatenated in the exact order declared by
// attnProjConfigs: [zern | hist | radial | angmag], each carrying the
// dual-view min-then-diff of the two L2/L1-normalized rotation-invariant
// descriptors — the raw signal the jointly-trained input projectors compress
// into tokens, replacing the former hand-crafted similarity scores.
func buildPairData(q *Feat, refs []*Feat) [][]float64 {
	out := make([][]float64, len(refs))
	for i, r := range refs {
		out[i] = buildPairVec(q, r)
	}
	return out
}

// buildPairVec returns the raw feature blocks for one (query, ref) pair.
// Per descriptor: [min(q,r) | |q-r|] — the overlap view says "how much both
// have", the diff view says "who is missing what". Block layout (raw dims):
// zern 2*49, hist 2*288, radial 2*16, angmag 2*192, struct 2*16 (rotation-
// pooled structure embeddings of query and ref).
func buildPairVec(q, r *Feat) []float64 {
	out := make([]float64, 0, nnProjRawIn())
	out = append(out, overlapDiff(q.Zern, r.Zern)...)
	out = append(out, overlapDiff(q.Hist, r.Hist)...)
	out = append(out, overlapDiff(q.Radial, r.Radial)...)
	out = append(out, overlapDiff(q.AngMag, r.AngMag)...)
	out = append(out, overlapDiff(q.Emb[:snEmb], r.Emb[:snEmb])...)
	return out
}

// overlapDiff returns [min(a,b) | |a-b|] for each component of two
// non-negative, normalized vectors: overlap in [0,1], diff in [0,1].
func overlapDiff(a, b []float64) []float64 {
	out := make([]float64, 0, 2*len(a))
	for i := range a {
		if a[i] < b[i] {
			out = append(out, a[i])
		} else {
			out = append(out, b[i])
		}
		dv := a[i] - b[i]
		if dv < 0 {
			dv = -dv
		}
		out = append(out, dv)
	}
	return out
}

// ---- structure-CNN deferred backward context ----
//
// The structure embedding of every query/ref is recomputed once per epoch; the
// per-pair backprop accumulates gradients w.r.t. those embeddings into qAcc/rAcc
// (via snSplitGrad, called from the attention backprop for the structure block),
// and at the end of the epoch runTrain runs the CNN/MLP backward once per image
// with the accumulated gradients.
var snCur = &snCtx{}

// snCtx carries the per-epoch structure-CNN scaffolding. The structure CNN is
// updated ONCE per epoch from gradients accumulated per image (deferred
// backward: conv/MLP path recomputed per image with the accumulated embedding
// gradients), which keeps training fast (forwards cached per epoch, no
// per-sample conv recompute). To keep the large accumulated step from
// destabilizing, snStep clamps the effective gradient by max-norm and the CNN
// uses a smaller learning rate snLR.
type snCtx struct {
	active   bool
	qIdx, rIdx int     // current sample's query/ref indices (set each sample)
	qE, rE   []float64 // current pair's q/r embeddings (for snSplitGrad)
	qDemb, rDemb []float64 // per-sample embedding grad scratch (zeroed each sample)
	qAcc, rAcc  []float64 // per-image accumulated embedding gradients (whole epoch)
	qFs, qmhs   []float64 // per query snRot orientations: F and MLP-hidden rows
	qWins       []int     // per query winning orientation per emb dim
	rFs, rmhs   []float64 // per ref F and MLP-hidden rows
}

// setupSN prepares snCur for a training model: enables the deferred CNN path
// and allocates the per-epoch accumulators.
func setupSN(sn *StructNet, nQuery, nRef int) {
	snCur.active = true
	snCur.qAcc = make([]float64, nQuery*snEmb)
	snCur.rAcc = make([]float64, nRef*snEmb)
	snCur.qFs = make([]float64, nQuery*snRot*snFMap)
	snCur.qmhs = make([]float64, nQuery*snRot*snHid)
	snCur.qWins = make([]int, nQuery*snEmb)
	snCur.rFs = make([]float64, nRef*snFMap)
	snCur.rmhs = make([]float64, nRef*snHid)
	snCur.qDemb = make([]float64, snEmb)
	snCur.rDemb = make([]float64, snEmb)
}

// snSplitGrad routes the structure block's raw-input gradient din of the
// CURRENT sample into that sample's q/r embedding gradients (stored in scratch,
// then consumed by the inline CNN backward in backprop, which accumulates into
// per-image qAcc/rAcc). The block layout (like every overlapDiff block) is
// INTERLEAVED: [min0 diff0 min1 diff1 ...], so dim d contributes to positions
// 2d (overlap) and 2d+1 (directed diff).
func snSplitGrad(din []float64) {
	if snCur.qE == nil {
		return
	}
	dq := snCur.qDemb
	dr := snCur.rDemb
	for d := 0; d < snEmb; d++ {
		gm, gd := din[2*d], din[2*d+1]
		q, r := snCur.qE[d], snCur.rE[d]
		if q < r {
			dq[d] += gm
		} else {
			dr[d] += gm
		}
		switch {
		case q > r:
			dq[d] += gd
			dr[d] -= gd
		case q < r:
			dq[d] -= gd
			dr[d] += gd
		}
	}
	// accumulate into per-image accumulators
	qi := snCur.qIdx * snEmb
	ri := snCur.rIdx * snEmb
	for d := 0; d < snEmb; d++ {
		snCur.qAcc[qi+d] += dq[d]
		snCur.rAcc[ri+d] += dr[d]
	}
}

// pairSum sums a slice (used to recover scalar similarities from raw blocks
// for hard-negative sampling and diagnostics).
func pairSum(a []float64) float64 {
	var s float64
	for _, v := range a {
		s += v
	}
	return s
}

// cheapPref is the rotation-invariant, scan-free pre-filter score
// (0.4*hist + 0.2*zern + 0.2*radial + 0.2*angmag, all cheap cosines). It is
// used to shortlist refs before the NN/SC work.
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

func nnVec(pair []float64) []float64 {
	x := make([]float64, 0, nnInput)
	x = append(x, make([]float64, nnPairDim())...)
	x = append(x, pair...)
	return x
}

// rankNN ranks refs for query q by the attention-net's predicted relevance, in
// two stages so the cost grows sub-linearly with the library size:
//
//	stage 1 (all refs, cheap): scan-free rotation-invariant pre-filter score;
//	stage 2 (top-N only): the raw similarity blocks, the NN forward pass, and
//	the optional shape-context shortlist refinement.
//
// The final order is NN score within the shortlist (with SC refinement), and
// the pre-filtered-out refs trail by their cheap score. Measured on the
// original test_set a shortlist of 40 never drops the true ref, so recall@1/@3
// is preserved while big libraries avoid the per-ref block sweep for all but N
// entries.
func rankNN(q *Feat, refs []*Feat, m *AttnNet) []int {
	if m.SN != nil {
		ensureStructEmb(refs, m.SN)
		if len(q.Emb) != snEmb {
			q.Emb = m.SN.embedEmbRot(q.Thumb16)
		}
	}
	return rankNNPri(q, refs, func(x []float64) float64 { return m.predict(x) })
}

// rankNNPri is rankNN parameterized by the relevance predictor, so different
// architectures (the fixed attention net, the flexible netMLP, etc.) share the
// same two-stage ranking + SC refinement pipeline.
func rankNNPri(q *Feat, refs []*Feat, predict func([]float64) float64) []int {
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

	// ---- stage 2: raw blocks + NN on the shortlist ----
	keptRefs := make([]*Feat, nKeep)
	for j, ki := range keep {
		keptRefs[j] = refs[ki]
	}
	pairs := buildPairData(q, keptRefs)
	score := make([]float64, nKeep)
	for j := range keptRefs {
		score[j] = predict(nnVec(pairs[j]))
	}
	outKeep := make([]int, nKeep)
	for j := range outKeep {
		outKeep[j] = j
	}
	insertionSort(outKeep, func(a, b int) bool {
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

// nnSample identifies one training instance by query/ref indices; the input
// vector x is assembled lazily per epoch from q/r Feats into a shared buffer,
// so 96k samples cost ~3MB instead of ~1GB of materialized vectors.
type nnSample struct {
	qIdx int
	rIdx int
	y    float64
	w    float64
	tier negTier // negative difficulty tier (for diagnostics); negEasy for positives
}

// nnVecInto writes a fresh input vector for (q, r) into buf: zeroed projected-
// token slots (patched by forward) followed by the raw similarity blocks. The
// structure block is built from the precomputed q/r embeddings (set by the
// per-epoch recompute); if the deferred CNN is active this also snapshots the
// current sample's q/r indices and embedding values for backprop attribution.
func nnVecInto(buf []float64, q, r *Feat) {
	for i := range buf[:nnPairDim()] {
		buf[i] = 0
	}
	if snCur.active {
		snCur.qE = q.Emb[:snEmb]
		snCur.rE = r.Emb[:snEmb]
	}
	raw := buf[nnPairDim():]
	copy(raw[:len(q.Zern)*2], overlapDiff(q.Zern, r.Zern))
	off := len(q.Zern) * 2
	h := hBins * sBins * vBins
	copy(raw[off:off+h*2], overlapDiff(q.Hist, r.Hist))
	off += h * 2
	copy(raw[off:off+nRing*2], overlapDiff(q.Radial, r.Radial))
	off += nRing * 2
	copy(raw[off:], overlapDiff(q.AngMag, r.AngMag))
	off += nRing * nFreq * 2
	copy(raw[off:off+2*snEmb], overlapDiff(q.Emb[:snEmb], r.Emb[:snEmb]))
}

// nnQuery holds a query's Feat and true-ref index. The raw similarity blocks
// are NOT cached per ref — they are cheap elementwise mins, and materializing
// 545-dim vectors for every (query, ref) would cost gigabytes on the 8000-query
// train set. buildPairVec is called lazily for the selected samples only.
type nnQuery struct {
	q       *Feat
	trueIdx int
}

// runTrain builds a labeled dataset from a gentest-generated directory, splits
// refs and queries by source (no leakage), trains the fusion attention net
// with tiered hard negatives, evaluates on val/test each epoch, and persists
// the best-val weights.
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

	// --- load all refs ---
	refs, refNames := buildRefIndex(root)
	refSrcIDs := make([]string, len(refNames))
	for i, n := range refNames {
		refSrcIDs[i] = sourceIDOf(n)
	}

	// --- source-level split (no leakage) ---
	groups := groupRefsBySource(refNames)
	splitCfg := parseSplitConfig()
	trainSrcs, valSrcs, testSrcs := splitSources(groups, splitCfg)
	trainRefIdxs := collectSplitRefs(groups, trainSrcs)
	valRefIdxs := collectSplitRefs(groups, valSrcs)
	testRefIdxs := collectSplitRefs(groups, testSrcs)

	fmt.Printf("refs: %d  sources: %d\n", len(refs), len(groups))
	fmt.Printf("train refs: %d (%d sources)  val refs: %d (%d sources)  test refs: %d (%d sources)\n",
		len(trainRefIdxs), len(trainSrcs), len(valRefIdxs), len(valSrcs), len(testRefIdxs), len(testSrcs))

	// --- load manifest, extract queries, split by source ---
	entries := loadManifest(trainDir)

	tPrep := time.Now()
	type qPrep struct {
		trueIdx int
		q       *Feat
		srcID   string
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
		sid := e.SourceID
		if sid == "" {
			sid = sourceIDOf(e.Src)
		}
		prep[i] = qPrep{trueIdx: ti, q: buildFeat(px), srcID: sid}
		valid[i] = true
	})

	var trainQs, valQs, testQs []qPrep
	for i := range valid {
		if !valid[i] {
			continue
		}
		switch querySplitOf(prep[i].srcID, trainSrcs, valSrcs, testSrcs) {
		case "train":
			trainQs = append(trainQs, prep[i])
		case "val":
			valQs = append(valQs, prep[i])
		case "test":
			testQs = append(testQs, prep[i])
		}
	}
	fmt.Printf("queries: %d  train: %d  val: %d  test: %d  (%.1fs)\n",
		len(trainQs)+len(valQs)+len(testQs), len(trainQs), len(valQs), len(testQs),
		time.Since(tPrep).Seconds())
	if len(trainQs) == 0 {
		fatal(fmt.Errorf("no usable train queries in %s", trainDir))
	}

	// --- build nnQuery arrays ---
	queries := make([]nnQuery, len(trainQs))
	parFor(len(trainQs), func(i int) {
		queries[i] = buildNNQuery(trainQs[i].q, trainQs[i].trueIdx)
	})
	valQueries := make([]nnQuery, len(valQs))
	parFor(len(valQs), func(i int) {
		valQueries[i] = buildNNQuery(valQs[i].q, valQs[i].trueIdx)
	})
	testQueries := make([]nnQuery, len(testQs))
	parFor(len(testQs), func(i int) {
		testQueries[i] = buildNNQuery(testQs[i].q, testQs[i].trueIdx)
	})

	// --- build training samples with tiered hard negatives ---
	nc := defaultNegCounts()
	var samples []nnSample
	nEasy, nMed, nHard := 0, 0, 0
	for i := range queries {
		qu := &queries[i]
		qSrc := refSrcIDs[qu.trueIdx]
		samples = append(samples, nnSample{qIdx: i, rIdx: qu.trueIdx, y: 1, w: 1, tier: negEasy})
		easy, medium, hard := tieredHardNegs(qu.q, refs, refSrcIDs, trainRefIdxs,
			qu.trueIdx, qSrc, nc, i)
		for _, j := range easy {
			samples = append(samples, nnSample{qIdx: i, rIdx: j, y: 0, w: 1, tier: negEasy})
			nEasy++
		}
		for _, j := range medium {
			samples = append(samples, nnSample{qIdx: i, rIdx: j, y: 0, w: 1, tier: negMedium})
			nMed++
		}
		for _, j := range hard {
			samples = append(samples, nnSample{qIdx: i, rIdx: j, y: 0, w: 1, tier: negHard})
			nHard++
		}
	}
	fmt.Printf("samples: %d  pos=%d  easy_neg=%d  medium_neg=%d  hard_neg=%d\n",
		len(samples), len(queries), nEasy, nMed, nHard)

	// --- class-balanced BCE weights ---
	nPos, nNeg := len(queries), nEasy+nMed+nHard
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
	fmt.Printf("pos=%d neg=%d  wPos=%.2f wNeg=%.2f\n", nPos, nNeg, wPos, wNeg)

	// --- build gallery structures for eval ---
	trainGalleryIdxs := trainRefIdxs
	trainGalleryRefs := makeGalleryRefs(refs, trainGalleryIdxs)
	trainGallerySrcIDs := makeGallerySrcIDs(refSrcIDs, trainGalleryIdxs)
	trainGalleryMap := makeGalleryMap(trainGalleryIdxs)

	valGalleryRefs := makeGalleryRefs(refs, valRefIdxs)
	valGallerySrcIDs := makeGallerySrcIDs(refSrcIDs, valRefIdxs)
	valGalleryMap := makeGalleryMap(valRefIdxs)

	testGalleryRefs := makeGalleryRefs(refs, testRefIdxs)
	testGallerySrcIDs := makeGallerySrcIDs(refSrcIDs, testRefIdxs)
	testGalleryMap := makeGalleryMap(testRefIdxs)

	// --- model + optimizer setup ---
	m := newAttnNet(42)
	adam := newAdam(m)
	rng := rand.New(rand.NewSource(7))
	order := make([]int, len(samples))
	for i := range order {
		order[i] = i
	}
	xbuf := make([]float64, nnInput)

	sn := m.SN
	snLR := 0.02 * lr
	setupSN(sn, len(queries), len(refs))
	trainRefSet := make(map[int]bool, len(trainRefIdxs))
	for _, idx := range trainRefIdxs {
		trainRefSet[idx] = true
	}

	// recomputeEmb: only train refs + train queries (non-train refs get fresh
	// embeddings at eval time via evalAllEmb).
	recomputeEmb := func() {
		for i, r := range refs {
			if !trainRefSet[i] {
				continue
			}
			F, mh, emb := sn.embed(r.Thumb16)
			r.Emb = emb
			copy(snCur.rFs[i*snFMap:], F)
			copy(snCur.rmhs[i*snHid:], mh)
		}
		for i := range queries {
			Fs, mhs, emb, wins := sn.embedRot(queries[i].q.Thumb16)
			queries[i].q.Emb = emb
			for k := 0; k < snRot; k++ {
				copy(snCur.qFs[(i*snRot+k)*snFMap:], Fs[k])
				copy(snCur.qmhs[(i*snRot+k)*snHid:], mhs[k])
			}
			copy(snCur.qWins[i*snEmb:], wins)
		}
		clear(snCur.qAcc)
		clear(snCur.rAcc)
	}

	snBack := func() {
		sg := newSNGrads(sn)
		for i, r := range refs {
			if !trainRefSet[i] {
				continue
			}
			sn.backward(snCur.rAcc[i*snEmb:(i+1)*snEmb], r.Thumb16,
				snCur.rFs[i*snFMap:(i+1)*snFMap], snCur.rmhs[i*snHid:(i+1)*snHid], sg)
		}
		for i := range queries {
			Fs := make([][]float64, snRot)
			mhs := make([][]float64, snRot)
			for k := 0; k < snRot; k++ {
				Fs[k] = snCur.qFs[(i*snRot+k)*snFMap : (i*snRot+k+1)*snFMap]
				mhs[k] = snCur.qmhs[(i*snRot+k)*snHid : (i*snRot+k+1)*snHid]
			}
			sn.backwardRot(snCur.qAcc[i*snEmb:(i+1)*snEmb], queries[i].q.Thumb16,
				Fs, mhs, snCur.qWins[i*snEmb:(i+1)*snEmb], sg)
		}
		snCur.active = false
		adam.snStep(sn, sg, len(samples), snLR)
		snCur.active = true
	}

	// evalAllEmb computes fresh SN embeddings for ALL refs and all eval queries,
	// so val/test galleries use current weights.
	evalAllEmb := func() {
		for _, r := range refs {
			_, _, r.Emb = sn.embed(r.Thumb16)
		}
		for i := range valQueries {
			valQueries[i].q.Emb = sn.embedEmbRot(valQueries[i].q.Thumb16)
		}
		for i := range testQueries {
			testQueries[i].q.Emb = sn.embedEmbRot(testQueries[i].q.Thumb16)
		}
	}

	recomputeEmb()
	nnTraining = true
	best := newAttnNet(42)
	bestValAcc := -1.0
	t0 := time.Now()
	for ep := 0; ep < epochs; ep++ {
		recomputeEmb()
		rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
		var loss float64
		for off := 0; off < len(order); off += batch {
			end := min(off+batch, len(order))
			g := newGrads(m)
			n := end - off
			for _, si := range order[off:end] {
				s := samples[si]
				qu := &queries[s.qIdx]
				snCur.qIdx, snCur.rIdx = s.qIdx, s.rIdx
				nnVecInto(xbuf, qu.q, refs[s.rIdx])
				p, _ := m.forward(xbuf)
				loss += -s.w * (s.y*mathLog(p) + (1-s.y)*mathLog(1-p))
				m.backprop(g, xbuf, s.y, s.w)
			}
			adam.step(m, g, n, lr)
		}
		snBack()
		if ep%8 == 0 || ep == epochs-1 {
			nnTraining = false
			snCur.active = false
			// quick train@1 on a 200-query subset + full val metrics
			for _, r := range refs {
				_, _, r.Emb = sn.embed(r.Thumb16)
			}
			for i := range valQueries {
				valQueries[i].q.Emb = sn.embedEmbRot(valQueries[i].q.Thumb16)
			}
			// sample 200 train queries for a fast train@1 estimate
			trainSubset := queries
			if len(queries) > 200 {
				trainSubset = make([]nnQuery, 200)
				step := len(queries) / 200
				for k := 0; k < 200; k++ {
					trainSubset[k] = queries[k*step]
					trainSubset[k].q.Emb = sn.embedEmbRot(queries[k*step].q.Thumb16)
				}
			}
			trainMt := computeRankMetrics(m, trainSubset, trainGalleryRefs,
				trainGallerySrcIDs, trainGalleryMap)
			valMt := rankMetrics{}
			if len(valQueries) > 0 {
				valMt = computeRankMetrics(m, valQueries, valGalleryRefs,
					valGallerySrcIDs, valGalleryMap)
			}
			if valMt.refRecall1 > bestValAcc {
				bestValAcc = valMt.refRecall1
				best = cloneNet(m)
			}
			nnTraining = true
			snCur.active = true
			fmt.Printf("epoch %d/%d  loss=%.4f  train@1=%.1f%%  val@1=%.1f%%  val@5=%.1f%%  val@10=%.1f%%  mrr=%.3f  [%.1fs]\n",
				ep+1, epochs, loss/float64(len(samples)),
				100*trainMt.refRecall1,
				100*valMt.refRecall1, 100*valMt.refRecall5, 100*valMt.refRecall10,
				valMt.mrr, time.Since(t0).Seconds())
		}
	}
	fmt.Printf("best val@1=%.1f%%, restoring best-epoch weights\n", 100*bestValAcc)
	m = best
	fmt.Printf("epoch phase: %.1fs\n", time.Since(t0).Seconds())
	nnTraining = false
	snCur.active = false

	// --- final evaluation ---
	fmt.Printf("\n=== Final Evaluation ===\n")
	evalAllEmb()

	trainMt := computeRankMetrics(m, queries, trainGalleryRefs,
		trainGallerySrcIDs, trainGalleryMap)
	printMetrics("Train", trainMt)

	if len(valQueries) > 0 {
		valMt := computeRankMetrics(m, valQueries, valGalleryRefs,
			valGallerySrcIDs, valGalleryMap)
		printMetrics("Validation", valMt)
	}
	if len(testQueries) > 0 {
		testMt := computeRankMetrics(m, testQueries, testGalleryRefs,
			testGallerySrcIDs, testGalleryMap)
		printMetrics("Test", testMt)
	}

	// embedding distance diagnostics
	stEmb := computeEmbDistStats(m, queries, refs, samples, refSrcIDs, 2000)
	fmt.Printf("\n  emb pos: mean=%.3f min=%.3f max=%.3f\n", stEmb.posMean, stEmb.posMin, stEmb.posMax)
	fmt.Printf("  emb neg: mean=%.3f min=%.3f max=%.3f\n", stEmb.negMean, stEmb.negMin, stEmb.negMax)
	fmt.Printf("  emb hard: mean=%.3f min=%.3f max=%.3f\n", stEmb.hardMean, stEmb.hardMin, stEmb.hardMax)

	// challenge set (optional)
	runChallengeEval(root, m, refs, refNames, refSrcIDs, trainSrcs, valSrcs, testSrcs, groups)

	if err := m.save(weightsPath); err != nil {
		fatal(err)
	}
	fmt.Printf("\nweights saved to %s\n", weightsPath)
}

func buildNNQuery(q *Feat, trueIdx int) nnQuery {
	return nnQuery{q: q, trueIdx: trueIdx}
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

func trainAt1(m *AttnNet, queries []nnQuery, refs []*Feat) int {
	ok := 0
	xbuf := make([]float64, nnInput)
	for i := range queries {
		qu := &queries[i]
		bestScore, bestIdx := -1.0, -1
		for j := range refs {
			nnVecInto(xbuf, qu.q, refs[j])
			if v := m.predict(xbuf); v > bestScore {
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
// be confusable along color OR shape: the top refs by a hist+zern overlap
// blend, plus a few random refs (usually color-family outsiders), so the model
// learns both within-family shape ties and cross-family rejection. Scoring is
// on-the-fly (scalar per ref, no 545-dim block materialization).
func hardNegs(q *Feat, refs []*Feat, trueIdx, seed int) []int {
	type hd struct {
		idx int
		s   float64
	}
	cands := make([]hd, 0, len(refs)-1)
	for j, r := range refs {
		if j == trueIdx {
			continue
		}
		cands = append(cands, hd{idx: j, s: refOverlap(q, r)})
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].s > cands[b].s })
	sel := make([]int, 0, 16)
	for _, c := range cands[:min(8, len(cands))] {
		sel = append(sel, c.idx)
	}
	rng := rand.New(rand.NewSource(int64(seed)*7919 + 13))
	for k := 0; k < 3; k++ {
		sel = append(sel, rng.Intn(len(refs)))
	}
	seen := make(map[int]bool, len(sel))
	out := make([]int, 0, len(sel))
	for _, j := range sel {
		if j == trueIdx || seen[j] {
			continue
		}
		seen[j] = true
		out = append(out, j)
	}
	return out
}

// refOverlap is the color+shape confusability scalar used to select hard
// negatives: half the HSV histogram intersection plus half the Zernike
// per-mode overlap. No raw blocks are materialized.
func refOverlap(q, r *Feat) float64 {
	var h float64
	for i := range q.Hist {
		if q.Hist[i] < r.Hist[i] {
			h += q.Hist[i]
		} else {
			h += r.Hist[i]
		}
	}
	var z float64
	for i := range q.Zern {
		if q.Zern[i] < r.Zern[i] {
			z += q.Zern[i]
		} else {
			z += r.Zern[i]
		}
	}
	return 0.5*h + 0.5*z
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
	m := &AttnNet{}
	if len(args) > 0 {
		var err error
		if m, err = loadAttnNet(args[0]); err != nil {
			fatal(err)
		}
	} else {
		m = newAttnNet(1)
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
		nn := make([]float64, len(refs))
		pairs := buildPairData(q, refs)
		// gap column: best-to-runner-up on the hist overlap (the min view of
		// the raw hist block, which starts at 2*len(zernModes)), a cheap
		// query-difficulty indicator.
		b1, b2 := -1.0, -1.0
		for j := range pairs {
			v := pairSum(pairs[j][2*len(zernModes) : 2*len(zernModes)+hBins*sBins*vBins])
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
		gap[i] = b1 - b2
		for j, r := range refs {
			sAll[j] = scores(q, r)
			m64[j] = maskScore(q, r)
			m32[j] = maskScoreGrid(q, r, 32)
			sc[j] = shapeContextSim(q, r)
			nn[j] = m.predict(nnVec(pairs[j]))
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
	m, err := loadAttnNet(weightsPath)
	if err != nil {
		fatal(err)
	}
	refs, refNames := buildRefIndex(root)
	fmt.Printf("index: %d reference sprites\n", len(refs))
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
		rows[i] = evalRow{img: e.Image, want: e.Src, nnRank: rankNN(q, refs, m), baseRank: compositeAdaptive(q, refs), ok: true}
	})
	total := 0
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
	fmt.Printf("neural (attn)     : recall@1=%d/%d (%.1f%%)  recall@3=%d/%d (%.1f%%)  recall@5=%d/%d (%.1f%%)\n",
		nn1, total, 100*float64(nn1)/float64(total), nn3, total, 100*float64(nn3)/float64(total), nn5, total, 100*float64(nn5)/float64(total))

	if len(nnMisses) > 0 {
		fmt.Printf("\n-- NN misses (%d) --\n", len(nnMisses))
		for _, m2 := range nnMisses {
			fmt.Printf("%-18s want=%-60s got=%v\n", m2.img, m2.want, m2.got)
		}
	}
}
