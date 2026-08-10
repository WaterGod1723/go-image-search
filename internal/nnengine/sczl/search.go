package sczl

import (
	"math"
	"runtime"
	"sort"
	"sync"
)

// search.go SCZL 检索：占据栅格 + FD 全局签名粗排 → SC 匈牙利精排 + HOG 梯度 → 融合打分。
// 占据栅格直接刻画内部结构（主判别力，免疫颜色/背景变化）；
// NCC 互相关提供亮度不变的全图结构匹配（替代脆弱的 pHash）；
// HOG 梯度方向提供颜色无关的边缘结构判别（替代脆弱的 LBP）；
// FD 提供外轮廓全局形状；SC 提供点分布局部形状。
// 全局签名先做低成本的余弦/相交召回控制候选数，SC 匈牙利仅在候选集上做。

// cosine 两向量的余弦相似度 [0,1]（值非负时）。向量已 L2 归一化时即点积。
func cosine(a, b []float64) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	dot, na, nb := 0.0, 0.0, 0.0
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na <= 0 || nb <= 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// occSim 占据栅格相似度（基于孔洞填充后的占据栅格）：粗分辨率更鲁棒。
func occSim(q, e Descriptor) float64 {
	c64 := cosine(q.Occupancy64, e.Occupancy64)
	c32 := cosine(q.Occupancy32, e.Occupancy32)
	c16 := cosine(q.Occupancy16, e.Occupancy16)
	return 0.25*c64 + 0.35*c32 + 0.40*c16
}

// histIntersect 归一化直方图交集 [0,1]。
func histIntersect(a, b []float64) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	s := 0.0
	for i := range a {
		x := a[i]
		if b[i] < x {
			x = b[i]
		}
		s += x
	}
	return s
}

// nccSim 归一化互相关相似度 [−1,1]→[0,1]：两去均值 L2 归一化灰度块的点积。
// 比 pHash 保留完整空间结构，且对亮度/对比度仿射变化不变。
func nccSim(q, e Descriptor) float64 {
	if len(q.Patch) == 0 || len(e.Patch) == 0 || len(q.Patch) != len(e.Patch) {
		return 0
	}
	dot := 0.0
	for i := range q.Patch {
		dot += q.Patch[i] * e.Patch[i]
	}
	if dot < 0 {
		return 0
	}
	return dot
}

// hogSim HOG 梯度方向直方图余弦相似度 [0,1]。
func hogSim(q, e Descriptor) float64 {
	return cosine(q.HOG, e.HOG)
}

// globalSimRot 旋转不变的全局签名相似度，用于粗排阶段。
// occ/ncc 使用预计算的 query 旋转副本取最佳对齐（旋转不变），
// 同时取 max(Rmax-旋转扫描, BBox-紧裁剪) 兼顾非旋转精度。
// reg/fd/rad 本身旋转不变。ang=0 步即原 occSim/nccSim，非旋转场景不退化。
// 权重偏向旋转不变特征（fd/rad/reg 共 60%），因为查询图（彩色背景上的截图）
// 的前景提取比图库透明 PNG 噪声更大，occ/ncc 即使旋转扫描也因掩码差异而偏低。
func globalSimRot(q, e Descriptor, qOccRots, qNCCRots [][]float64) float64 {
	occ := bestOccDot(qOccRots, e.Occupancy64, q.Occupancy64)
	if occBBox := cosine(q.OccBBox64, e.OccBBox64); occBBox > occ {
		occ = occBBox
	}
	ncc := bestNCCDot(qNCCRots, e.Patch, q.Patch)
	if nccBBox := dot(q.PatchBBox, e.PatchBBox); nccBBox > ncc {
		ncc = nccBBox
	}
	reg := regionSim(q, e)
	fd := cosine(q.Fourier, e.Fourier)
	rad := histIntersect(q.Radial, e.Radial)
	return 0.20*occ + 0.20*ncc + 0.20*reg + 0.25*fd + 0.15*rad
}

// regionSim 多区域匈牙利匹配得分 [0,1]。
// 部件数量/位置/形状一致才得高分：dashboard(4 部件) vs dashboard(4 部件)≈1，
// vs chaoshibianli(1 部件) 仅 1 对可配 → 覆盖低 → 低分。
func regionSim(q, e Descriptor) float64 {
	qr, er := q.Regions, e.Regions
	if len(qr) == 0 || len(er) == 0 {
		return 0
	}
	nq, ne := len(qr), len(er)
	// 代价 = 0.7×(1-cos 占据) + 0.3×质心距离。
	cost := make([][]float64, nq)
	for i := range cost {
		cost[i] = make([]float64, ne)
		for j := range cost[i] {
			cos := cosine(qr[i].Occ, er[j].Occ)
			// 旋转不变：质心到图标中心的径向距离差（旋转只改变角度，径向距离不变）。
			qRad := math.Hypot(qr[i].NX-0.5, qr[i].NY-0.5)
			eRad := math.Hypot(er[j].NX-0.5, er[j].NY-0.5)
			cdist := math.Abs(qRad - eRad)
			if cdist > 1 {
				cdist = 1
			}
			cost[i][j] = 0.7*(1-cos) + 0.3*cdist
		}
	}
	assign := hungarian(cost)
	matchedW, scoreSum, totalW := 0.0, 0.0, 0.0
	for _, r := range qr {
		totalW += float64(r.Area)
	}
	for i, j := range assign {
		if j < 0 {
			continue
		}
		sim := 1 - cost[i][j]
		if sim < 0 {
			sim = 0
		}
		w := float64(qr[i].Area)
		matchedW += w
		scoreSum += w * sim
	}
	if totalW <= 0 || matchedW <= 0 {
		return 0
	}
	avgSim := scoreSum / matchedW
	coverage := matchedW / totalW
	return avgSim * coverage
}

// candidate 粗排候选条目。
type candidate struct {
	idx int
	g   float64 // 全局相似度
}

// rerank 精排候选条目：携带各维度相似度得分，供自适应权重计算与融合打分。
type rerank struct {
	idx                        int
	occ, ncc, reg, hog, sc, fd float64
}

// scSolidityThr 实度低于此值视为镂空/碎片化，SC 匹配置零（轮廓不可靠）。
const scSolidityThr = 0.4

// Query 用查询描述子检索，返回按融合得分降序的 top-K。
func (ix *Index) Query(q Descriptor, opts Options) []Match {
	if !q.Valid || len(ix.Entries) == 0 {
		return nil
	}
	if opts.TopK <= 0 {
		opts.TopK = 5
	}
	if opts.OccWeight <= 0 {
		opts.OccWeight = 0.20
	}
	if opts.NccWeight <= 0 {
		opts.NccWeight = 0.20
	}
	if opts.RegWeight <= 0 {
		opts.RegWeight = 0.20
	}
	if opts.HogWeight <= 0 {
		opts.HogWeight = 0.10
	}
	if opts.SCWeight <= 0 {
		opts.SCWeight = 0.10
	}
	if opts.FDWeight <= 0 {
		opts.FDWeight = 0.20
	}
	if opts.FDPreFilter <= 0 {
		opts.FDPreFilter = 200
	}
	if opts.FDCut < 0 {
		opts.FDCut = 0.0
	}
	if opts.SCMaxPoints <= 0 {
		opts.SCMaxPoints = 48
	}
	if opts.RotSteps <= 0 {
		opts.RotSteps = 36
	}
	rotSteps := opts.RotSteps
	// 权重归一化。
	wSum := opts.OccWeight + opts.NccWeight + opts.RegWeight + opts.HogWeight + opts.SCWeight + opts.FDWeight
	if wSum <= 0 {
		wSum = 1
	}
	wOcc := opts.OccWeight / wSum
	wNCC := opts.NccWeight / wSum
	wReg := opts.RegWeight / wSum
	wHOG := opts.HogWeight / wSum
	wSC := opts.SCWeight / wSum
	wFD := opts.FDWeight / wSum

	// 1) 全局签名粗排（并行，旋转不变）。
	// 预计算 query 的 occ/ncc 旋转副本，粗排时每候选只需点积取最大。
	nEnt := len(ix.Entries)
	qOccRots := precomputeOccRots(q.Occupancy64, rotSteps)
	qNCCRots := precomputeNCCRots(q.Patch, rotSteps)
	gScores := make([]float64, nEnt)
	workers := runtime.NumCPU()
	if workers > nEnt {
		workers = nEnt
	}
	if workers < 1 {
		workers = 1
	}
	var wg sync.WaitGroup
	chunk := (nEnt + workers - 1) / workers
	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if end > nEnt {
			end = nEnt
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				gScores[i] = globalSimRot(q, ix.Entries[i], qOccRots, qNCCRots)
			}
		}(start, end)
	}
	wg.Wait()

	cands := make([]candidate, 0, nEnt)
	for i := 0; i < nEnt; i++ {
		if gScores[i] < opts.FDCut {
			continue
		}
		cands = append(cands, candidate{idx: i, g: gScores[i]})
	}
	if len(cands) == 0 {
		for i := 0; i < nEnt; i++ {
			cands = append(cands, candidate{idx: i, g: gScores[i]})
		}
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].g > cands[b].g })
	if len(cands) > opts.FDPreFilter {
		cands = cands[:opts.FDPreFilter]
	}

	// 2) 精排（并行）：占据栅格 + NCC + 区域匹配 + HOG + SC + FD。
	//    每候选独立计算 6 维得分，按索引写入预分配数组无竞争。
	//    最重的 scRotSim（18 步旋转+匈牙利）和 maskRotSim（18 步旋转 cosine）并行后
	//    吞吐量随 CPU 核数线性提升。
	//    qNCCRots 已在粗排前预计算（与粗排共用）。
	nCand := len(cands)
	reranks := make([]rerank, nCand)
	workers2 := workers
	if workers2 > nCand {
		workers2 = nCand
	}
	if workers2 < 1 {
		workers2 = 1
	}
	chunk2 := (nCand + workers2 - 1) / workers2
	for w := 0; w < workers2; w++ {
		start := w * chunk2
		end := start + chunk2
		if end > nCand {
			end = nCand
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for k := s; k < e; k++ {
				c := cands[k]
				entry := ix.Entries[c.idx]
				// occ 维度：取 max(Rmax-旋转扫描, BBox-紧裁剪直接匹配)。
				occ := maskRotSim(q.Occupancy64, entry.Occupancy64, rotSteps)
				if occBBox := cosine(q.OccBBox64, entry.OccBBox64); occBBox > occ {
					occ = occBBox
				}
				// ncc 维度：取 max(Rmax-旋转扫描, BBox-紧裁剪直接匹配)。
				ncc := bestNCCDot(qNCCRots, entry.Patch, q.Patch)
				if nccBBox := dot(q.PatchBBox, entry.PatchBBox); nccBBox > ncc {
					ncc = nccBBox
				}
				reg := regionSim(q, entry)
				hog := hogRotSim(q.HOG, entry.HOG)
				scSimVal := scRotSim(q, entry, opts.SCMaxPoints, rotSteps)
				fdSim := cosine(q.Fourier, entry.Fourier)
				reranks[k] = rerank{
					idx: c.idx, occ: occ, ncc: ncc, reg: reg, hog: hog, sc: scSimVal, fd: fdSim,
				}
			}
		}(start, end)
	}
	wg.Wait()

	// 融合权重：启用自适应时按候选集各维度得分分布动态计算，否则用固定先验。
	awOcc, awNCC, awReg, awHOG, awSC, awFD := wOcc, wNCC, wReg, wHOG, wSC, wFD
	if opts.Adaptive && len(reranks) >= 3 {
		awOcc, awNCC, awReg, awHOG, awSC, awFD = adaptiveWeights(reranks, opts)
	}

	matches := make([]Match, 0, len(reranks))
	for _, r := range reranks {
		e := ix.Entries[r.idx]
		score := awOcc*r.occ + awNCC*r.ncc + awReg*r.reg + awHOG*r.hog + awFD*r.fd + awSC*r.sc
		matches = append(matches, Match{
			ImageID: e.ImageID,
			Score:   score,
			FDSim:   r.fd,
			SCSim:   r.sc,
			HOGSim:  r.hog,
		})
	}

	sort.SliceStable(matches, func(a, b int) bool {
		if matches[a].Score != matches[b].Score {
			return matches[a].Score > matches[b].Score
		}
		return matches[a].ImageID < matches[b].ImageID
	})
	if len(matches) > opts.TopK {
		matches = matches[:opts.TopK]
	}
	return matches
}

// adaptiveWeights 基于 query 在精排候选集上的各维度得分分布，计算自适应融合权重。
//
// 直觉：某维度"头部与主体分离越明显"（top1 显著高于 top2..topK 的均值），
// 说明该维度对当前 query 区分度越强，应给更高权重；各候选得分挤在一起、
// 差值很小的维度区分度弱，降权。用相邻两点差值太容易被单个噪声样本扰动，
// 故改用"top1 相对 top2..topK 主体的 z-score"作为区分度信号，更稳健。
//
//	对每个维度 d 计算 z-score（无量纲，消除各维度量纲差异）：
//	  disc(d) = (top1 - mean(top2..topK)) / std(top2..topK)
//	再 softmax(disc/T) 得到自适应权重分布，与先验固定权重按 α 混合、上下界裁剪后归一化：
//	  final_w(d) = clip(α·prior_norm(d) + (1-α)·softmax(disc/T)(d), lo, hi)
//	所有维度 disc ≤ 0（无正区分度）时回退先验，保证退化安全。
func adaptiveWeights(rs []rerank, opts Options) (wOcc, wNCC, wReg, wHOG, wSC, wFD float64) {
	alpha := opts.AdaptiveAlpha
	if alpha <= 0 {
		alpha = 0.5
	}
	temp := opts.AdaptiveTemp
	if temp <= 0 {
		temp = 1.0
	}
	k := opts.AdaptiveK
	if k <= 0 {
		k = 10
	}
	lo := opts.AdaptiveLo
	if lo <= 0 {
		lo = 0.05
	}
	hi := opts.AdaptiveHi
	if hi <= 0 {
		hi = 0.5
	}

	const dims = 6
	prior := [dims]float64{opts.OccWeight, opts.NccWeight, opts.RegWeight, opts.HogWeight, opts.SCWeight, opts.FDWeight}
	pSum := 0.0
	for _, p := range prior {
		pSum += p
	}
	if pSum <= 0 {
		pSum = 1
	}
	var pNorm [dims]float64
	for d := 0; d < dims; d++ {
		pNorm[d] = prior[d] / pSum
	}

	// 收集各维度得分列。
	var cols [dims][]float64
	for d := 0; d < dims; d++ {
		cols[d] = make([]float64, len(rs))
	}
	for i, r := range rs {
		cols[0][i] = r.occ
		cols[1][i] = r.ncc
		cols[2][i] = r.reg
		cols[3][i] = r.hog
		cols[4][i] = r.sc
		cols[5][i] = r.fd
	}

	var disc [dims]float64
	hasPos := false
	for d := 0; d < dims; d++ {
		s := append([]float64(nil), cols[d]...)
		sort.Sort(sort.Reverse(sort.Float64Slice(s)))
		kk := k
		if kk > len(s) {
			kk = len(s)
		}
		if kk < 3 {
			disc[d] = 0
			continue
		}
		top1 := s[0]
		rest := s[1:kk]
		mean, std := meanStd(rest)
		if std <= 1e-9 {
			disc[d] = 0
			continue
		}
		z := (top1 - mean) / std
		if z < 0 {
			z = 0
		}
		disc[d] = z
		if z > 0 {
			hasPos = true
		}
	}

	// 无正区分度：回退先验。
	if !hasPos {
		return pNorm[0], pNorm[1], pNorm[2], pNorm[3], pNorm[4], pNorm[5]
	}

	// softmax(disc/T)（数值稳定版：减最大值）。
	maxZ := 0.0
	for _, z := range disc {
		if z > maxZ {
			maxZ = z
		}
	}
	expSum := 0.0
	var expZ [dims]float64
	for d := 0; d < dims; d++ {
		expZ[d] = math.Exp((disc[d] - maxZ) / temp)
		expSum += expZ[d]
	}
	if expSum <= 0 {
		expSum = 1
	}
	var adaptive [dims]float64
	for d := 0; d < dims; d++ {
		adaptive[d] = expZ[d] / expSum
	}

	// α·先验 + (1-α)·自适应，裁剪后归一化。
	var w [dims]float64
	wSum := 0.0
	for d := 0; d < dims; d++ {
		v := alpha*pNorm[d] + (1-alpha)*adaptive[d]
		if v < lo {
			v = lo
		} else if v > hi {
			v = hi
		}
		w[d] = v
		wSum += v
	}
	if wSum <= 0 {
		wSum = 1
	}
	for d := 0; d < dims; d++ {
		w[d] /= wSum
	}
	return w[0], w[1], w[2], w[3], w[4], w[5]
}

// meanStd 样本均值与总体标准差。
func meanStd(s []float64) (mean, std float64) {
	if len(s) == 0 {
		return 0, 0
	}
	sum := 0.0
	for _, v := range s {
		sum += v
	}
	mean = sum / float64(len(s))
	var ss float64
	for _, v := range s {
		d := v - mean
		ss += d * d
	}
	std = math.Sqrt(ss / float64(len(s)))
	return
}

// GlobalScoresAll returns, for every index entry, the cheap color-agnostic
// global similarity (rotation-scanned occupancy + NCC + region + Fourier +
// radial, same fusion as the coarse phase of Query). SC and HOG are excluded
// because they are expensive; the caller can fuse this as an extra expert
// signal per (query, ref) pair without running the full Query.
func (ix *Index) GlobalScoresAll(q Descriptor, rotSteps int) []float64 {
	if !q.Valid {
		return nil
	}
	pq := PrepareQuery(q, rotSteps)
	out := make([]float64, len(ix.Entries))
	for i := range ix.Entries {
		out[i] = pq.GlobalScoreOf(ix.Entries[i])
	}
	return out
}

// PreparedQuery holds a query descriptor plus its precomputed rotation copies,
// so per-reference global similarity can be scored one entry at a time (cheap
// pre-filter / two-stage search) instead of scanning the whole index.
type PreparedQuery struct {
	q       Descriptor
	occRots [][]float64
	nccRots [][]float64
}

// PrepareQuery precomputes a query's rotation copies for occ/NCC matching.
func PrepareQuery(q Descriptor, rotSteps int) *PreparedQuery {
	if rotSteps <= 0 {
		rotSteps = 36
	}
	return &PreparedQuery{
		q:       q,
		occRots: precomputeOccRots(q.Occupancy64, rotSteps),
		nccRots: precomputeNCCRots(q.Patch, rotSteps),
	}
}

// GlobalScoreOf is the cheap color-agnostic global similarity (occ+ncc+region+
// Fourier+radial) of this prepared query against one reference descriptor.
func (pq *PreparedQuery) GlobalScoreOf(e Descriptor) float64 {
	return globalSimRot(pq.q, e, pq.occRots, pq.nccRots)
}

// GlobalScoresRow returns the individual rotation-invariant sub-signals of the
// global similarity as separate scalars: [occ, ncc, region, fourier, radial,
// hog]. The NN pair model consumes these separately (instead of the single
// fused GlobalScoreOf) so it can learn each sub-signal's weight — they have
// markedly different error sets (e.g. Fourier/radial outline vs occupancy
// internal structure vs HOG gradient orientation). Each is in [0,1].
func (pq *PreparedQuery) GlobalScoresRow(e Descriptor) []float64 {
	q := pq.q
	occ := bestOccDot(pq.occRots, e.Occupancy64, q.Occupancy64)
	if occBBox := cosine(q.OccBBox64, e.OccBBox64); occBBox > occ {
		occ = occBBox
	}
	ncc := bestNCCDot(pq.nccRots, e.Patch, q.Patch)
	if nccBBox := dot(q.PatchBBox, e.PatchBBox); nccBBox > ncc {
		ncc = nccBBox
	}
	reg := regionSim(q, e)
	fd := cosine(q.Fourier, e.Fourier)
	rad := histIntersect(q.Radial, e.Radial)
	return []float64{occ, ncc, reg, fd, rad, hogSim(q, e)}
}
