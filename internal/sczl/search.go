package sczl

import (
	"math"
	"sort"
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

// globalSim 全局签名相似度：占据栅格(孔洞填充) + NCC 互相关 + 区域匹配 为主，FD 外轮廓 + 径向为辅。
// 用于粗排阶段（轻量，不含 SC/匈牙利）。
func globalSim(q, e Descriptor) float64 {
	occ := occSim(q, e)
	ncc := nccSim(q, e)
	reg := regionSim(q, e)
	fd := cosine(q.Fourier, e.Fourier)
	rad := histIntersect(q.Radial, e.Radial)
	return 0.30*occ + 0.30*ncc + 0.15*reg + 0.15*fd + 0.10*rad
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
			cdist := math.Hypot(qr[i].NX-er[j].NX, qr[i].NY-er[j].NY)
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
		opts.OccWeight = 0.25
	}
	if opts.NccWeight <= 0 {
		opts.NccWeight = 0.25
	}
	if opts.RegWeight <= 0 {
		opts.RegWeight = 0.15
	}
	if opts.HogWeight <= 0 {
		opts.HogWeight = 0.15
	}
	if opts.SCWeight <= 0 {
		opts.SCWeight = 0.10
	}
	if opts.FDWeight <= 0 {
		opts.FDWeight = 0.10
	}
	if opts.FDPreFilter <= 0 {
		opts.FDPreFilter = 200
	}
	if opts.FDCut <= 0 {
		opts.FDCut = 0.4
	}
	if opts.SCMaxPoints <= 0 {
		opts.SCMaxPoints = 48
	}
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

	// 1) 全局签名粗排。
	cands := make([]candidate, 0, len(ix.Entries))
	for i := range ix.Entries {
		g := globalSim(q, ix.Entries[i])
		if g < opts.FDCut {
			continue
		}
		cands = append(cands, candidate{idx: i, g: g})
	}
	if len(cands) == 0 {
		for i := range ix.Entries {
			cands = append(cands, candidate{idx: i, g: globalSim(q, ix.Entries[i])})
		}
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].g > cands[b].g })
	if len(cands) > opts.FDPreFilter {
		cands = cands[:opts.FDPreFilter]
	}

	// 2) 精排：占据栅格 + NCC + 区域匹配 + HOG + SC + FD。
	matches := make([]Match, 0, len(cands))
	for _, c := range cands {
		e := ix.Entries[c.idx]
		occ := occSim(q, e)
		ncc := nccSim(q, e)
		reg := regionSim(q, e)
		hog := hogSim(q, e)
		scSimVal := 0.0
		if len(q.SC) > 0 && len(e.SC) > 0 && q.Solidity >= scSolidityThr && e.Solidity >= scSolidityThr {
			cost := scMatchCost(q.SC, e.SC, opts.SCMaxPoints)
			if cost != nil {
				assign := hungarian(cost)
				scSimVal = scSimilarity(cost, assign)
			}
		}
		fdSim := c.g
		score := wOcc*occ + wNCC*ncc + wReg*reg + wHOG*hog + wFD*fdSim + wSC*scSimVal
		matches = append(matches, Match{
			ImageID: e.ImageID,
			Score:   score,
			FDSim:   fdSim,
			SCSim:   scSimVal,
			HOGSim:  hog,
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
