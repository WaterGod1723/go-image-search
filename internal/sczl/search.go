package sczl

import (
	"math"
	"sort"
)

// search.go SCZL 检索：占据栅格 + FD 全局签名粗排 → SC 匈牙利精排 + LBP 纹理 → 融合打分。
// 占据栅格直接刻画内部结构（主判别力，免疫颜色/背景变化）；
// FD 提供外轮廓全局形状；SC 提供点分布局部形状；LBP 提供纹理。
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

// occSim 占据栅格相似度：粗分辨率更鲁棒（抗子像素对齐/反锯齿差异），权重更高。
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

// globalSim 全局签名相似度：占据栅格为主，FD 外轮廓 + 径向为辅。
// 用于粗排阶段（轻量，不含 SC）。
func globalSim(q, e Descriptor) float64 {
	occ := occSim(q, e)
	fd := cosine(q.Fourier, e.Fourier)
	rad := histIntersect(q.Radial, e.Radial)
	return 0.75*occ + 0.18*fd + 0.07*rad
}

// candidate 粗排候选条目。
type candidate struct {
	idx int
	g   float64 // 全局相似度
}

// Query 用查询描述子检索，返回按融合得分降序的 top-K。
func (ix *Index) Query(q Descriptor, opts Options) []Match {
	if !q.Valid || len(ix.Entries) == 0 {
		return nil
	}
	if opts.TopK <= 0 {
		opts.TopK = 5
	}
	if opts.OccWeight <= 0 {
		opts.OccWeight = 0.5
	}
	if opts.SCWeight <= 0 {
		opts.SCWeight = 0.15
	}
	if opts.FDWeight <= 0 {
		opts.FDWeight = 0.2
	}
	if opts.LBPWeight <= 0 {
		opts.LBPWeight = 0.15
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
	wSum := opts.OccWeight + opts.SCWeight + opts.FDWeight + opts.LBPWeight
	if wSum <= 0 {
		wSum = 1
	}
	wOcc := opts.OccWeight / wSum
	wSC := opts.SCWeight / wSum
	wFD := opts.FDWeight / wSum
	wLBP := opts.LBPWeight / wSum

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

	// 2) 精排：占据栅格 + SC + LBP + FD。
	matches := make([]Match, 0, len(cands))
	for _, c := range cands {
		e := ix.Entries[c.idx]
		occ := occSim(q, e)
		scSim := 0.0
		if len(q.SC) > 0 && len(e.SC) > 0 {
			cost := scMatchCost(q.SC, e.SC, opts.SCMaxPoints)
			if cost != nil {
				assign := hungarian(cost)
				scSim = scSimilarity(cost, assign)
			}
		}
		lbpSim := lbpIntersect(q.LBP, e.LBP)
		fdSim := c.g // 粗排全局分（含 occ+fd+rad）
		score := wOcc*occ + wFD*fdSim + wLBP*lbpSim + wSC*scSim
		matches = append(matches, Match{
			ImageID: e.ImageID,
			Score:   score,
			FDSim:   fdSim,
			SCSim:   scSim,
			LBPSim:  lbpSim,
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
