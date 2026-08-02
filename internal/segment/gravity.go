// Package segment: gravity.go
// 基于引力的区域聚合。
//
// 当区域划分产生过多区域时（说明划分质量较差），按"引力 + 时间步"模型
// 将区域聚合到至少 MinRegions 个：面积越大引力越大（吸引周围区域融合），
// 但体积越大、距离越远的区域对所需时间步越多——因此小的、近的区域先融合，
// 大的、远的在后时间步才融合。融合后的新区域 bbox 包住所有被融合区域。
// 聚合结果作为辅助区域参与检索。
package segment

import (
	"image"
	"image/color"
	"math"
	"sort"

	"go-image-search/internal/phash"
)

// GravityConfig 引力聚合参数。
type GravityConfig struct {
	Enabled      bool // 是否启用
	MinRegions   int  // 至少保留的区域数（达到后停止聚合），默认 5
	TriggerCount int  // 区域数超过该值才启动聚合，默认 5（>5 即触发）
	Additive     bool // true：仅返回组合区域（≥2 成员）作为辅助追加；false：返回聚合后完整集合替换原区域
}

// DefaultGravityConfig 推荐默认参数。
func DefaultGravityConfig() GravityConfig {
	return GravityConfig{
		Enabled:      true,
		MinRegions:   5,
		TriggerCount: 5,
		Additive:     true,
	}
}

// GravityMerge 基于引力的区域聚合。
//
// 当 len(regions) > TriggerCount 时，反复选择"所需时间步"最小的区域对融合，
// 直到区域数 ≤ MinRegions。所需时间步 = 质心欧氏距离 × (面积_i + 面积_j)：
// 距离越远、体积（面积和）越大则时间步越多，因此小的、近的区域先融合，
// 大的、远的在后时间步才融合。每轮循环（一次融合）视为一个时间步。
// 融合后区域 bbox 为成员 bbox 之并（包住被融合区域），面积为成员面积之和，
// 平均色按面积加权，并对并集 bbox 重新计算感知哈希与结构哈希。
//
// 面积越大引力越大体现在：大区域作为"吸引中心"，其与小区域的配对因距离近、
// 而另一方小，所需时间步适中，会在中期被融合；最终大区域吸纳周围小区域。
//
// Additive=true 时仅返回组合区域（≥2 成员，重新编号）供调用方追加为辅助区域；
// Additive=false 时返回聚合后的完整区域集合（替换原区域）。
// 未触发聚合（len ≤ TriggerCount）时 Additive 模式返回 nil，replace 模式返回原集合副本。
func GravityMerge(src image.Image, regions []MergedRegion, cfg GravityConfig) []MergedRegion {
	if !cfg.Enabled || len(regions) < 2 {
		if cfg.Additive {
			return nil
		}
		return cloneMerged(regions)
	}
	if cfg.MinRegions < 1 {
		cfg.MinRegions = 1
	}
	if cfg.TriggerCount < cfg.MinRegions {
		cfg.TriggerCount = cfg.MinRegions
	}
	if len(regions) <= cfg.TriggerCount {
		if cfg.Additive {
			return nil
		}
		return cloneMerged(regions)
	}

	regs := cloneMerged(regions)
	centroids := make([][2]float64, len(regs))
	for i, r := range regs {
		centroids[i] = [2]float64{
			float64(r.BBox.Min.X+r.BBox.Max.X) / 2,
			float64(r.BBox.Min.Y+r.BBox.Max.Y) / 2,
		}
	}

	// 反复融合，直到区域数 ≤ MinRegions。每轮一个时间步。
	for len(regs) > cfg.MinRegions {
		bestI, bestJ := -1, -1
		bestT := math.Inf(1)
		for i := 0; i < len(regs); i++ {
			ci := centroids[i]
			for j := i + 1; j < len(regs); j++ {
				cj := centroids[j]
				dx := ci[0] - cj[0]
				dy := ci[1] - cj[1]
				dist := math.Sqrt(dx*dx + dy*dy)
				// 所需时间步：距离 × 体积（面积和）。小且近 → 时间步少 → 先融合。
				t := dist * float64(regs[i].Area+regs[j].Area)
				if t < bestT {
					bestT = t
					bestI, bestJ = i, j
				}
			}
		}
		if bestI < 0 {
			break
		}
		merged := mergeTwoMerged(src, regs[bestI], regs[bestJ])
		regs[bestI] = merged
		centroids[bestI] = [2]float64{
			float64(merged.BBox.Min.X + merged.BBox.Max.X) / 2,
			float64(merged.BBox.Min.Y + merged.BBox.Max.Y) / 2,
		}
		last := len(regs) - 1
		if bestJ != last {
			regs[bestJ] = regs[last]
			centroids[bestJ] = centroids[last]
		}
		regs = regs[:last]
		centroids = centroids[:last]
	}

	// 排序并重新编号（按最小成员 ID 确定顺序，保证确定性）
	sort.SliceStable(regs, func(i, j int) bool {
		return minID(regs[i].Members) < minID(regs[j].Members)
	})
	if cfg.Additive {
		out := make([]MergedRegion, 0, len(regs))
		for _, r := range regs {
			if len(r.Members) >= 2 {
				out = append(out, r)
			}
		}
		for i := range out {
			out[i].ID = i + 1
		}
		return out
	}
	for i := range regs {
		regs[i].ID = i + 1
	}
	return regs
}

// mergeTwoMerged 融合两个区域为一个组合区域（bbox 为成员之并，重算哈希）。
func mergeTwoMerged(src image.Image, a, b MergedRegion) MergedRegion {
	members := make([]int, 0, len(a.Members)+len(b.Members))
	members = append(members, a.Members...)
	members = append(members, b.Members...)
	area := a.Area + b.Area
	bbox := a.BBox.Union(b.BBox)
	var sr, sg, sb int64
	sr += int64(a.MeanColor.R) * int64(a.Area)
	sg += int64(a.MeanColor.G) * int64(a.Area)
	sb += int64(a.MeanColor.B) * int64(a.Area)
	sr += int64(b.MeanColor.R) * int64(b.Area)
	sg += int64(b.MeanColor.G) * int64(b.Area)
	sb += int64(b.MeanColor.B) * int64(b.Area)
	mean := color.RGBA{A: 255}
	if area > 0 {
		mean = color.RGBA{
			R: uint8(sr / int64(area)),
			G: uint8(sg / int64(area)),
			B: uint8(sb / int64(area)),
			A: 255,
		}
	}
	mr := MergedRegion{
		Members:   members,
		Area:      area,
		BBox:      bbox,
		MeanColor: mean,
	}
	if crop := cropRect(src, bbox); crop != nil {
		mr.Hash = phash.Hash(crop)
		mr.Shape = phash.Hash(structuralMask(crop))
	}
	return mr
}

// cloneMerged 返回 regions 的副本（Members 切片独立，避免共享底层数组）。
func cloneMerged(regions []MergedRegion) []MergedRegion {
	out := make([]MergedRegion, len(regions))
	for i, r := range regions {
		m := make([]int, len(r.Members))
		copy(m, r.Members)
		r.Members = m
		out[i] = r
	}
	return out
}
