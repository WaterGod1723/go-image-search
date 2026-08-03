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
	CombineFew   int  // 最终区域数少于该值时，将全部最终区域合并为一个整体辅助索引区域；0 表示不启用，默认 5
	CombineAlways bool // true：无论区域数多少都无条件追加整图辅助区域（用于查询阶段，不受 CombineFew 门槛限制）
	FrameRatio   float64 // 丢弃 bbox 覆盖图像宽、高均 ≥ 该比例的边框区域（可能为无效边框/背景）；0 表示不启用，默认 0.9
}

// DefaultGravityConfig 推荐默认参数。
func DefaultGravityConfig() GravityConfig {
	return GravityConfig{
		Enabled:       true,
		MinRegions:    5,
		TriggerCount:  5,
		Additive:      true,
		CombineFew:    5,
		CombineAlways: true,
		FrameRatio:    0.9,
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

// FilterFullFrame 丢弃与图像边界几乎重合的区域（可能为无效边框/背景）。
// 区域 bbox 覆盖图像宽度与高度的比例均 ≥ ratio 时判定为边框区域并丢弃。
// ratio ≤ 0 或尺寸无效时原样返回。返回新切片，不修改入参。
func FilterFullFrame(regions []MergedRegion, w, h int, ratio float64) []MergedRegion {
	if ratio <= 0 || len(regions) == 0 || w <= 0 || h <= 0 {
		return regions
	}
	out := make([]MergedRegion, 0, len(regions))
	for _, r := range regions {
		if float64(r.BBox.Dx()) >= float64(w)*ratio && float64(r.BBox.Dy()) >= float64(h)*ratio {
			continue
		}
		out = append(out, r)
	}
	return out
}

// ShouldAddWholeAux 判断是否需要追加"整图"辅助区域（绿框）。
// 触发依据优先取蓝色框（引力辅助组合区域）数量，其次才以最初的红框（划分区域）数量为准：
//   - CombineAlways 时无条件追加（仍需 partition >= 2）；
//   - 存在蓝色辅助区域时，蓝色数量 < CombineFew 即追加；
//   - 否则回退到最初划分区域（红框）数量 < CombineFew。
func ShouldAddWholeAux(g GravityConfig, partition, aux []MergedRegion) bool {
	if !g.Enabled || g.CombineFew <= 0 || len(partition) < 2 {
		return false
	}
	if g.CombineAlways {
		return true
	}
	if len(aux) > 0 {
		return len(aux) < g.CombineFew
	}
	return len(partition) < g.CombineFew
}

// MergeContainedOverlapping 合并 bbox 相交或相互包含的辅助区域为新的组合区域。
// 相交或包含（A 在 B 内或 B 在 A 内）的区域归为一组，组内依次融合为一个新区域
// （bbox 为成员之并、面积求和、哈希与平均色重算），以提升每个索引区域的全局信息，
// 避免碎片化区域影响召回。结果按最小成员 ID 确定性排序；无合并时原样返回副本。
func MergeContainedOverlapping(src image.Image, regions []MergedRegion) []MergedRegion {
	if len(regions) < 2 {
		return cloneMerged(regions)
	}
	n := len(regions)
	parent := make([]int, n)
	for i := range parent {
		parent[i] = i
	}
	var find func(int) int
	find = func(x int) int {
		if parent[x] != x {
			parent[x] = find(parent[x])
		}
		return parent[x]
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			// 相交或相互包含：交集非空即满足
			if !regions[i].BBox.Intersect(regions[j].BBox).Empty() {
				ri, rj := find(i), find(j)
				if ri != rj {
					parent[rj] = ri
				}
			}
		}
	}
	groups := make(map[int][]MergedRegion)
	for i := range regions {
		root := find(i)
		groups[root] = append(groups[root], regions[i])
	}
	out := make([]MergedRegion, 0, len(groups))
	for _, g := range groups {
		if len(g) == 1 {
			out = append(out, g[0])
			continue
		}
		m := g[0]
		for _, r := range g[1:] {
			m = mergeTwoMerged(src, m, r)
		}
		out = append(out, m)
	}
	sort.SliceStable(out, func(i, j int) bool {
		return minID(out[i].Members) < minID(out[j].Members)
	})
	return out
}

// GlobalWeight 返回区域包含的"全局信息"权重（≥1）：
// 整图辅助区域（绿框）权重最高；组合辅助区域（蓝框）随成员数递增；
// 单个划分区域权重为 1。用于召回阶段提升全局信息多区域的权重。
func GlobalWeight(m MergedRegion) float64 {
	if m.Whole {
		return 3.0
	}
	n := len(m.Members)
	if n <= 1 {
		return 1.0
	}
	return 1 + 0.5*math.Log(float64(n)) // 2→1.35, 4→1.69, 8→2.04
}

// MergeAll 将全部区域合并为一个整体组合区域（bbox 为成员之并，重算哈希与平均色）。
// 用于最终区域数过少时构建一个"整图"辅助索引区域，返回的区域 Whole=true。
//
// 整图区域的颜色哈希在背景归一化后计算：局部视图查询常把不透明深色底叠加
// 在原图透明底（pHash 渲染为白底）上，导致整图亮度结构整体反相、颜色哈希近
// 互补（汉明距离 40+）。bgNormalize 把"边框主导暗底 + 内部含亮色图标"的
// 图像（典型如深底局部视图）归一化为白底后再哈希，使这类背景反相的局部视图
// 与原图整图哈希对齐、可召回且具备判别力（图标本身不变）。
//
// 触发条件严格：边框过半为暗（含透明）且内部存在足够亮像素（>150）才归一化。
// 这精确命中"暗底亮图标"的局部视图（如 TEST11），而不会误伤"暗图标"图像
// （如黑底深灰网格的 dashboard、深色线条图标 fact_check）——后者内部无亮像素，
// 不触发归一化。归一化仅作用于整图辅助区域（Whole=true）。
func MergeAll(src image.Image, regions []MergedRegion) MergedRegion {
	all := cloneMerged(regions)
	m := all[0]
	for _, r := range all[1:] {
		m = mergeTwoMerged(src, m, r)
	}
	m.Whole = true
	if crop := cropRect(src, m.BBox); crop != nil {
		m.Hash = phash.Hash(bgNormalize(crop))
	}
	return m
}

// bgNormalize 把"边框主导暗底 + 内部含亮色图标"的图像归一化为白底。
// 边框过半为暗（lum<80 或 alpha=0 透明）且内部亮像素（lum>150）占比 ≥5%
// 时，把全图暗像素（lum<80 或透明）替换为白；否则原样返回。
// 这样仅"暗底亮图标"被归一为白底（与原图透明底渲染一致），而"暗图标"
// 图像（内部无亮像素）不触发，避免抹除深色图标破坏其哈希。
func bgNormalize(src image.Image) image.Image {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	if w < 4 || h < 4 {
		return src
	}
	dark := 0
	total := 0
	border := func(x, y int) {
		_, _, _, a := src.At(b.Min.X+x, b.Min.Y+y).RGBA()
		total++
		if a == 0 {
			dark++
			return
		}
		r, g, bl, _ := src.At(b.Min.X+x, b.Min.Y+y).RGBA()
		lum := (299*float64(r) + 587*float64(g) + 114*float64(bl)) / 1000 / 257
		if lum < 80 {
			dark++
		}
	}
	for x := 0; x < w; x += 2 {
		border(x, 0)
		border(x, h-1)
	}
	for y := 0; y < h; y += 2 {
		border(0, y)
		border(w-1, y)
	}
	if total == 0 || dark*2 < total {
		return src // 边框非暗底主导
	}
	// 内部亮像素占比
	bright := 0
	interior := 0
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			r, g, bl, a := src.At(b.Min.X+x, b.Min.Y+y).RGBA()
			if a == 0 {
				continue
			}
			interior++
			lum := (299*float64(r) + 587*float64(g) + 114*float64(bl)) / 1000 / 257
			if lum > 150 {
				bright++
			}
		}
	}
	if interior == 0 || bright*20 < interior { // <5% 亮像素
		return src // 非暗底亮图标（内部无亮像素，是暗图标）
	}
	out := image.NewRGBA(b)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, bl, a := src.At(b.Min.X+x, b.Min.Y+y).RGBA()
			if a == 0 {
				out.Set(b.Min.X+x, b.Min.Y+y, color.RGBA{255, 255, 255, 255})
				continue
			}
			lum := (299*float64(r) + 587*float64(g) + 114*float64(bl)) / 1000 / 257
			if lum < 80 {
				out.Set(b.Min.X+x, b.Min.Y+y, color.RGBA{255, 255, 255, 255})
			} else {
				out.Set(b.Min.X+x, b.Min.Y+y, color.RGBA{uint8(r >> 8), uint8(g >> 8), uint8(bl >> 8), uint8(a >> 8)})
			}
		}
	}
	return out
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
