package segment

import (
	"image"
	"image/color"
	"testing"
)

// mkGravityRegions 构造 n 个等大的竖向条带区域（用于引力聚合测试输入）。
// 第 i 个区域 bbox = [i*bw, (i+1)*bw) × [0, h)，面积 = bw*h。
func mkGravityRegions(n, bw, h int, c color.RGBA) []MergedRegion {
	out := make([]MergedRegion, n)
	for i := 0; i < n; i++ {
		bbox := image.Rect(i*bw, 0, (i+1)*bw, h)
		out[i] = MergedRegion{
			ID:        i + 1,
			Members:   []int{i + 1},
			Area:      bw * h,
			BBox:      bbox,
			MeanColor: c,
		}
	}
	return out
}

// findRegionByBBox 在列表中查找 bbox 完全匹配的区域，返回索引与是否找到。
func findRegionByBBox(rs []MergedRegion, b image.Rectangle) (int, bool) {
	for i, r := range rs {
		if r.BBox == b {
			return i, true
		}
	}
	return -1, false
}

func TestGravityDisabled(t *testing.T) {
	img := mkBands(60, 20, []color.RGBA{{R: 80, G: 80, B: 80, A: 255}})
	regs := mkGravityRegions(6, 10, 20, color.RGBA{80, 80, 80, 255})
	cfg := GravityConfig{Enabled: false, MinRegions: 5, TriggerCount: 5, Additive: false}

	// replace 模式：禁用时返回原集合副本
	out := GravityMerge(img, regs, cfg)
	if len(out) != len(regs) {
		t.Fatalf("禁用时 replace 模式应返回原集合，得到 %d", len(out))
	}
	for i := range out {
		if out[i].ID != regs[i].ID {
			t.Fatalf("禁用时应原样返回，ID 不符")
		}
	}

	// Additive 模式：禁用时返回 nil
	cfg.Additive = true
	if out2 := GravityMerge(img, regs, cfg); out2 != nil {
		t.Fatalf("禁用时 Additive 模式应返回 nil，得到 %d", len(out2))
	}
}

func TestGravityNoTrigger(t *testing.T) {
	img := mkBands(60, 20, []color.RGBA{{R: 80, G: 80, B: 80, A: 255}})
	// 5 个区域，TriggerCount=5：5 ≤ 5 不触发
	regs := mkGravityRegions(5, 10, 20, color.RGBA{80, 80, 80, 255})
	cfg := GravityConfig{Enabled: true, MinRegions: 5, TriggerCount: 5, Additive: false}

	out := GravityMerge(img, regs, cfg)
	if len(out) != 5 {
		t.Fatalf("未触发聚合应返回原 5 个区域，得到 %d", len(out))
	}
	cfg.Additive = true
	if out2 := GravityMerge(img, regs, cfg); out2 != nil {
		t.Fatalf("未触发聚合 Additive 应返回 nil，得到 %d", len(out2))
	}
}

// TestGravityMergesToOneBelowMin 验证触发后聚合到 MinRegions 停止。
func TestGravityMergesToMinRegions(t *testing.T) {
	img := mkBands(60, 20, []color.RGBA{{R: 80, G: 80, B: 80, A: 255}})
	// 6 个等大相邻条带，面积=200，质心间距=10
	// 时间步 = 10 * 400 = 4000（任意相邻对相同，取第一对融合）
	// 融合后剩 5 个 = MinRegions，停止
	regs := mkGravityRegions(6, 10, 20, color.RGBA{80, 80, 80, 255})
	cfg := GravityConfig{Enabled: true, MinRegions: 5, TriggerCount: 5, Additive: false}

	out := GravityMerge(img, regs, cfg)
	if len(out) != 5 {
		t.Fatalf("期望聚合到 5 个区域，得到 %d", len(out))
	}
	// 应有 1 个组合区域 bbox = (0,0,20,20)（前两条带之并），面积 400
	idx, ok := findRegionByBBox(out, image.Rect(0, 0, 20, 20))
	if !ok {
		t.Fatalf("期望存在 bbox=(0,0,20,20) 的组合区域，实际: %v", bboxes(out))
	}
	if out[idx].Area != 400 {
		t.Fatalf("组合区域面积应为 400，得到 %d", out[idx].Area)
	}
	if len(out[idx].Members) != 2 {
		t.Fatalf("组合区域应含 2 个成员，得到 %d", len(out[idx].Members))
	}
}

// TestGravityMultiStepMerging 验证多轮融合（8→5）且每轮选时间步最小的对。
func TestGravityMultiStepMerging(t *testing.T) {
	img := mkBands(80, 20, []color.RGBA{{R: 80, G: 80, B: 80, A: 255}})
	// 8 个等大相邻条带，每轮融合最近邻对（距离 10、面积和 200→逐步变化）
	// 8→7→6→5 共 3 轮，最终 5 个区域
	regs := mkGravityRegions(8, 10, 20, color.RGBA{80, 80, 80, 255})
	cfg := GravityConfig{Enabled: true, MinRegions: 5, TriggerCount: 5, Additive: false}

	out := GravityMerge(img, regs, cfg)
	if len(out) != 5 {
		t.Fatalf("期望 8 区域聚合到 5，得到 %d", len(out))
	}
	// 总面积守恒：8 * 200 = 1600
	total := 0
	for _, r := range out {
		total += r.Area
	}
	if total != 1600 {
		t.Fatalf("聚合后总面积应守恒为 1600，得到 %d", total)
	}
}

// TestGravitySmallNearFirst 验证"小且近的先融合"：
// 一个大区域远离一对小区域，小区域彼此相邻 → 小+小先融合。
func TestGravitySmallNearFirst(t *testing.T) {
	img := mkBands(100, 20, []color.RGBA{{R: 80, G: 80, B: 80, A: 255}})
	// 区域 A：小，bbox (0,0,10,20)，面积 200
	// 区域 B：小，bbox (10,0,20,20)，面积 200（A、B 相邻，距离 10）
	// 区域 C：大，bbox (80,0,100,20)，面积 400（远离 A、B）
	// 时间步(A,B) = 10 * 400 = 4000
	// 时间步(A,C) ≈ 75 * 600 = 45000；时间步(B,C) ≈ 65 * 600 = 39000
	// → A、B 先融合。MinRegions=2 不再继续（仅 3 区域时若 MinRegions=2 触发一次融合后停止）
	regs := []MergedRegion{
		{ID: 1, Members: []int{1}, Area: 200, BBox: image.Rect(0, 0, 10, 20), MeanColor: color.RGBA{80, 80, 80, 255}},
		{ID: 2, Members: []int{2}, Area: 200, BBox: image.Rect(10, 0, 20, 20), MeanColor: color.RGBA{80, 80, 80, 255}},
		{ID: 3, Members: []int{3}, Area: 400, BBox: image.Rect(80, 0, 100, 20), MeanColor: color.RGBA{80, 80, 80, 255}},
	}
	// 3 区域 > TriggerCount=2，触发；MinRegions=2，融合 1 次后停止
	cfg := GravityConfig{Enabled: true, MinRegions: 2, TriggerCount: 2, Additive: false}

	out := GravityMerge(img, regs, cfg)
	if len(out) != 2 {
		t.Fatalf("期望聚合到 2 个区域，得到 %d", len(out))
	}
	// A、B 应已融合为 bbox=(0,0,20,20) 面积=400
	idx, ok := findRegionByBBox(out, image.Rect(0, 0, 20, 20))
	if !ok {
		t.Fatalf("期望 A、B 先融合为 bbox=(0,0,20,20)，实际: %v", bboxes(out))
	}
	if out[idx].Area != 400 || len(out[idx].Members) != 2 {
		t.Fatalf("融合区域应有 2 成员面积 400，得到 成员=%d 面积=%d", len(out[idx].Members), out[idx].Area)
	}
}

// TestGravityAdditiveOnlyReturnsCombined 验证 Additive 模式仅返回 ≥2 成员的组合区域。
func TestGravityAdditiveOnlyReturnsCombined(t *testing.T) {
	img := mkBands(60, 20, []color.RGBA{{R: 80, G: 80, B: 80, A: 255}})
	regs := mkGravityRegions(6, 10, 20, color.RGBA{80, 80, 80, 255})
	cfg := GravityConfig{Enabled: true, MinRegions: 5, TriggerCount: 5, Additive: true}

	out := GravityMerge(img, regs, cfg)
	// 6→5 仅融合 1 次 → 1 个组合区域
	if len(out) != 1 {
		t.Fatalf("Additive 模式应仅返回 1 个组合区域，得到 %d", len(out))
	}
	if len(out[0].Members) < 2 {
		t.Fatalf("Additive 模式返回的区域应是组合区域（≥2 成员），成员=%d", len(out[0].Members))
	}
	if out[0].ID != 1 {
		t.Fatalf("Additive 模式应重新编号从 1 开始，得到 ID=%d", out[0].ID)
	}
}

// TestGravityBBoxEnclosesMembers 验证融合后 bbox 包住所有被融合区域。
func TestGravityBBoxEnclosesMembers(t *testing.T) {
	img := mkBands(60, 20, []color.RGBA{{R: 80, G: 80, B: 80, A: 255}})
	// 构造两个不相邻的小区域 + 一个中间区域，强制融合后 bbox 为两端之并
	regs := []MergedRegion{
		{ID: 1, Members: []int{1}, Area: 100, BBox: image.Rect(0, 0, 10, 10), MeanColor: color.RGBA{80, 80, 80, 255}},
		{ID: 2, Members: []int{2}, Area: 100, BBox: image.Rect(50, 10, 60, 20), MeanColor: color.RGBA{80, 80, 80, 255}},
		{ID: 3, Members: []int{3}, Area: 100, BBox: image.Rect(25, 5, 35, 15), MeanColor: color.RGBA{80, 80, 80, 255}},
	}
	// TriggerCount=2 触发，MinRegions=2 融合 1 次停止
	// 最近邻对：1-3 距离≈? 1(5,5) 3(30,10) dist≈√(625+25)=25.5；2-3 dist≈25.5；1-2 dist≈√(2500+100)=50.99
	// 时间步(1,3)=25.5*200=5100；(2,3)=25.5*200=5100；(1,2)=50.99*200≈10198
	// 1-3 与 2-3 相同，取第一对（1,3）。融合后 bbox=(0,0,35,15)
	cfg := GravityConfig{Enabled: true, MinRegions: 2, TriggerCount: 2, Additive: false}

	out := GravityMerge(img, regs, cfg)
	if len(out) != 2 {
		t.Fatalf("期望聚合到 2 个区域，得到 %d", len(out))
	}
	// 找出组合区域（成员≥2），其 bbox 应包住成员 bbox 之并
	var combined *MergedRegion
	for i := range out {
		if len(out[i].Members) >= 2 {
			combined = &out[i]
			break
		}
	}
	if combined == nil {
		t.Fatalf("未找到组合区域，实际: %v", bboxes(out))
	}
	// bbox 应为成员 bbox 之并，即包住被融合区域
	origUnion := image.Rect(0, 0, 0, 0)
	for _, r := range regs {
		if r.ID == combined.Members[0] || r.ID == combined.Members[1] {
			origUnion = origUnion.Union(r.BBox)
		}
	}
	if combined.BBox != origUnion {
		t.Fatalf("融合 bbox 应为成员之并 %v，得到 %v", origUnion, combined.BBox)
	}
}

// TestGravityMinRegionsClamp 验证 MinRegions < 1 被钳制为 1，不致无限循环。
func TestGravityMinRegionsClamp(t *testing.T) {
	img := mkBands(60, 20, []color.RGBA{{R: 80, G: 80, B: 80, A: 255}})
	regs := mkGravityRegions(6, 10, 20, color.RGBA{80, 80, 80, 255})
	cfg := GravityConfig{Enabled: true, MinRegions: 0, TriggerCount: 5, Additive: false}

	out := GravityMerge(img, regs, cfg)
	if len(out) != 1 {
		t.Fatalf("MinRegions=0 应钳制为 1，期望聚合到 1 个区域，得到 %d", len(out))
	}
	// 总面积守恒
	if out[0].Area != 1200 {
		t.Fatalf("聚合后总面积应为 1200，得到 %d", out[0].Area)
	}
	// bbox 应为全图
	if out[0].BBox != image.Rect(0, 0, 60, 20) {
		t.Fatalf("聚合到 1 个区域时 bbox 应为全图 (0,0,60,20)，得到 %v", out[0].BBox)
	}
}

// bboxes 返回区域列表的 bbox 字符串，用于错误信息。
func bboxes(rs []MergedRegion) []string {
	out := make([]string, len(rs))
	for i, r := range rs {
		out[i] = r.BBox.String()
	}
	return out
}
