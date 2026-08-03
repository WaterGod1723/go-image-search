package segment

import (
	"image"
	"image/color"
	"reflect"
	"testing"

	"go-image-search/internal/phash"
)

// referenceProcessRegions 复刻历史 processRegions 的实现，作为默认流水线的等价性基准。
func referenceProcessRegions(img image.Image, cfg Config, mergeCfg MergeConfig, grav GravityConfig) (*Result, []MergedRegion, []MergedRegion, error) {
	res, err := Segment(img, cfg)
	if err != nil {
		return nil, nil, nil, err
	}
	infos := make([]RegionInfo, 0, len(res.Regions))
	for _, reg := range res.Regions {
		crop := res.Crop(img, reg.ID)
		if crop == nil {
			continue
		}
		shape := phash.Hash(structuralMask(crop))
		infos = append(infos, RegionInfo{
			ID: reg.ID, Hash: phash.Hash(crop), Shape: shape,
			Area: reg.Area, Color: reg.MeanColor, BBox: reg.BBox,
		})
	}
	merged := MergeSimilar(img, res, infos, mergeCfg)
	partition := merged
	var aux []MergedRegion
	if grav.Enabled {
		gravity := GravityMerge(img, merged, grav)
		if grav.Additive {
			aux = gravity
		} else {
			partition = gravity
		}
	}
	if grav.FrameRatio > 0 {
		partition = FilterFullFrame(partition, img.Bounds().Dx(), img.Bounds().Dy(), grav.FrameRatio)
	}
	var whole *MergedRegion
	if ShouldAddWholeAux(grav, partition, aux) {
		w := MergeAll(img, partition)
		whole = &w
	}
	aux = MergeContainedOverlapping(img, aux)
	if grav.Enabled && grav.CombineFew > 0 && len(partition) >= 2 {
		for i := range partition {
			p := partition[i]
			p.Whole = false
			aux = append(aux, p)
		}
	}
	if whole != nil {
		aux = append(aux, *whole)
	}
	id := 0
	for i := range partition {
		id++
		partition[i].ID = id
	}
	for i := range aux {
		id++
		aux[i].ID = id
	}
	return res, partition, aux, nil
}

// mergedEqual 比较两个 MergedRegion 的全部字段（Members 顺序敏感）。
func mergedEqual(a, b MergedRegion) bool {
	return a.ID == b.ID && a.Whole == b.Whole && a.Area == b.Area &&
		a.Hash == b.Hash && a.Shape == b.Shape && a.BBox == b.BBox &&
		a.MeanColor == b.MeanColor && reflect.DeepEqual(a.Members, b.Members)
}

// mergedSlicesEqual 比较两个区域切片。
func mergedSlicesEqual(a, b []MergedRegion) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !mergedEqual(a[i], b[i]) {
			return false
		}
	}
	return true
}

// TestRunDefaultEquivalent 验证默认流水线与历史 processRegions 行为完全一致。
func TestRunDefaultEquivalent(t *testing.T) {
	images := []image.Image{
		mkSegImage(64, 64),
		mkBands(120, 30, []color.RGBA{
			{R: 60, G: 60, B: 60, A: 255},
			{R: 200, G: 200, B: 200, A: 255},
			{R: 40, G: 120, B: 220, A: 255},
		}),
		mkTianGrid(100, 100, 6),
	}
	gravs := []GravityConfig{
		DefaultGravityConfig(),
		{Enabled: false, FrameRatio: 0.9},
		{Enabled: true, MinRegions: 5, TriggerCount: 5, Additive: false, CombineFew: 5, FrameRatio: 0.9},
		{Enabled: true, MinRegions: 3, TriggerCount: 3, Additive: true, CombineFew: 0, FrameRatio: 0},
	}
	for gi, g := range gravs {
		for ii, img := range images {
			wantRes, wantPart, wantAux, err := referenceProcessRegions(img, DefaultConfig(), DefaultMergeConfig(), g)
			if err != nil {
				t.Fatalf("[grav%d/img%d] reference 失败: %v", gi, ii, err)
			}
			res, out, err := RunDefault(img, DefaultConfig(), DefaultMergeConfig(), g)
			if err != nil {
				t.Fatalf("[grav%d/img%d] RunDefault 失败: %v", gi, ii, err)
			}
			if res.Width != wantRes.Width || res.Height != wantRes.Height {
				t.Errorf("[grav%d/img%d] 尺寸不符: got %dx%d want %dx%d", gi, ii, res.Width, res.Height, wantRes.Width, wantRes.Height)
			}
			if !mergedSlicesEqual(out.Partition, wantPart) {
				t.Errorf("[grav%d/img%d] Partition 不一致:\n got  %v\n want %v", gi, ii, bboxes(out.Partition), bboxes(wantPart))
			}
			if !mergedSlicesEqual(out.Aux, wantAux) {
				t.Errorf("[grav%d/img%d] Aux 不一致:\n got  %v\n want %v", gi, ii, bboxes(out.Aux), bboxes(wantAux))
			}
			// 编号全局唯一：partition 1..P，aux P+1..P+A
			seen := map[int]bool{}
			for _, r := range out.All() {
				if seen[r.ID] {
					t.Errorf("[grav%d/img%d] 区域 ID 重复: %d", gi, ii, r.ID)
				}
				seen[r.ID] = true
			}
		}
	}
}

// TestRunDefaultIDsSequential 验证默认流水线统一编号规则。
func TestRunDefaultIDsSequential(t *testing.T) {
	img := mkSegImage(64, 64)
	_, out, err := RunDefault(img, DefaultConfig(), DefaultMergeConfig(), DefaultGravityConfig())
	if err != nil {
		t.Fatal(err)
	}
	all := out.All()
	if len(all) == 0 {
		t.Fatal("空产出")
	}
	for i, r := range all {
		if r.ID != i+1 {
			t.Fatalf("编号应连续 1..N，第 %d 个区域 ID=%d", i, r.ID)
		}
	}
}

// TestRunCustomGridPartition 用自定义"网格划分"策略替换像素颜色划分策略，
// 验证策略可插拔、可独立于默认流水线运行。
func TestRunCustomGridPartition(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	grid := named("grid", func(ctx *Context, in Outcome) (Outcome, error) {
		w, h := 64, 64
		cellW, cellH := w/2, h/2
		var partition []MergedRegion
		id := 0
		for gy := 0; gy < 2; gy++ {
			for gx := 0; gx < 2; gx++ {
				id++
				bbox := image.Rect(gx*cellW, gy*cellH, (gx+1)*cellW, (gy+1)*cellH)
				partition = append(partition, MergedRegion{
					ID: id, Members: []int{id}, Area: cellW * cellH, BBox: bbox,
				})
			}
		}
		ctx.Res = &Result{Width: w, Height: h}
		out := in
		out.Partition = partition
		return out, nil
	})

	res, out, err := Run(img, grid, Relabel())
	if err != nil {
		t.Fatal(err)
	}
	if res.Width != 64 || res.Height != 64 {
		t.Fatalf("自定义策略未填充 ctx.Res: %+v", res)
	}
	if len(out.Partition) != 4 {
		t.Fatalf("期望 4 个网格区域，得到 %d", len(out.Partition))
	}
	for i, r := range out.Partition {
		if r.ID != i+1 {
			t.Fatalf("Relabel 后 ID 应为 1..4，得到 %d", r.ID)
		}
	}
}

// TestRunCombineStrategies 验证自定义策略可与内置策略混合组合：
// 颜色划分 → 自定义"只保留前 2 个区域"策略 → 相似合并 → 统一编号。
func TestRunCombineStrategies(t *testing.T) {
	img := mkSegImage(64, 64)
	keepTwo := StrategyFunc(func(ctx *Context, in Outcome) (Outcome, error) {
		out := in
		if len(out.Partition) > 2 {
			out.Partition = out.Partition[:2]
		}
		return out, nil
	})
	res, out, err := Run(img, ColorSegment(DefaultConfig()), keepTwo, Merge(DefaultMergeConfig()), Relabel())
	if err != nil {
		t.Fatal(err)
	}
	if res == nil || len(res.Regions) == 0 {
		t.Fatal("像素级结果缺失")
	}
	if len(out.Partition) == 0 {
		t.Fatal("划分区域为空")
	}
	for i, r := range out.Partition {
		if r.ID != i+1 {
			t.Fatalf("编号应连续，得到 %d", r.ID)
		}
	}
}

// TestPipelineSkipsDisabled 验证 Gravity 策略在 Enabled=false 时保持原有划分不变。
func TestPipelineGravityDisabled(t *testing.T) {
	img := mkSegImage(64, 64)
	cfg := DefaultConfig()
	g0 := GravityConfig{Enabled: false}
	_, out, err := Run(img, ColorSegment(cfg), Merge(DefaultMergeConfig()), Gravity(g0))
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Aux) != 0 {
		t.Fatalf("引力禁用时不应有辅助区域，得到 %d", len(out.Aux))
	}
	if len(out.Partition) == 0 {
		t.Fatal("划分区域不应为空")
	}
}

// TestOutcomeAll 验证 All() 的拼接顺序。
func TestOutcomeAll(t *testing.T) {
	out := Outcome{
		Partition: []MergedRegion{{ID: 1}, {ID: 2}},
		Aux:       []MergedRegion{{ID: 3}},
	}
	all := out.All()
	if len(all) != 3 {
		t.Fatalf("All() 长度应为 3，得到 %d", len(all))
	}
	if all[0].ID != 1 || all[1].ID != 2 || all[2].ID != 3 {
		t.Fatalf("All() 应按 Partition → Aux 顺序: %+v", all)
	}
}
