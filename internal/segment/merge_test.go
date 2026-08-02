package segment

import (
	"image"
	"image/color"
	"testing"
)

// mkBands 生成竖向色带图像（作为合并/裁剪的源图）。
func mkBands(w, h int, colors []color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	band := w / len(colors)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := colors[x/band]
			img.Set(x, y, c)
		}
	}
	return img
}

// mkResult 按竖向色带标签构造 Result（bands 为各带的区域 ID，0 表示背景）。
func mkResult(w, h int, bands []int32) *Result {
	labels := make([]int32, w*h)
	bw := w / len(bands)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			labels[y*w+x] = bands[x/bw]
		}
	}
	area := map[int32]int{}
	minX := map[int32]int{}
	minY := map[int32]int{}
	maxX := map[int32]int{}
	maxY := map[int32]int{}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			l := labels[y*w+x]
			if l <= 0 {
				continue
			}
			area[l]++
			if _, ok := minX[l]; !ok {
				minX[l], minY[l], maxX[l], maxY[l] = x, y, x, y
			} else {
				if x < minX[l] {
					minX[l] = x
				}
				if x > maxX[l] {
					maxX[l] = x
				}
				if y < minY[l] {
					minY[l] = y
				}
				if y > maxY[l] {
					maxY[l] = y
				}
			}
		}
	}
	var regions []*Region
	ids := make([]int32, 0, len(area))
	for l := range area {
		ids = append(ids, l)
	}
	sortInt32(ids)
	for _, l := range ids {
		regions = append(regions, &Region{
			ID:   int(l),
			BBox: image.Rect(minX[l], minY[l], maxX[l]+1, maxY[l]+1),
			Area: area[l],
		})
	}
	return &Result{Width: w, Height: h, Labels: labels, Regions: regions}
}

func sortInt32(v []int32) {
	for i := 1; i < len(v); i++ {
		for j := i; j > 0 && v[j] < v[j-1]; j-- {
			v[j], v[j-1] = v[j-1], v[j]
		}
	}
}

func TestMergeAdjacentSimilar(t *testing.T) {
	img := mkBands(40, 20, []color.RGBA{
		{R: 60, G: 60, B: 60, A: 255},
		{R: 90, G: 90, B: 90, A: 255},
	})
	res := mkResult(40, 20, []int32{1, 2})
	infos := []RegionInfo{
		{ID: 1, Hash: 0x123456789abcdef0, Area: 400, Color: color.RGBA{60, 60, 60, 255}, BBox: image.Rect(0, 0, 20, 20)},
		{ID: 2, Hash: 0x123456789abcdef0, Area: 400, Color: color.RGBA{90, 90, 90, 255}, BBox: image.Rect(20, 0, 40, 20)},
	}
	merged := MergeSimilar(img, res, infos, DefaultMergeConfig())
	if len(merged) != 1 {
		t.Fatalf("期望合并为 1 个组合区域，得到 %d", len(merged))
	}
	m := merged[0]
	if len(m.Members) != 2 {
		t.Fatalf("组合区域应含 2 个成员，得到 %d", len(m.Members))
	}
	if m.Area != 800 {
		t.Fatalf("组合区域面积错误: %d", m.Area)
	}
	if m.BBox != image.Rect(0, 0, 40, 20) {
		t.Fatalf("组合区域 bbox 错误: %v", m.BBox)
	}
}

func TestMergeAdjacentDissimilar(t *testing.T) {
	img := mkBands(40, 20, []color.RGBA{
		{R: 60, G: 60, B: 60, A: 255},
		{R: 200, G: 200, B: 200, A: 255},
	})
	res := mkResult(40, 20, []int32{1, 2})
	infos := []RegionInfo{
		{ID: 1, Hash: 0x123456789abcdef0, Area: 400, Color: color.RGBA{60, 60, 60, 255}, BBox: image.Rect(0, 0, 20, 20)},
		{ID: 2, Hash: 0x123456789abcdef0, Area: 400, Color: color.RGBA{200, 200, 200, 255}, BBox: image.Rect(20, 0, 40, 20)},
	}
	merged := MergeSimilar(img, res, infos, DefaultMergeConfig())
	if len(merged) != 2 {
		t.Fatalf("差异大的相邻区域不应合并: 期望 2 个，得到 %d", len(merged))
	}
}

func TestMergeNonAdjacent(t *testing.T) {
	img := mkBands(60, 20, []color.RGBA{
		{R: 60, G: 60, B: 60, A: 255},
		{R: 200, G: 200, B: 200, A: 255},
		{R: 60, G: 60, B: 60, A: 255},
	})
	res := mkResult(60, 20, []int32{1, 3, 2})
	infos := []RegionInfo{
		{ID: 1, Hash: 0x123456789abcdef0, Area: 400, Color: color.RGBA{60, 60, 60, 255}, BBox: image.Rect(0, 0, 20, 20)},
		{ID: 2, Hash: 0x123456789abcdef0, Area: 400, Color: color.RGBA{60, 60, 60, 255}, BBox: image.Rect(40, 0, 60, 20)},
		{ID: 3, Hash: 0x0, Area: 400, Color: color.RGBA{200, 200, 200, 255}, BBox: image.Rect(20, 0, 40, 20)},
	}
	merged := MergeSimilar(img, res, infos, DefaultMergeConfig())
	if len(merged) != 3 {
		t.Fatalf("不相邻的相似区域不应合并: 期望 3 个，得到 %d", len(merged))
	}
}

func TestMergeTransitive(t *testing.T) {
	img := mkBands(60, 20, []color.RGBA{
		{R: 60, G: 60, B: 60, A: 255},
		{R: 80, G: 80, B: 80, A: 255},
		{R: 70, G: 70, B: 70, A: 255},
	})
	res := mkResult(60, 20, []int32{1, 2, 3})
	infos := []RegionInfo{
		{ID: 1, Hash: 0x01, Area: 400, Color: color.RGBA{60, 60, 60, 255}, BBox: image.Rect(0, 0, 20, 20)},
		{ID: 2, Hash: 0x03, Area: 400, Color: color.RGBA{80, 80, 80, 255}, BBox: image.Rect(20, 0, 40, 20)},
		{ID: 3, Hash: 0x02, Area: 400, Color: color.RGBA{70, 70, 70, 255}, BBox: image.Rect(40, 0, 60, 20)},
	}
	merged := MergeSimilar(img, res, infos, DefaultMergeConfig())
	if len(merged) != 1 {
		t.Fatalf("期望传递合并为 1 个组合区域，得到 %d", len(merged))
	}
	if len(merged[0].Members) != 3 {
		t.Fatalf("组合区域应含 3 个成员，得到 %d", len(merged[0].Members))
	}
}

func TestMergeDisabled(t *testing.T) {
	img := mkBands(40, 20, []color.RGBA{
		{R: 60, G: 60, B: 60, A: 255},
		{R: 90, G: 90, B: 90, A: 255},
	})
	res := mkResult(40, 20, []int32{1, 2})
	infos := []RegionInfo{
		{ID: 1, Hash: 0x01, Area: 400, Color: color.RGBA{60, 60, 60, 255}, BBox: image.Rect(0, 0, 20, 20)},
		{ID: 2, Hash: 0x01, Area: 400, Color: color.RGBA{90, 90, 90, 255}, BBox: image.Rect(20, 0, 40, 20)},
	}
	cfg := DefaultMergeConfig()
	cfg.Enabled = false
	merged := MergeSimilar(img, res, infos, cfg)
	if len(merged) != 2 {
		t.Fatalf("关闭合并时应原样返回单区域: 期望 2 个，得到 %d", len(merged))
	}
}
