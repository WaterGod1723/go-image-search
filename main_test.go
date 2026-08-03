package main

import (
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"go-image-search/internal/index"
	"go-image-search/internal/segment"
)

type iconSpec struct {
	name string
	bg   color.RGBA
	fg   color.RGBA
	// shape: 0 方块, 1 圆形(近似), 2 三角形(近似)
	shape int
}

func writeIcon(path string, spec iconSpec, size int) {
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			img.Set(x, y, spec.bg)
		}
	}
	cx, cy := size/2, size/2
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			var inside bool
			switch spec.shape {
			case 0:
				inside = x > size/4 && x < 3*size/4 && y > size/4 && y < 3*size/4
			case 1:
				dx, dy := float64(x-cx), float64(y-cy)
				inside = dx*dx+dy*dy < float64(size/4)*float64(size/4)
			case 2:
				inside = x >= size/4 && x <= 3*size/4 && y >= size/4 &&
					y <= 3*size/4 && y <= (2*(x-size/4)+size/2)
			}
			if inside {
				img.Set(x, y, spec.fg)
			}
		}
	}
	f, _ := os.Create(path)
	defer f.Close()
	png.Encode(f, img)
}

// 生成一个带轻微变化的同款图标，模拟“同一图标的不同版本”。
func writeVariant(path string, spec iconSpec, size int, delta uint8) {
	s := spec
	s.fg = color.RGBA{s.fg.R + delta, s.fg.G - delta/2, s.fg.B, 255}
	writeIcon(path, s, size)
}

func TestEndToEndSearch(t *testing.T) {
	dir := t.TempDir()
	lib := filepath.Join(dir, "lib")
	qdir := filepath.Join(dir, "q")
	os.MkdirAll(lib, 0o755)
	os.MkdirAll(qdir, 0o755)

	icons := []iconSpec{
		{name: "red_square.png", bg: color.RGBA{255, 255, 255, 255}, fg: color.RGBA{220, 40, 40, 255}, shape: 0},
		{name: "blue_circle.png", bg: color.RGBA{240, 240, 240, 255}, fg: color.RGBA{40, 40, 220, 255}, shape: 1},
		{name: "green_tri.png", bg: color.RGBA{255, 255, 255, 255}, fg: color.RGBA{40, 200, 60, 255}, shape: 2},
		{name: "yellow_square.png", bg: color.RGBA{255, 255, 255, 255}, fg: color.RGBA{230, 200, 30, 255}, shape: 0},
	}
	for _, ic := range icons {
		writeIcon(filepath.Join(lib, ic.name), ic, 64)
	}

	// 查询：红色方块的缩放+亮度微调版本
	writeVariant(filepath.Join(qdir, "query.png"), icons[0], 80, 12)

	// 建索引
	ix := index.New()
	cfg := segment.DefaultConfig()
	files, err := filepath.Glob(filepath.Join(lib, "*.png"))
	if err != nil || len(files) == 0 {
		t.Fatal("无索引图片")
	}
	for _, f := range files {
		img, err := loadTestImage(f)
		if err != nil {
			t.Fatal(err)
		}
		hashes, err := hashImage(img, cfg, segment.DefaultMergeConfig(), segment.GravityConfig{Enabled: false})
		if err != nil {
			t.Fatal(err)
		}
		ix.AddImage(f, hashes)
	}

	// 查询
	qimg, err := loadTestImage(filepath.Join(qdir, "query.png"))
	if err != nil {
		t.Fatal(err)
	}
	qhashes, err := hashImage(qimg, cfg, segment.DefaultMergeConfig(), segment.GravityConfig{Enabled: false})
	if err != nil {
		t.Fatal(err)
	}
	var query []index.QueryRegion
	for _, h := range qhashes {
		query = append(query, index.QueryRegion{
			Hash: h.Hash, Area: h.Area, Color: h.Color,
			NX: h.NX, NY: h.NY, Fill: h.Fill, Aspect: h.Aspect,
		})
	}
	matches := ix.Search(query, index.SearchOptions{MaxDist: 12, ColorWeight: 0.8})
	if len(matches) == 0 {
		t.Fatal("无匹配结果")
	}
	got := filepath.Base(matches[0].ImageID)
	if got != "red_square.png" {
		t.Fatalf("期望 top1=red_square.png，得到 %s (score=%.3f)", got, matches[0].Score)
	}
	// 形状相同仅颜色不同的图标应排在形状不同的图标之前
	rankYellow, rankBlue := -1, -1
	for i, m := range matches {
		switch filepath.Base(m.ImageID) {
		case "yellow_square.png":
			rankYellow = i
		case "blue_circle.png":
			rankBlue = i
		}
	}
	if rankYellow < 0 || rankBlue < 0 {
		t.Fatalf("缺少期望候选: yellow=%d blue=%d", rankYellow, rankBlue)
	}
	if rankYellow > rankBlue {
		t.Fatalf("同形不同色应排在异形之前: yellow=%d blue=%d", rankYellow, rankBlue)
	}
	// top1 与明显不同图标（异形）得分差距应显著
	if matches[0].Score-matches[rankBlue].Score < 0.2 {
		t.Fatalf("区分度不足: top1=%.3f blue=%s=%.3f", matches[0].Score, matches[rankBlue].ImageID, matches[rankBlue].Score)
	}
}

// TestProcessRegionsCombineFew 验证最终区域数过少时追加"整图"辅助区域：
// CombineFew 生效时 aux 包含一个 bbox 覆盖全图的组合区域；
// CombineFew=0 禁用时 aux 不含整图区域。
func TestProcessRegionsCombineFew(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			img.Set(x, y, color.RGBA{250, 250, 250, 255})
		}
	}
	fillRect := func(r image.Rectangle, c color.RGBA) {
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				img.Set(x, y, c)
			}
		}
	}
	fillRect(image.Rect(10, 10, 30, 30), color.RGBA{200, 30, 30, 255})
	fillRect(image.Rect(60, 60, 90, 90), color.RGBA{30, 30, 200, 255})

	cfg := segment.DefaultConfig()
	mc := segment.DefaultMergeConfig()

	grav := segment.DefaultGravityConfig()
	_, out, err := segment.RunDefault(img, cfg, mc, grav)
	if err != nil {
		t.Fatalf("segment.RunDefault 失败: %v", err)
	}
	partition := out.Partition
	all := out.All()
	// 白色背景区域 bbox 与全图重合，应被边框过滤丢弃
	for _, p := range partition {
		if p.BBox == image.Rect(0, 0, 100, 100) {
			t.Fatalf("全图背景区域应被边框过滤丢弃: %+v", p)
		}
	}
	hasWhole := false
	for _, a := range all {
		if a.Whole && len(a.Members) > 1 {
			hasWhole = true
		}
	}
	if !hasWhole {
		t.Fatalf("CombineFew 生效时应有整图辅助区域，得到 %v", all)
	}

	grav2 := grav
	grav2.CombineFew = 0
	_, out2, err := segment.RunDefault(img, cfg, mc, grav2)
	if err != nil {
		t.Fatalf("segment.RunDefault(禁用) 失败: %v", err)
	}
	for _, a := range out2.All() {
		if a.Whole {
			t.Fatalf("CombineFew=0 时不应有整图辅助区域: %+v", a)
		}
	}
}

func loadTestImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}
