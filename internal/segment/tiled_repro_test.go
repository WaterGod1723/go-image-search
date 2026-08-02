package segment

import (
	"fmt"
	"image"
	"image/color"
	"testing"

	"go-image-search/internal/phash"
)

// mkTianGrid 生成 4 个相同的空心"口"形（黑色描边），按田字排列。
func mkTianGrid(w, h int, gap int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	white := color.RGBA{255, 255, 255, 255}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, white)
		}
	}
	drawBoxRing := func(x0, y0, side, thick int) {
		black := color.RGBA{0, 0, 0, 255}
		for dy := 0; dy < side; dy++ {
			for dx := 0; dx < side; dx++ {
				on := dx < thick || dx >= side-thick || dy < thick || dy >= side-thick
				if on {
					img.Set(x0+dx, y0+dy, black)
				}
			}
		}
	}
	side := 20
	xs := []int{10, 10 + side + gap}
	ys := []int{10, 10 + side + gap}
	for _, y0 := range ys {
		for _, x0 := range xs {
			drawBoxRing(x0, y0, side, 3)
		}
	}
	return img
}

func TestTiledGridRepro(t *testing.T) {
	for _, med := range []int{0, 3} {
		img := mkTianGrid(100, 100, 6)
		cfg := DefaultConfig()
		cfg.MedianFilterK = med
		res, err := Segment(img, cfg)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Printf("=== median=%d 原始区域: %d 个 ===\n", med, len(res.Regions))
		infos := make([]RegionInfo, 0, len(res.Regions))
		for _, reg := range res.Regions {
			crop := res.Crop(img, reg.ID)
			infos = append(infos, RegionInfo{
				ID: reg.ID, Hash: phash.Hash(crop),
				Area: reg.Area, Color: reg.MeanColor, BBox: reg.BBox,
			})
			fmt.Printf("  #%d area=%d bbox=%v mean=%v\n", reg.ID, reg.Area, reg.BBox, reg.MeanColor)
		}
		merged := MergeSimilar(img, res, infos, DefaultMergeConfig())
		fmt.Printf("--- 合并后: %d 个 ---\n", len(merged))
		for _, m := range merged {
			fmt.Printf("  #%d members=%v bbox=%v\n", m.ID, m.Members, m.BBox)
		}
	}
}
