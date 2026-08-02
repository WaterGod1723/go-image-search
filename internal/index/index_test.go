package index

import (
	"image"
	"image/color"
	"os"
	"path/filepath"
	"testing"

	"go-image-search/internal/phash"
)

func mkRegionHash(regionID int, area int, bbox image.Rectangle, fill color.RGBA) RegionHash {
	img := image.NewRGBA(bbox)
	for y := bbox.Min.Y; y < bbox.Max.Y; y++ {
		for x := bbox.Min.X; x < bbox.Max.X; x++ {
			img.Set(x, y, fill)
		}
	}
	return RegionHash{RegionID: regionID, Hash: phash.Hash(img), Area: area, BBox: bbox}
}

func TestIndexSearch(t *testing.T) {
	ix := New()
	ix.AddImage("a.png", []RegionHash{
		mkRegionHash(1, 100, image.Rect(0, 0, 10, 10), color.RGBA{255, 0, 0, 255}),
		mkRegionHash(2, 300, image.Rect(10, 10, 30, 30), color.RGBA{0, 0, 255, 255}),
	})
	ix.AddImage("b.png", []RegionHash{
		mkRegionHash(1, 100, image.Rect(0, 0, 10, 10), color.RGBA{0, 255, 0, 255}),
		mkRegionHash(2, 300, image.Rect(10, 10, 30, 30), color.RGBA{255, 255, 0, 255}),
	})

	// 查询与 a 相同的红色块 + 蓝色块
	query := []QueryRegion{
		{Hash: mkRegionHash(9, 100, image.Rect(0, 0, 10, 10), color.RGBA{255, 0, 0, 255}).Hash, Area: 100},
		{Hash: mkRegionHash(9, 300, image.Rect(10, 10, 30, 30), color.RGBA{0, 0, 255, 255}).Hash, Area: 300},
	}
	matches := ix.Search(query, SearchOptions{MaxDist: 12})
	if len(matches) == 0 {
		t.Fatal("无匹配结果")
	}
	if matches[0].ImageID != "a.png" {
		t.Fatalf("期望匹配 a.png，得到 %s", matches[0].ImageID)
	}
}

func TestSaveLoadRoundtrip(t *testing.T) {
	ix := New()
	ix.AddImage("a.png", []RegionHash{
		mkRegionHash(1, 100, image.Rect(0, 0, 10, 10), color.RGBA{255, 0, 0, 255}),
	})
	path := filepath.Join(t.TempDir(), "idx.bin")
	if err := ix.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Len() != 1 {
		t.Fatalf("加载后条目数错误: %d", loaded.Len())
	}
	if !loaded.Images["a.png"] {
		t.Fatal("加载后丢失图像记录")
	}
	matches := loaded.Search([]QueryRegion{{Hash: ix.Entries[0].Hash, Area: 100}}, SearchOptions{})
	if len(matches) == 0 || matches[0].ImageID != "a.png" {
		t.Fatal("加载后检索失败")
	}
}

func TestVariants(t *testing.T) {
	v := variants(0x0000)
	// 1 + 16 + 120 = 137
	if len(v) != 137 {
		t.Fatalf("变体数量错误: %d", len(v))
	}
	seen := map[uint16]bool{}
	for _, x := range v {
		seen[x] = true
	}
	if !seen[0x0001] || !seen[0x8000] || !seen[0x0003] {
		t.Fatal("变体缺少预期值")
	}
}

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}
