package phash

import (
	"image"
	"image/color"
	"testing"
)

func mkIcon(w, h int, bg, fg color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, bg)
		}
	}
	// 中心画个方块
	for y := h / 4; y < 3*h/4; y++ {
		for x := w / 4; x < 3*w/4; x++ {
			img.Set(x, y, fg)
		}
	}
	return img
}

func TestHashSimilar(t *testing.T) {
	base := mkIcon(64, 64, color.RGBA{255, 255, 255, 255}, color.RGBA{10, 120, 200, 255})

	// 缩放 + 轻微亮度变化
	resized := mkIcon(96, 96, color.RGBA{255, 255, 255, 255}, color.RGBA{20, 110, 190, 255})

	h1 := Hash(base)
	h2 := Hash(resized)
	if d := Hamming(h1, h2); d > 10 {
		t.Fatalf("相似图像哈希距离过大: %d", d)
	}
}

func TestHashDifferent(t *testing.T) {
	a := mkIcon(64, 64, color.RGBA{255, 255, 255, 255}, color.RGBA{10, 120, 200, 255})
	b := mkIcon(64, 64, color.RGBA{0, 0, 0, 255}, color.RGBA{255, 80, 10, 255})
	if d := Hamming(Hash(a), Hash(b)); d < 20 {
		t.Fatalf("不同图像哈希距离过小: %d", d)
	}
}

func TestHamming(t *testing.T) {
	if Hamming(0, 0) != 0 {
		t.Fatal("Hamming(0,0) != 0")
	}
	if Hamming(0xff, 0x00) != 8 {
		t.Fatal("Hamming(0xff,0x00) != 8")
	}
}
