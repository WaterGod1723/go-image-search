package imageproc

import (
	"image"
	"image/color"
	"testing"
)

// mkCanvas 生成纯背景色 + 中央方块主体的测试图。
func mkCanvas(w, h int, bg, fg color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, bg)
		}
	}
	for y := h / 4; y < 3*h/4; y++ {
		for x := w / 4; x < 3*w/4; x++ {
			img.Set(x, y, fg)
		}
	}
	return img
}

func TestQueryVariantsIncludesOriginalAndSkeleton(t *testing.T) {
	src := mkCanvas(64, 64, color.RGBA{255, 255, 255, 255}, color.RGBA{10, 10, 10, 255})
	vs := QueryVariants(src)
	if len(vs) < 2 {
		t.Fatalf("应生成至少2张（原图+骨架）, got %d", len(vs))
	}
	if vs[0] != src {
		t.Fatal("第一张衍生图应为原图")
	}
	if _, ok := vs[1].(*image.Gray); !ok {
		t.Fatalf("第二张应为灰度骨架图, got %T", vs[1])
	}
	for i, v := range vs {
		if v == nil || v.Bounds().Empty() {
			t.Fatalf("衍生图 %d 为空", i)
		}
	}
}

func TestSkeletonThinsFilledSquare(t *testing.T) {
	src := mkCanvas(64, 64, color.RGBA{255, 255, 255, 255}, color.RGBA{0, 0, 0, 255})
	sk := Skeleton(src, DefaultSkeletonOptions())
	if sk == nil {
		t.Fatal("Skeleton 返回 nil")
	}
	if sk.Bounds().Dx() != 64 || sk.Bounds().Dy() != 64 {
		t.Fatalf("骨架尺寸应保持: %d x %d", sk.Bounds().Dx(), sk.Bounds().Dy())
	}
	// 填充方块应被细化，骨架像素数远少于原面积。
	height := float64(sk.Bounds().Dy())
	width := float64(sk.Bounds().Dx())
	fg := 0
	for y := 0; y < int(height); y++ {
		for x := 0; x < int(width); x++ {
			if sk.GrayAt(x, y).Y == 0 {
				fg++
			}
		}
	}
	// 原方块面积 = 32×32 = 1024
	if fg >= 900 {
		t.Fatalf("实心方块未被细化, 骨架像素 %d", fg)
	}
	// 骨架应非空（得有至少一个骨架像素）
	if fg == 0 {
		t.Fatal("细化后骨架不应为空")
	}
}

func TestOtsuBinomialSplit(t *testing.T) {
	// 双峰直方图：一半 0、一半 255，Otsu 阈值应落在中间。
	g := image.NewGray(image.Rect(0, 0, 8, 8))
	for i := 0; i < len(g.Pix); i++ {
		if i < len(g.Pix)/2 {
			g.Pix[i] = 10
		} else {
			g.Pix[i] = 200
		}
	}
	thr := otsuChoose(g.Pix)
	if thr < 10 || thr > 200 {
		t.Fatalf("Otsu 阈值异常: %d", thr)
	}
	// 二值化后应把 10 归为前景、200 归为背景。
	bin := otsuBinarize(g, DefaultSkeletonOptions())
	if bin == nil {
		t.Fatal("otsuBinarize 返回 nil")
	}
	fg := 0
	for _, v := range bin.Pix {
		if v == 0 {
			fg++
		}
	}
	if fg != len(g.Pix)/2 {
		t.Fatalf("二值化前景数应为一半, got %d/%d", fg, len(g.Pix))
	}
}

func TestZhangSuenPreservesLineThickness(t *testing.T) {
	// 一根 1px 水平线不应被删除（连通性保持）。
	fg := make([][]bool, 5)
	for x := range fg {
		fg[x] = make([]bool, 5)
	}
	const mid = 2
	for x := 0; x < 5; x++ {
		fg[x][mid] = true
	}
	zhangSuen(fg, 5, 5)
	// 线两端各保留至少 1 个像素，中点应当保留
	if !fg[2][mid] {
		t.Fatal("单像素线中点不应被删除")
	}
	cnt := 0
	for x := 0; x < 5; x++ {
		if fg[x][mid] {
			cnt++
		}
	}
	if cnt == 0 {
		t.Fatal("线被完全删除")
	}
	if cnt < 3 {
		t.Fatalf("单像素线不应过度缩短, 保留 %d", cnt)
	}
}