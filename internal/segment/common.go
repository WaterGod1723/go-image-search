// 共享工具：被多个算法文件（color / merge / gravity）复用的辅助函数。
package segment

import (
	"image"
	"image/color"
)

// cropRect 裁剪出矩形区域的内容。
func cropRect(src image.Image, b image.Rectangle) *image.RGBA {
	bb := b.Intersect(src.Bounds())
	if bb.Empty() {
		return nil
	}
	out := image.NewRGBA(bb)
	for y := bb.Min.Y; y < bb.Max.Y; y++ {
		for x := bb.Min.X; x < bb.Max.X; x++ {
			out.Set(x, y, src.At(x, y))
		}
	}
	return out
}

// structuralMask 生成颜色无关的结构掩码（灰度 → Otsu 二值，前景为黑）。
// 与 imageproc.StructuralMask 语义一致，供区域计算结构哈希使用，
// 避免 segment 引入对 imageproc 的依赖。
func structuralMask(src image.Image) *image.Gray {
	b := src.Bounds()
	g := image.NewGray(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			g.Set(x, y, color.GrayModel.Convert(src.At(x, y)))
		}
	}
	hist := make([]int, 256)
	for _, v := range g.Pix {
		hist[v]++
	}
	total := len(g.Pix)
	if total == 0 {
		return g
	}
	sum := 0
	for i, n := range hist {
		sum += i * n
	}
	sumB, wB := 0, 0
	bestThr, bestVar := uint8(0), float64(-1)
	for t := 0; t < 256; t++ {
		wB += hist[t]
		if wB == 0 {
			continue
		}
		wF := total - wB
		if wF == 0 {
			break
		}
		sumB += t * hist[t]
		mB := float64(sumB) / float64(wB)
		mF := float64(sum-sumB) / float64(wF)
		between := float64(wB) * float64(wF) * (mB - mF) * (mB - mF)
		if between >= bestVar {
			bestVar = between
			bestThr = uint8(t)
		}
	}
	out := image.NewGray(b)
	for i, v := range g.Pix {
		if v < bestThr {
			out.Pix[i] = 0
		} else {
			out.Pix[i] = 255
		}
	}
	return out
}

// minID 返回区域成员 ID 列表中的最小值，用于确定性排序。
func minID(ids []int) int {
	m := ids[0]
	for _, v := range ids[1:] {
		if v < m {
			m = v
		}
	}
	return m
}
