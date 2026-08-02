package imageproc

import (
	"image"
	"image/color"
)

// Grayscale 将任意图片转为灰度图，返回 *image.Gray。
func Grayscale(src image.Image) *image.Gray {
	b := src.Bounds()
	g := image.NewGray(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			g.Set(x, y, color.GrayModel.Convert(src.At(x, y)))
		}
	}
	return g
}

// MedianFilter 对灰度图做 k×k 中值滤波（k 为奇数），去孤立噪声点。
// 边界处取不完整窗口内已有像素的中值。
func MedianFilter(src *image.Gray, k int) *image.Gray {
	if k < 1 {
		return cloneGray(src)
	}
	k |= 1 // 保证奇数
	half := k / 2
	b := src.Bounds()
	dst := cloneGray(src)
	tmp := make([]uint8, k*k)

	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			n := 0
			for dy := -half; dy <= half; dy++ {
				for dx := -half; dx <= half; dx++ {
					cx, cy := x+dx, y+dy
					if cx < b.Min.X || cx >= b.Max.X || cy < b.Min.Y || cy >= b.Max.Y {
						continue
					}
					tmp[n] = src.GrayAt(cx, cy).Y
					n++
				}
			}
			dst.SetGray(x, y, color.Gray{Y: medianU8(tmp[:n])})
		}
	}
	return dst
}

func medianU8(v []uint8) uint8 {
	// 插入排序，n 很小（≤ k²）
	for i := 1; i < len(v); i++ {
		for j := i; j > 0 && v[j] < v[j-1]; j-- {
			v[j], v[j-1] = v[j-1], v[j]
		}
	}
	return v[len(v)/2]
}

func cloneGray(src *image.Gray) *image.Gray {
	g := image.NewGray(src.Bounds())
	copy(g.Pix, src.Pix)
	return g
}
