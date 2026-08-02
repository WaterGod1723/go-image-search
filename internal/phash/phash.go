// Package phash 实现基于 DCT 的感知哈希，用于衡量区域图像的相似度。
package phash

import (
	"image"
	"image/color"
	"math"
)

const (
	// Size 为 pHash 内部灰度图尺寸。
	Size = 32
	// Bits 为输出哈希位数。
	Bits = 64
)

// Hash 计算图像的 64-bit 感知哈希：
// 缩放到 32×32 → 灰度 → 二维 DCT → 取左上 8×8 低频系数 → 与中位数比较。
func Hash(img image.Image) uint64 {
	g := grayscaleResize(img, Size)
	gray := make([]float64, Size*Size)
	for i := range gray {
		gray[i] = float64(g.Pix[i])
	}

	coef := dct2D(gray, Size)

	// 低频 8×8 系数（跳过 DC 系数，减少对整体亮度的依赖）
	m := 8
	vals := make([]float64, 0, m*m)
	idx := 0
	for y := 0; y < m; y++ {
		for x := 0; x < m; x++ {
			if x == 0 && y == 0 {
				continue
			}
			vals = append(vals, coef[y*Size+x])
			idx++
		}
	}
	med := median(vals)

	var hash uint64
	for i, v := range vals {
		if v > med {
			hash |= 1 << uint(i)
		}
	}
	return hash
}

// Hamming 计算两个哈希的汉明距离（位不同的数量）。
func Hamming(a, b uint64) int {
	d := a ^ b
	cnt := 0
	for d != 0 {
		d &= d - 1
		cnt++
	}
	return cnt
}

// Distance01 返回归一化相似度 [0,1]，1 表示完全相同。
func Distance01(a, b uint64) float64 {
	h := Hamming(a, b)
	return 1 - float64(h)/Bits
}

func grayscaleResize(src image.Image, size int) *image.Gray {
	g := image.NewGray(image.Rect(0, 0, size, size))
	sb := src.Bounds()
	sw, sh := sb.Dx(), sb.Dy()
	if sw == 0 || sh == 0 {
		return g
	}
	// 双线性插值
	for y := 0; y < size; y++ {
		sy := float64(y) * float64(sh) / float64(size)
		y0 := int(sy)
		if y0 >= sh-1 {
			y0 = sh - 2
		}
		fy := sy - float64(y0)
		if y0 < 0 {
			y0 = 0
			fy = 0
		}
		for x := 0; x < size; x++ {
			sx := float64(x) * float64(sw) / float64(size)
			x0 := int(sx)
			if x0 >= sw-1 {
				x0 = sw - 2
			}
			fx := sx - float64(x0)
			if x0 < 0 {
				x0 = 0
				fx = 0
			}

			c00 := lumAt(src, sb, x0, y0)
			c10 := lumAt(src, sb, x0+1, y0)
			c01 := lumAt(src, sb, x0, y0+1)
			c11 := lumAt(src, sb, x0+1, y0+1)

			top := c00*(1-fx) + c10*fx
			bot := c01*(1-fx) + c11*fx
			v := top*(1-fy) + bot*fy
			g.SetGray(x, y, color.Gray{Y: uint8(v)})
		}
	}
	return g
}

func lumAt(img image.Image, b image.Rectangle, x, y int) float64 {
	r, g, b2, a := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
	if a == 0 {
		return 255 // 透明视为白色
	}
	lum := 0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b2)
	return lum / 65535 * 255
}

// dct2D 对 size×size 矩阵做二维 DCT-II，返回同尺寸系数矩阵。
func dct2D(in []float64, size int) []float64 {
	// 行变换
	row := make([]float64, size*size)
	for y := 0; y < size; y++ {
		dct1D(in[y*size:(y+1)*size], row[y*size:(y+1)*size], size)
	}
	out := make([]float64, size*size)
	// 列变换
	tmp := make([]float64, size)
	col := make([]float64, size)
	for x := 0; x < size; x++ {
		for y := 0; y < size; y++ {
			tmp[y] = row[y*size+x]
		}
		dct1D(tmp, col, size)
		for y := 0; y < size; y++ {
			out[y*size+x] = col[y]
		}
	}
	return out
}

func dct1D(in, out []float64, n int) {
	for k := 0; k < n; k++ {
		sum := 0.0
		for i := 0; i < n; i++ {
			sum += in[i] * math.Cos(math.Pi*float64(k)*(2*float64(i)+1)/(2*float64(n)))
		}
		ck := 1.0
		if k == 0 {
			ck = 1 / math.Sqrt2
		}
		out[k] = ck * sum * math.Sqrt(2/float64(n))
	}
}

func median(v []float64) float64 {
	cp := make([]float64, len(v))
	copy(cp, v)
	// 简单插入排序，长度小
	for i := 1; i < len(cp); i++ {
		for j := i; j > 0 && cp[j] < cp[j-1]; j-- {
			cp[j], cp[j-1] = cp[j-1], cp[j]
		}
	}
	return cp[len(cp)/2]
}
