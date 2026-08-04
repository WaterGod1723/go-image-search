package sczl

import (
	"image"
	"image/color"
	"math"
)

// toRGBA 把任意图像转为 RGBA 像素缓冲（按 b 平铺，左上对齐到 0,0）。
func toRGBA(src image.Image) (pix []color.RGBA, w, h int) {
	b := src.Bounds()
	w, h = b.Dx(), b.Dy()
	pix = make([]color.RGBA, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, bb, a := src.At(b.Min.X+x, b.Min.Y+y).RGBA()
			pix[y*w+x] = color.RGBA{
				R: uint8(r >> 8),
				G: uint8(g >> 8),
				B: uint8(bb >> 8),
				A: uint8(a >> 8),
			}
		}
	}
	return pix, w, h
}

// luminance 计算 RGBA 像素的感知亮度（0~255）。
func luminance(c color.RGBA) float64 {
	if c.A == 0 {
		return 255 // 透明视为白底
	}
	return 0.299*float64(c.R) + 0.587*float64(c.G) + 0.114*float64(c.B)
}

// colorEuclid 计算两个 RGBA 在 0~255 空间的欧氏距离。
func colorEuclid(a, b color.RGBA) float64 {
	dr := float64(a.R) - float64(b.R)
	dg := float64(a.G) - float64(b.G)
	db := float64(a.B) - float64(b.B)
	return sqrtf(dr*dr + dg*dg + db*db)
}

// resizeGrayBilinear 把灰度缓冲双线性缩放到 size×size。
func resizeGrayBilinear(src []float64, sw, sh, size int) []float64 {
	dst := make([]float64, size*size)
	if sw <= 0 || sh <= 0 {
		return dst
	}
	for y := 0; y < size; y++ {
		sy := float64(y) * float64(sh) / float64(size)
		y0 := int(sy)
		if y0 >= sh-1 {
			y0 = sh - 1
		}
		if y0 < 0 {
			y0 = 0
		}
		fy := sy - float64(y0)
		if y0+1 >= sh {
			fy = 0
		}
		y1 := y0 + 1
		if y1 >= sh {
			y1 = y0
		}
		for x := 0; x < size; x++ {
			sx := float64(x) * float64(sw) / float64(size)
			x0 := int(sx)
			if x0 >= sw-1 {
				x0 = sw - 1
			}
			if x0 < 0 {
				x0 = 0
			}
			fx := sx - float64(x0)
			if x0+1 >= sw {
				fx = 0
			}
			x1 := x0 + 1
			if x1 >= sw {
				x1 = x0
			}
			c00 := src[y0*sw+x0]
			c10 := src[y0*sw+x1]
			c01 := src[y1*sw+x0]
			c11 := src[y1*sw+x1]
			dst[y*size+x] = c00*(1-fx)*(1-fy) + c10*fx*(1-fy) + c01*(1-fx)*fy + c11*fx*fy
		}
	}
	return dst
}

// sqrtf 平方根包装，仅为统一取 math.Sqrt。
func sqrtf(x float64) float64 {
	return math.Sqrt(x)
}
