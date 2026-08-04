package sczl

import (
	"image"
	"math"
)

// extract.go 描述子提取总装：图像 → 前景（含文字带剔除）→ 轮廓 → 占据栅格/FD/径向/SC/LBP → Descriptor。

// Extract 从图像提取 SCZL 描述子。失败（无法提取前景）返回 invalid Descriptor。
func Extract(img image.Image) Descriptor {
	pix, w, h := toRGBA(img)
	if w == 0 || h == 0 {
		return Descriptor{}
	}
	cfg := defaultFGConfig()
	fg, bbox, ok := extractForeground(pix, w, h, cfg)
	if !ok {
		return Descriptor{}
	}
	// 文字带剔除：剔除与主体分离的附属文字带/填充带。
	fg, bbox = trimTextBands(fg, w, h, bbox)
	if !hasForeground(fg) {
		return Descriptor{}
	}

	pts, radii := radialBoundary(fg, bbox, contourSamples)
	if len(pts) != contourSamples {
		return Descriptor{}
	}

	fd := fourierDescriptors(pts)
	radial := radialHistogram(radii)
	sc := computeSC(pts)

	// 归一化 64×64 灰度块 + 软前景掩码（scale-normalized，消除源图尺寸差异）。
	grayFull := make([]float64, w*h)
	for i, c := range pix {
		grayFull[i] = luminance(c)
	}
	grayPatch := cropResizeGray(grayFull, w, bbox, normSize)
	maskSoft := cropResizeMaskSoft(fg, bbox, normSize)
	maskBin := thresholdMask(maskSoft, 0.5)
	lbp := lbpHistogram(grayPatch, maskBin)

	// 占据栅格从 64×64 软掩码下采样，保证查询/图库同尺寸可比。
	occ16 := downsampleMask(maskSoft, normSize, occBin16)
	occ32 := downsampleMask(maskSoft, normSize, occBin32)
	occ64 := l2normalize(maskSoft)

	// 面积元信息。
	area := 0
	for _, v := range fg.cell {
		if v {
			area++
		}
	}

	return Descriptor{
		Occupancy64: occ64,
		Occupancy16: occ16,
		Occupancy32: occ32,
		Fourier:     fd,
		Radial:      radial,
		SC:          sc,
		SCPoints:    pts,
		LBP:         lbp,
		Area:        area,
		BBox:        bbox,
		Valid:       true,
	}
}

// l2normalize 对向量做 L2 归一化（就地拷贝返回新切片）。
func l2normalize(v []float64) []float64 {
	out := make([]float64, len(v))
	copy(out, v)
	norm := 0.0
	for _, x := range out {
		norm += x * x
	}
	if norm > 0 {
		norm = math.Sqrt(norm)
		for i := range out {
			out[i] /= norm
		}
	}
	return out
}

// hasForeground 判断掩码是否仍有前景像素。
func hasForeground(m fgMask) bool {
	for _, v := range m.cell {
		if v {
			return true
		}
	}
	return false
}

// cropResizeMaskSoft 把前景掩码在 bbox 内按面积比例软占据到 size×size：
// 每格取该子区域前景像素占比（[0,1]）。软占据对边界对齐误差稳健。
// 关键：以固定 size 采样，使查询（小截图）与图库（大原图）产生可比的占据栅格。
func cropResizeMaskSoft(fg fgMask, bbox image.Rectangle, size int) []float64 {
	bw, bh := bbox.Dx(), bbox.Dy()
	out := make([]float64, size*size)
	if bw <= 0 || bh <= 0 {
		return out
	}
	for gy := 0; gy < size; gy++ {
		y0 := bbox.Min.Y + gy*bh/size
		y1 := bbox.Min.Y + (gy+1)*bh/size
		if y1 <= y0 {
			y1 = y0 + 1
		}
		if y1 > bbox.Min.Y+bh {
			y1 = bbox.Min.Y + bh
		}
		for gx := 0; gx < size; gx++ {
			x0 := bbox.Min.X + gx*bw/size
			x1 := bbox.Min.X + (gx+1)*bw/size
			if x1 <= x0 {
				x1 = x0 + 1
			}
			if x1 > bbox.Min.X+bw {
				x1 = bbox.Min.X + bw
			}
			fgCnt, tot := 0, 0
			for yy := y0; yy < y1; yy++ {
				for xx := x0; xx < x1; xx++ {
					tot++
					if fg.at(xx, yy) {
						fgCnt++
					}
				}
			}
			if tot > 0 {
				out[gy*size+gx] = float64(fgCnt) / float64(tot)
			}
		}
	}
	return out
}

// thresholdMask 把软掩码二值化（>= t 视为前景）。
func thresholdMask(soft []float64, t float64) []bool {
	out := make([]bool, len(soft))
	for i, v := range soft {
		out[i] = v >= t
	}
	return out
}

// downsampleMask 把 size×size 软掩码按块平均下采样到 dst×dst。
func downsampleMask(soft []float64, size, dst int) []float64 {
	if dst <= 0 || size <= 0 || len(soft) != size*size {
		return nil
	}
	out := make([]float64, dst*dst)
	if dst >= size {
		out[0] = 0
		return out
	}
	// 每个目标格覆盖 size/dst × size/dst 的源块（近似整数分块）。
	for gy := 0; gy < dst; gy++ {
		y0 := gy * size / dst
		y1 := (gy + 1) * size / dst
		if y1 <= y0 {
			y1 = y0 + 1
		}
		for gx := 0; gx < dst; gx++ {
			x0 := gx * size / dst
			x1 := (gx + 1) * size / dst
			if x1 <= x0 {
				x1 = x0 + 1
			}
			sum, cnt := 0.0, 0
			for yy := y0; yy < y1; yy++ {
				for xx := x0; xx < x1; xx++ {
					sum += soft[yy*size+xx]
					cnt++
				}
			}
			if cnt > 0 {
				out[gy*dst+gx] = sum / float64(cnt)
			}
		}
	}
	// L2 归一化便于余弦比较。
	norm := 0.0
	for _, v := range out {
		norm += v * v
	}
	if norm > 0 {
		norm = math.Sqrt(norm)
		for i := range out {
			out[i] /= norm
		}
	}
	return out
}

// cropResizeGray 从全图灰度缓冲裁剪 bbox 子区域并双线性缩放到 size×size。
func cropResizeGray(grayFull []float64, fullW int, bbox image.Rectangle, size int) []float64 {
	bw, bh := bbox.Dx(), bbox.Dy()
	if bw <= 0 || bh <= 0 {
		return make([]float64, size*size)
	}
	sub := make([]float64, bw*bh)
	for y := 0; y < bh; y++ {
		for x := 0; x < bw; x++ {
			sub[y*bw+x] = grayFull[(bbox.Min.Y+y)*fullW+(bbox.Min.X+x)]
		}
	}
	return resizeGrayBilinear(sub, bw, bh, size)
}

// cropResizeMask 把前景掩码裁剪 bbox 后缩放到 size×size（阈值二值化）。
func cropResizeMask(fg fgMask, bbox image.Rectangle, size int) []bool {
	bw, bh := bbox.Dx(), bbox.Dy()
	if bw <= 0 || bh <= 0 {
		return make([]bool, size*size)
	}
	// 直接对原始掩码做最近邻缩放（前景/背景二值，最近邻保边）。
	out := make([]bool, size*size)
	for y := 0; y < size; y++ {
		sy := int(math.Floor(float64(y) * float64(bh) / float64(size)))
		if sy >= bh {
			sy = bh - 1
		}
		for x := 0; x < size; x++ {
			sx := int(math.Floor(float64(x) * float64(bw) / float64(size)))
			if sx >= bw {
				sx = bw - 1
			}
			out[y*size+x] = fg.at(bbox.Min.X+sx, bbox.Min.Y+sy)
		}
	}
	return out
}
