package sczl

import (
	"image"
	"math"
)

// extract.go 描述子提取总装：图像 → 前景 → 质心 R98 归一化框架 → 占据栅格/NCC/HOG/FD/SC/Regions。
//
// 归一化架构（旋转不变）：所有子描述子统一以 质心 为原点，R98*2.4 正方形 为尺度基准。
// 旋转后质心不变、点到质心距离不变 → 归一化框架不变 → 所有子描述子在旋转前/后对齐一致。
// 原 bbox 归一化会随旋转扭曲宽高比，导致 Occ/NCC/HOG/SC 归一化坐标失真，现已替换。

// Extract 从图像提取 SCZL 描述子。失败（无法提取前景）返回 invalid Descriptor。
func Extract(img image.Image) Descriptor {
	pix, w, h := toRGBA(img)
	if w == 0 || h == 0 {
		return Descriptor{}
	}
	cfg := defaultFGConfig()
	fg, _, ok := extractForeground(pix, w, h, cfg) // bbox 不再使用，用 frame 代替
	if !ok {
		return Descriptor{}
	}
	fg, _ = trimTextBands(fg, w, h, image.Rect(0, 0, w, h)) // bbox 仅用于画布安全下界检查，不传实际 bbox
	if !hasForeground(fg) {
		return Descriptor{}
	}

	// 旋转不变归一化框架：质心 + R98*2.4。
	frame, ok := makeNormFrame(fg)
	if !ok {
		return Descriptor{}
	}

	// 掩码区域划分（多部件判别），传 frame 以做质心/R98 归一化。
	regions := extractRegions(fg, frame)

	// 64×64 灰度块 + 软前景掩码：按质心+R98 正方形裁剪（不再用 bbox），
	// 使旋转后的图标与原图缩放/裁剪完全一致。
	grayFull := make([]float64, w*h)
	for i, c := range pix {
		grayFull[i] = luminance(c)
	}
	grayPatch := cropFrameGray(grayFull, w, frame, normSize)
	maskSoft := cropFrameMaskSoft(fg, frame, normSize)
	maskBin := thresholdMask(maskSoft, 0.5)

	// HOG：基于 Sobel 梯度，仅前景掩码贡献。
	hog := hogDescriptor(grayPatch, maskBin)

	// 聚合策略：碎片化/镂空图标（实度<fillThr）做孔洞填充。
	sol := solidity(maskSoft)
	occSrc := maskSoft
	if sol < fillThr {
		occSrc = holeFill(maskSoft)
	}
	occ16 := downsampleMask(occSrc, normSize, occBin16)
	occ32 := downsampleMask(occSrc, normSize, occBin32)
	occ64 := l2normalize(occSrc)

	// NCC 归一化互相关模板：去均值 + L2 归一化。
	patch := nccPatch(grayPatch)

	// BBox 紧裁剪备份（非旋转精确匹配）：用轴对齐 bbox 紧裁剪到 64×64 正方形，
	// 内容填满栅格（无空白稀释），对非旋转变体的 Occ/NCC 精度更高。
	// 搜索时与 Rmax 旋转扫描版取 max，两种场景各取所长。
	bbox := frameBBox(fg)
	grayPatchBBox := cropBBoxGray(grayFull, w, bbox, normSize)
	maskSoftBBox := cropBBoxMaskSoft(fg, bbox, normSize)
	occBBox64 := l2normalize(maskSoftBBox)
	patchBBox := nccPatch(grayPatchBBox)

	// 轮廓签名（FD/径向/SC）：质心/R98 归一化框架下的径向采样。
	pts, radii := radialBoundary(fg, frame, contourSamples)
	var fd []float64
	var radial []float64
	var sc [][]float64
	if len(pts) == contourSamples {
		fd = fourierDescriptors(pts)
		radial = radialHistogram(radii)
		sc = computeSC(pts)
	}

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
		Patch:       patch,
		HOG:         hog,
		Regions:     regions,
		Solidity:    sol,
		Area:        area,
		BBox:        bbox,
		OccBBox64:   occBBox64,
		PatchBBox:   patchBBox,
		Valid:       true,
	}
}

// frameBBox 从 fg 计算一个仅参考的轴对齐 bbox（不再影响描述子）。
func frameBBox(m fgMask) image.Rectangle {
	minX, minY, maxX, maxY := m.w, m.h, -1, -1
	for y := 0; y < m.h; y++ {
		for x := 0; x < m.w; x++ {
			if m.cell[y*m.w+x] {
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	if maxX < 0 {
		return image.Rect(0, 0, 0, 0)
	}
	return image.Rect(minX, minY, maxX+1, maxY+1)
}

// cropFrameMaskSoft 按 normFrame 正方形区域（中心=质心，边长=frame.Scale）软缩放到 size×size。
// 每个 size×size 格取其对应像素窗口的前景比例（[0,1]）。画布外视为非前景（0）。
// 与原 bbox 裁剪的核心差异：正方形旋转后仍然正方形（bbox 长方形旋转后会变扁/变高）。
func cropFrameMaskSoft(fg fgMask, frame normFrame, size int) []float64 {
	out := make([]float64, size*size)
	if frame.Scale <= 0 {
		return out
	}
	// 每个 size 格覆盖的像素宽高
	pxPerCell := frame.Scale / float64(size)
	if pxPerCell <= 0 {
		return out
	}
	for gy := 0; gy < size; gy++ {
		// 该格中心在像素坐标系：质心 + (gy - size/2)/size * scale
		// 取整数窗口
		winY0 := frame.CY + (float64(gy)/float64(size)-0.5)*frame.Scale
		winY1 := winY0 + pxPerCell
		iy0, iy1 := int(math.Floor(winY0)), int(math.Ceil(winY1))
		for gx := 0; gx < size; gx++ {
			winX0 := frame.CX + (float64(gx)/float64(size)-0.5)*frame.Scale
			winX1 := winX0 + pxPerCell
			ix0, ix1 := int(math.Floor(winX0)), int(math.Ceil(winX1))
			fgCnt, tot := 0, 0
			for yy := iy0; yy < iy1; yy++ {
				for xx := ix0; xx < ix1; xx++ {
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

// cropFrameGray 按质心/R98 正方形裁剪灰度块并双线性缩放到 size×size。
func cropFrameGray(grayFull []float64, fullW int, frame normFrame, size int) []float64 {
	out := make([]float64, size*size)
	if frame.Scale <= 0 {
		return out
	}
	for gy := 0; gy < size; gy++ {
		// size×size 网格 (gx, gy) 的中心在像素坐标系中的位置：
		py := frame.CY + (float64(gy)+0.5)/float64(size)*frame.Scale - frame.Scale*0.5
		for gx := 0; gx < size; gx++ {
			px := frame.CX + (float64(gx)+0.5)/float64(size)*frame.Scale - frame.Scale*0.5
			// 双线性采样 grayFull[py, px]
			x0 := int(math.Floor(px))
			y0 := int(math.Floor(py))
			x1, y1 := x0+1, y0+1
			fx, fy := px-float64(x0), py-float64(y0)
			v := func(x, y int) float64 {
				if x < 0 || y < 0 || x >= fullW {
					return 0
				}
				i := y*fullW + x
				if i < 0 || i >= len(grayFull) {
					return 0
				}
				return grayFull[i]
			}
			v00 := v(x0, y0)
			v10 := v(x1, y0)
			v01 := v(x0, y1)
			v11 := v(x1, y1)
			top := v00*(1-fx) + v10*fx
			bot := v01*(1-fx) + v11*fx
			out[gy*size+gx] = top*(1-fy) + bot*fy
		}
	}
	return out
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

// nccPatch 将灰度块做去均值 + L2 归一化，供 NCC 互相关匹配。
// 背景（非 mask 区域）在调用端已置为 0，但不参与去均值会让前景均值偏高，
// 因此使用"全 patch 含背景"的均值（与 0 背景合起来相当于"灰度减去全局均值"）。
// 对 legacy 非旋转匹配，这保证与原 bbox 紧裁剪（背景=0但几乎没背景）的行为一致；
// 对质心/Rmax 裁剪（四边有 0 背景），0 背景会拉低均值但前景值-均值的差值
// 被 L2 归一化统一缩放后整体比例保持，仍是正确的 cross-correlation 度量。
func nccPatch(grayPatch []float64) []float64 {
	n := len(grayPatch)
	if n == 0 {
		return nil
	}
	mean := 0.0
	for _, v := range grayPatch {
		mean += v
	}
	mean /= float64(n)
	out := make([]float64, n)
	for i, v := range grayPatch {
		out[i] = v - mean
	}
	return l2normalize(out)
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

// cropBBoxMaskSoft 用轴对齐 bbox 紧裁剪到正方形（取 max(w,h) 居中）再缩放到 size×size 软掩码。
// 与 cropFrameMaskSoft 的核心差异：bbox 紧裁剪让内容填满整个栅格（无空白稀释），
// 对非旋转变体的 Occ 精度更高；但旋转后 bbox 扭曲（长方形变斜），故仅用于非旋转路径。
func cropBBoxMaskSoft(fg fgMask, bbox image.Rectangle, size int) []float64 {
	out := make([]float64, size*size)
	if bbox.Empty() {
		return out
	}
	// 正方形边长 = max(bbox 宽, 高)，以 bbox 中心居中。
	side := float64(bbox.Dx())
	if bh := float64(bbox.Dy()); bh > side {
		side = bh
	}
	if side <= 0 {
		return out
	}
	cx := float64(bbox.Min.X) + float64(bbox.Dx())*0.5
	cy := float64(bbox.Min.Y) + float64(bbox.Dy())*0.5
	pxPerCell := side / float64(size)
	for gy := 0; gy < size; gy++ {
		winY0 := cy + (float64(gy)/float64(size)-0.5)*side
		winY1 := winY0 + pxPerCell
		iy0, iy1 := int(math.Floor(winY0)), int(math.Ceil(winY1))
		for gx := 0; gx < size; gx++ {
			winX0 := cx + (float64(gx)/float64(size)-0.5)*side
			winX1 := winX0 + pxPerCell
			ix0, ix1 := int(math.Floor(winX0)), int(math.Ceil(winX1))
			fgCnt, tot := 0, 0
			for yy := iy0; yy < iy1; yy++ {
				for xx := ix0; xx < ix1; xx++ {
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

// cropBBoxGray 用 bbox 紧裁剪灰度块并双线性缩放到 size×size（正方形紧裁剪）。
func cropBBoxGray(grayFull []float64, fullW int, bbox image.Rectangle, size int) []float64 {
	out := make([]float64, size*size)
	if bbox.Empty() {
		return out
	}
	side := float64(bbox.Dx())
	if bh := float64(bbox.Dy()); bh > side {
		side = bh
	}
	if side <= 0 {
		return out
	}
	cx := float64(bbox.Min.X) + float64(bbox.Dx())*0.5
	cy := float64(bbox.Min.Y) + float64(bbox.Dy())*0.5
	for gy := 0; gy < size; gy++ {
		py := cy + (float64(gy)+0.5)/float64(size)*side - side*0.5
		for gx := 0; gx < size; gx++ {
			px := cx + (float64(gx)+0.5)/float64(size)*side - side*0.5
			x0 := int(math.Floor(px))
			y0 := int(math.Floor(py))
			x1, y1 := x0+1, y0+1
			fx, fy := px-float64(x0), py-float64(y0)
			v := func(x, y int) float64 {
				if x < 0 || y < 0 || x >= fullW {
					return 0
				}
				i := y*fullW + x
				if i < 0 || i >= len(grayFull) {
					return 0
				}
				return grayFull[i]
			}
			v00 := v(x0, y0)
			v10 := v(x1, y0)
			v01 := v(x0, y1)
			v11 := v(x1, y1)
			top := v00*(1-fx) + v10*fx
			bot := v01*(1-fx) + v11*fx
			out[gy*size+gx] = top*(1-fy) + bot*fy
		}
	}
	return out
}
