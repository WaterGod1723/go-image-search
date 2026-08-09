package sczl

import "math"

// maskrot.go 旋转不变的占据栅格匹配。
//
// 原始 occSim 对 query/entry 的 64×64 软占据栅格做直接余弦，旋转即失配
// （栅格是固定朝向的空间布局）。maskRotSim 旋转 query 侧的 occ64 多个角度，
// 每步双线性重采样后重新 L2 归一化，与 entry occ64 做余弦，取最大值。
//
// 旋转只发生在 query 侧（参考侧不动），ang=0 那步即原 occSim 的 64 分辨率项，
// 故非旋转场景不会退化；旋转场景取最佳对齐恢复判别力。

// maskRotSim 旋转 query 的 64×64 软占据栅格，求与 entry 的最佳余弦相似度。
// steps 个角度均匀覆盖 [0, 2π)。steps<=1 时退化为基础余弦（不旋转）。
func maskRotSim(qOcc, eOcc []float64, steps int) float64 {
	const N = normSize // 64
	if len(qOcc) != N*N || len(eOcc) != N*N {
		return cosine(qOcc, eOcc)
	}
	if steps <= 1 {
		return cosine(qOcc, eOcc)
	}
	// entry 侧预归一化（cosine 内部会算，但这里每步复用，提前算省重复）。
	eNorm := l2normalize(eOcc)
	best := 0.0
	for s := 0; s < steps; s++ {
		ang := 2 * math.Pi * float64(s) / float64(steps)
		rot := rotateOccBilinear(qOcc, N, ang)
		rotNorm := l2normalize(rot)
		c := dot(rotNorm, eNorm)
		if c > best {
			best = c
		}
	}
	return best
}

// rotateOccBilinear 将 size×size 的占据栅格绕中心旋转 ang 弧度（双线性逆映射）。
// 返回旋转后的栅格（未归一化）。
func rotateOccBilinear(src []float64, size int, ang float64) []float64 {
	out := make([]float64, size*size)
	cs, sn := math.Cos(ang), math.Sin(ang)
	c := float64(size-1) / 2
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			// 逆映射：out(x,y) = src(旋转 -ang 后的坐标)
			dx, dy := float64(x)-c, float64(y)-c
			sx := dx*cs + dy*sn + c
			sy := -dx*sn + dy*cs + c
			out[y*size+x] = sampleBilinear(src, size, sx, sy)
		}
	}
	return out
}

// sampleBilinear 在 size×size 栅格上做双线性采样，越界裁剪到边缘。
func sampleBilinear(src []float64, size int, x, y float64) float64 {
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	max := float64(size - 1)
	if x > max {
		x = max
	}
	if y > max {
		y = max
	}
	x0, y0 := int(x), int(y)
	x1, y1 := x0+1, y0+1
	if x1 >= size {
		x1 = size - 1
	}
	if y1 >= size {
		y1 = size - 1
	}
	fx, fy := x-float64(x0), y-float64(y0)
	v00 := src[y0*size+x0]
	v10 := src[y0*size+x1]
	v01 := src[y1*size+x0]
	v11 := src[y1*size+x1]
	top := v00*(1-fx) + v10*fx
	bot := v01*(1-fx) + v11*fx
	return top*(1-fy) + bot*fy
}

// dot 两向量的点积（不归一化）。
func dot(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	if s < 0 {
		return 0
	}
	return s
}

// precomputeNCCRots 预计算 query NCC patch 的多个旋转副本（去均值 + L2 归一化）。
// 旋转 query 侧，entry 侧不动。返回 steps 个归一化向量（含 ang=0 原始）。
// 精排时每候选只需对这 steps 个向量求点积取最大，避免重复旋转。
func precomputeNCCRots(qPatch []float64, steps int) [][]float64 {
	const N = normSize
	if len(qPatch) != N*N || steps <= 1 {
		return nil
	}
	rots := make([][]float64, steps)
	for s := 0; s < steps; s++ {
		ang := 2 * math.Pi * float64(s) / float64(steps)
		rot := rotateOccBilinear(qPatch, N, ang)
		rots[s] = nccPatch(rot) // 去均值 + L2 归一化
	}
	return rots
}

// bestNCCDot 在预计算的 query 旋转 NCC 向量中，找与 entry patch 点积最大的值。
// rots 为 nil 时退化为基础 NCC 点积。
func bestNCCDot(rots [][]float64, ePatch, qPatch []float64) float64 {
	if len(rots) == 0 {
		return dot(qPatch, ePatch)
	}
	best := 0.0
	for _, r := range rots {
		c := dot(r, ePatch)
		if c > best {
			best = c
		}
	}
	return best
}

// precomputeOccRots 预计算 query occ64 的多个旋转副本（L2 归一化）。
// 旋转 query 侧，entry 侧不动。返回 steps 个归一化向量（含 ang=0 原始）。
// 粗排时每候选只需对这 steps 个向量求点积取最大，避免重复旋转。
func precomputeOccRots(qOcc []float64, steps int) [][]float64 {
	const N = normSize
	if len(qOcc) != N*N || steps <= 1 {
		return nil
	}
	rots := make([][]float64, steps)
	for s := 0; s < steps; s++ {
		ang := 2 * math.Pi * float64(s) / float64(steps)
		rot := rotateOccBilinear(qOcc, N, ang)
		rots[s] = l2normalize(rot)
	}
	return rots
}

// bestOccDot 在预计算的 query 旋转 occ 向量中，找与 entry occ 点积最大的值。
// rots 和 eOcc 均已 L2 归一化，故 dot = cosine。rots 为 nil 时退化为基础余弦。
func bestOccDot(rots [][]float64, eOcc, qOcc []float64) float64 {
	if len(rots) == 0 {
		return cosine(qOcc, eOcc)
	}
	best := 0.0
	for _, r := range rots {
		c := dot(r, eOcc)
		if c > best {
			best = c
		}
	}
	return best
}
