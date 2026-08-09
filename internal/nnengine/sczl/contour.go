package sczl

import (
	"math"
)

// contour.go 轮廓采样（径向射线法，旋转不变归一化）。
//
// 归一化关键：所有点/距离以质心为原点、R98*frame.Scale 为尺度基准。
// 旋转不改变质心位置（刚体变换）、不改变点到质心的距离，因此归一化框架
// 对旋转完全不变，解决旋转后 bbox 扭曲导致的对齐失真。

// radialBoundary 从质心 frame.CX/CY 发射 n 个等角度射线，记录沿射线最外的前景像素。
// 对"质心落在前景外（凹形/空心）"稳健：射线先走直到首次进入前景，再继续至出前景为止，
// 记录沿该射线到达过的最远的前景位置（而非"首次碰到非前景"时停，避免质心外/空心误停）。
// 返回按角度有序的归一化点集（质心居中，frame.Scale 为单位尺度），以及归一化径向距离数组。
func radialBoundary(m fgMask, frame normFrame, n int) (pts []Point, radii []float64) {
	if n < 8 || frame.Scale <= 0 {
		return nil, nil
	}
	cx, cy := frame.CX, frame.CY
	scale := frame.Scale
	half := scale * 0.5
	if half <= 0 {
		return nil, nil
	}

	pts = make([]Point, 0, n)
	radii = make([]float64, 0, n)
	const step = 0.5
	// maxSteps 走到画布外或 2 倍画布对角线（防止死循环），保证最远前景一定被触达。
	maxSteps := int((math.Max(float64(m.w), float64(m.h)) * 3) / step)
	if maxSteps < 64 {
		maxSteps = 64
	}
	for k := 0; k < n; k++ {
		theta := 2 * math.Pi * float64(k) / float64(n)
		cth, sth := math.Cos(theta), math.Sin(theta)
		// 沿射线追踪：记录"最后一次处于前景"时的坐标。允许质心本身非前景（凹/空心）。
		lastX, lastY := cx, cy
		hasAny := false
		for s := 0; s <= maxSteps; s++ {
			t := step * float64(s)
			x := cx + t*cth
			y := cy + t*sth
			ix, iy := int(math.Round(x)), int(math.Round(y))
			if ix < 0 || ix >= m.w || iy < 0 || iy >= m.h {
				break
			}
			if m.at(ix, iy) {
				lastX, lastY = x, y
				hasAny = true
			}
			// 不 break，继续向前：空心图形会进进出出，我们要最外的。
		}
		if !hasAny {
			// 该射线方向完全没前景（少见），退回质心。
			pts = append(pts, normalizePixelToFrame(cx, cy, frame))
			radii = append(radii, 0)
			continue
		}
		dx := lastX - cx
		dy := lastY - cy
		// halfSide = scale/2：质心到正方形任一边的距离（像素）。
		// r ∈ [0, 1]：归一化径向距离，1 = 点落在正方形边上。
		halfSide := scale * 0.5
		r := 0.0
		if halfSide > 0 {
			r = math.Sqrt(dx*dx+dy*dy) / halfSide
		}
		pts = append(pts, normalizePixelToFrame(lastX, lastY, frame))
		radii = append(radii, r)
	}
	return pts, radii
}

// normalizePixelToFrame 将像素坐标 (px, py) 按 frame 归一化到 [0,1]^2 空间：
// 左边 cx - scale/2 → x=0；右边 cx + scale/2 → x=1；质心 (cx, cy) → (0.5, 0.5)。
// 旋转不变：质心+距离基准则旋转后不变。
func normalizePixelToFrame(px, py float64, frame normFrame) Point {
	if frame.Scale <= 0 {
		return Point{X: 0.5, Y: 0.5}
	}
	side := frame.Scale // 归一化正方形边长（像素）
	if side <= 0 {
		return Point{X: 0.5, Y: 0.5}
	}
	x := 0.5 + (px-frame.CX)/side
	y := 0.5 + (py-frame.CY)/side
	return Point{X: x, Y: y}
}
