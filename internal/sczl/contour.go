package sczl

import (
	"image"
	"math"
)

// contour.go 轮廓采样：径向射线法。
// 从前景质心出发，按等角度间隔发射 N 条射线，记录每条射线上"最远的前景像素"。
// 该点集天然按角度有序，构成闭合径向距离函数 r(θ)：
//   - 平移不变：原点取质心，随主体移动；
//   - 旋转不变：r(θ) 旋转即循环移位，Fourier 取幅值后消除；
//   - 缩放不变：r 归一化后 Fourier 取幅值并除以 |F_1|。
// 相比 Moore 边界追踪，径向法对噪声/反锯齿更稳，且实现简洁可验证。

// radialBoundary 从前景质心出发按 N 个等角度采样最远前景像素，
// 返回归一化坐标点集（按角度顺序）与归一化径向距离数组（用于 Fourier）。
// 若前景过小或无法采样，返回 nil。
func radialBoundary(m fgMask, bbox image.Rectangle, n int) (pts []Point, radii []float64) {
	if n < 8 || m.w == 0 || m.h == 0 {
		return nil, nil
	}
	cx, cy := foregroundCentroid(m)
	// 质心若落在前景外（凹陷区域），退化为 bbox 中心，保证原点稳健。
	if !m.at(int(cx), int(cy)) {
		cx = float64(bbox.Min.X) + float64(bbox.Dx())/2
		cy = float64(bbox.Min.Y) + float64(bbox.Dy())/2
	}

	// 最大半径 = bbox 对角线一半，作为归一化基准。
	maxR := math.Sqrt(float64(bbox.Dx()*bbox.Dx()+bbox.Dy()*bbox.Dy())) / 2
	if maxR <= 0 {
		return nil, nil
	}

	pts = make([]Point, 0, n)
	radii = make([]float64, 0, n)
	// 步长 0.5 像素，兼顾分辨率与速度。
	const step = 0.5
	for k := 0; k < n; k++ {
		theta := 2 * math.Pi * float64(k) / float64(n)
		cth, sth := math.Cos(theta), math.Sin(theta)
		// 沿射线前进，记录最后一个前景像素。
		lastX, lastY := cx, cy // 默认退化为质心
		found := false
		// 最远走到 bbox 边界外。
		maxSteps := 2 * (bbox.Dx() + bbox.Dy())
		for s := 1; s <= maxSteps; s++ {
			t := step * float64(s)
			x := cx + t*cth
			y := cy + t*sth
			ix, iy := int(math.Round(x)), int(math.Round(y))
			if !m.at(ix, iy) {
				break
			}
			lastX, lastY = x, y
			found = true
		}
		_ = found
		// 径向距离（像素），归一化。
		dx := lastX - cx
		dy := lastY - cy
		r := math.Sqrt(dx*dx+dy*dy) / maxR
		pts = append(pts, normalizePoint(lastX, lastY, bbox))
		radii = append(radii, r)
	}
	return pts, radii
}

// foregroundCentroid 返回前景像素的算术平均坐标。
func foregroundCentroid(m fgMask) (float64, float64) {
	var sx, sy float64
	cnt := 0
	for y := 0; y < m.h; y++ {
		for x := 0; x < m.w; x++ {
			if m.cell[y*m.w+x] {
				sx += float64(x)
				sy += float64(y)
				cnt++
			}
		}
	}
	if cnt == 0 {
		return 0, 0
	}
	return sx / float64(cnt), sy / float64(cnt)
}

// normalizePoint 将像素坐标按 bbox 居中、max(w,h) 缩放到 [0,1] 空间。
func normalizePoint(px, py float64, bbox image.Rectangle) Point {
	w := float64(bbox.Dx())
	h := float64(bbox.Dy())
	size := w
	if h > size {
		size = h
	}
	if size <= 0 {
		return Point{X: 0.5, Y: 0.5}
	}
	cx := float64(bbox.Min.X) + w/2
	cy := float64(bbox.Min.Y) + h/2
	x := 0.5 + (px-cx)/size
	y := 0.5 + (py-cy)/size
	return Point{X: x, Y: y}
}
