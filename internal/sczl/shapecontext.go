package sczl

import "math"

// shapecontext.go Shape Context 局部形状描述子。
// 对每个采样点，统计其余所有点在其 log-polar 坐标下的分布。
// log-径提供尺度不变；角度+径直方图刻画局部邻域结构；
// 匹配两形状时用 χ² 距离 + 匈牙利一对一分配（复用现有算法框架，换距离）。

// scLogMin / scLogMax 为对数径 bin 的范围（点已归一化到 [0,1]，max 距离 ~√2）。
const (
	scRMin = 0.1 // 内圈半径（归一化空间）
	scRMax = 2.0 // 外圈半径
)

// computeSC 对采样点集计算每个点的 shape-context 直方图（行优先）。
// 返回 n × scBins 的二维切片；每行求和为 1（归一化）。
func computeSC(pts []Point) [][]float64 {
	n := len(pts)
	if n < 2 {
		return nil
	}
	// 预计算径 bin 对数边界。
	logRMin := math.Log(scRMin)
	logRMax := math.Log(scRMax)
	rBin := func(r float64) int {
		if r <= scRMin {
			return 0
		}
		t := (math.Log(r) - logRMin) / (logRMax - logRMin)
		b := int(t * float64(scRadialBins))
		if b < 0 {
			b = 0
		}
		if b >= scRadialBins {
			b = scRadialBins - 1
		}
		return b
	}
	aBin := func(ang float64) int {
		// ang ∈ [0, 2π)
		for ang < 0 {
			ang += 2 * math.Pi
		}
		for ang >= 2*math.Pi {
			ang -= 2 * math.Pi
		}
		return int(ang / (2 * math.Pi) * float64(scAngularBins))
	}

	out := make([][]float64, n)
	for i := range out {
		out[i] = make([]float64, scBins)
	}
	for i := 0; i < n; i++ {
		pi := pts[i]
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			pj := pts[j]
			dx := pj.X - pi.X
			dy := pj.Y - pi.Y
			r := math.Sqrt(dx*dx + dy*dy)
			ang := math.Atan2(dy, dx)
			rb := rBin(r)
			ab := aBin(ang)
			out[i][ab*scRadialBins+rb]++
		}
		// 归一化。
		sum := 0.0
		for _, v := range out[i] {
			sum += v
		}
		if sum > 0 {
			for k := range out[i] {
				out[i][k] /= sum
			}
		}
	}
	return out
}

// scChiSquare 两点 shape-context 直方图的 χ² 距离（对称形式，越小越相似）。
func scChiSquare(a, b []float64) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 1
	}
	d := 0.0
	for k := range a {
		s := a[k] + b[k]
		if s <= 1e-12 {
			continue
		}
		diff := a[k] - b[k]
		d += diff * diff / s
	}
	return 0.5 * d
}

// scMatchCost 构建 query↔entry 两点集 shape-context 的代价矩阵。
// 为控制匈牙利规模，两例各自均匀下采样到 maxPts。
func scMatchCost(qSC [][]float64, eSC [][]float64, maxPts int) [][]float64 {
	q := subsampleSC(qSC, maxPts)
	e := subsampleSC(eSC, maxPts)
	if len(q) == 0 || len(e) == 0 {
		return nil
	}
	cost := make([][]float64, len(q))
	for i := range cost {
		cost[i] = make([]float64, len(e))
		for j := range cost[i] {
			cost[i][j] = scChiSquare(q[i], e[j])
		}
	}
	return cost
}

// subsampleSC 等间隔下采样 SC 矩阵到 maxPts 行（不足则原样返回）。
func subsampleSC(sc [][]float64, maxPts int) [][]float64 {
	n := len(sc)
	if n <= maxPts || n == 0 {
		return sc
	}
	out := make([][]float64, 0, maxPts)
	step := float64(n) / float64(maxPts)
	for k := 0; k < maxPts; k++ {
		idx := int(float64(k) * step)
		if idx >= n {
			idx = n - 1
		}
		out = append(out, sc[idx])
	}
	return out
}

// scSimilarity 由匈牙利分配结果计算 shape-context 相似度 [0,1]。
// d_avg = 平均匹配代价（χ² ∈ [0,1]）；sim = 1 - d_avg。
func scSimilarity(cost [][]float64, assign []int) float64 {
	if len(cost) == 0 || len(assign) == 0 {
		return 0
	}
	cnt := 0
	sum := 0.0
	for i, j := range assign {
		if j < 0 {
			continue
		}
		sum += cost[i][j]
		cnt++
	}
	if cnt == 0 {
		return 0
	}
	sim := 1 - sum/float64(cnt)
	if sim < 0 {
		sim = 0
	}
	return sim
}
