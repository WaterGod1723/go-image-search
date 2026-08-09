package sczl

// lbp.go uniform LBP(8,1) 纹理直方图。
// LBP 编码每个像素 8 邻域相对中心的二值模式，对单调光照变化不变（颜色无关）。
// uniform 模式（圆周上 0↔1 跳变 ≤2 次）共 58 种，其余归入 1 个"非 uniform"桶，
// 共 59 维。仅在前景内部像素统计，避免背景/边界干扰。

// lbpUniformMap 预计算 256 个 8-bit 模式到直方图 bin 的映射。
var lbpUniformMap [256]int

func init() {
	for c := 0; c < 256; c++ {
		if isUniformLBP(byte(c)) {
			// 给每个 uniform 模式分配唯一 bin（0..57）。
			lbpUniformMap[c] = uniformIndex(byte(c))
		} else {
			lbpUniformMap[c] = lbpDims - 1 // 非 uniform 最后一桶
		}
	}
}

// isUniformLBP 判定 8-bit 圆周模式跳变 ≤2。
func isUniformLBP(c byte) bool {
	bits := [8]bool{}
	for i := 0; i < 8; i++ {
		bits[i] = (c>>uint(i))&1 == 1
	}
	trans := 0
	for i := 0; i < 8; i++ {
		if bits[i] != bits[(i+1)%8] {
			trans++
		}
	}
	return trans <= 2
}

// uniformIndex 为 uniform 模式分配稳定 bin 号（0..57）。
// 用运行时计数保证同一模式同号、不同模式异号。
func uniformIndex(c byte) int {
	// 离线枚举所有 uniform 模式，按数值升序固定下标。
	idx := 0
	for v := 0; v < 256; v++ {
		if !isUniformLBP(byte(v)) {
			continue
		}
		if byte(v) == c {
			return idx
		}
		idx++
	}
	return lbpDims - 1
}

// lbpHistogram 计算归一化 uniform LBP 直方图。
// gray 为 normSize×normSize 灰度（0~255），mask 同尺寸前景掩码。
// 仅对"中心及 8 邻均为前景"的像素统计，消除背景/边界噪声。
func lbpHistogram(gray []float64, mask []bool) []float64 {
	n := normSize
	if len(gray) != n*n || len(mask) != n*n {
		return nil
	}
	hist := make([]float64, lbpDims)
	// 8 邻顺序（顺时针，从右上起），与 isUniformLBP 的位序一致。
	dxs := [8]int{1, 1, 0, -1, -1, -1, 0, 1}
	dys := [8]int{-1, 0, 1, 1, 0, -1, -1, -1}
	cnt := 0
	for y := 1; y < n-1; y++ {
		for x := 1; x < n-1; x++ {
			ci := y*n + x
			if !mask[ci] {
				continue
			}
			center := gray[ci]
			// 检查 8 邻是否均为前景。
			ok := true
			for k := 0; k < 8; k++ {
				nx, ny := x+dxs[k], y+dys[k]
				if !mask[ny*n+nx] {
					ok = false
					break
				}
			}
			if !ok {
				continue
			}
			// 编码 LBP。
			var code byte
			for k := 0; k < 8; k++ {
				nx, ny := x+dxs[k], y+dys[k]
				if gray[ny*n+nx] >= center {
					code |= 1 << uint(k)
				}
			}
			hist[lbpUniformMap[code]]++
			cnt++
		}
	}
	if cnt > 0 {
		for i := range hist {
			hist[i] /= float64(cnt)
		}
	}
	return hist
}

// lbpIntersect 两 LBP 直方图归一化交集相似度 [0,1]。
func lbpIntersect(a, b []float64) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	s := 0.0
	for i := range a {
		x := a[i]
		if b[i] < x {
			x = b[i]
		}
		s += x
	}
	return s
}
