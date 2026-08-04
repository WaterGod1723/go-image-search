package sczl

import "math"

// fourier.go Fourier 描述子（FD）与径向距离直方图。
// FD 基于复数边界序列 z_k = (x_k-0.5) + i(y_k-0.5) 的 FFT，取 |F_k|（k=1..K）。
// |F_k| 天然具备：平移不变（已减质心）、旋转不变（幅值）、缩放不变（除以 |F_1|）、
// 起点不变（幅值）。径向直方图补充整体轮廓分布，作为粗粒度全局签名。

// fft128 对长度为 128 的复数序列做就地 radix-2 Cooley-Tukey FFT。
// 长度非 128 时返回（不做变换）。
func fft128(re, im []float64) {
	const n = 128
	if len(re) != n || len(im) != n {
		return
	}
	// 位反转置换。
	j := 0
	for i := 1; i < n; i++ {
		bit := n >> 1
		for j&bit != 0 {
			j ^= bit
			bit >>= 1
		}
		j |= bit
		if i < j {
			re[i], re[j] = re[j], re[i]
			im[i], im[j] = im[j], im[i]
		}
	}
	// 蝶形运算。
	for size := 2; size <= n; size <<= 1 {
		half := size >> 1
		ang := -2 * math.Pi / float64(size)
		wr := math.Cos(ang)
		wi := math.Sin(ang)
		for i := 0; i < n; i += size {
			cr, ci := 1.0, 0.0
			for k := 0; k < half; k++ {
				// t = w * a[i+k+half]
				tr := cr*re[i+k+half] - ci*im[i+k+half]
				ti := cr*im[i+k+half] + ci*re[i+k+half]
				re[i+k+half] = re[i+k] - tr
				im[i+k+half] = im[i+k] - ti
				re[i+k] += tr
				im[i+k] += ti
				// w *= w_size
				nr := cr*wr - ci*wi
				ci = cr*wi + ci*wr
				cr = nr
			}
		}
	}
}

// fourierDescriptors 由归一化边界点计算 K 维 Fourier 描述子。
// 返回归一化幅值 |F_1..F_K|（除以 |F_1|，使缩放不变更鲁棒）。
// 输入点数须等于 contourSamples（128）。
func fourierDescriptors(pts []Point) []float64 {
	if len(pts) != contourSamples {
		return nil
	}
	re := make([]float64, contourSamples)
	im := make([]float64, contourSamples)
	for i, p := range pts {
		re[i] = p.X - 0.5 // 居中（已平移归一化）
		im[i] = p.Y - 0.5
	}
	fft128(re, im)
	out := make([]float64, 0, fourierDims)
	// |F_1| 作为缩放基准（跳过 DC 即 k=0）。
	f1 := math.Hypot(re[1], im[1])
	if f1 <= 1e-9 {
		f1 = 1e-9
	}
	for k := 1; k <= fourierDims; k++ {
		mag := math.Hypot(re[k], im[k]) / f1
		out = append(out, mag)
	}
	return out
}

// radialHistogram 由径向距离数组（已归一化到 [0,1]）生成 B bin 归一化直方图。
func radialHistogram(radii []float64) []float64 {
	if len(radii) == 0 {
		return nil
	}
	hist := make([]float64, radialBins)
	for _, r := range radii {
		b := int(r * float64(radialBins))
		if b < 0 {
			b = 0
		}
		if b >= radialBins {
			b = radialBins - 1
		}
		hist[b]++
	}
	sum := 0.0
	for _, v := range hist {
		sum += v
	}
	if sum > 0 {
		for i := range hist {
			hist[i] /= sum
		}
	}
	return hist
}
