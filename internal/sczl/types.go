// Package sczl 实现基于"轮廓形状上下文 + Fourier 描述子 + LBP 纹理"的
// icon 反向检索算法（SCZL）。与 internal/index 的区域感知哈希方案并行，
// 不依赖颜色分割：前景提取以边界泛洪去背景 + 连通域为主，颜色仅作弱辅证，
// 主判别力来自颜色无关的形状与纹理签名，从而对"背景色变化/icon 颜色变化/
// 四周文字/填充区域/线条干扰"更鲁棒。
//
// 全局签名 Fourier 描述子取幅值，天然具备 平移/旋转/缩放/起点 不变性；
// 径向距离直方图补充整体轮廓分布；Shape Context 提供局部形状判别；
// uniform LBP 提供颜色无关的局部纹理判别。
package sczl

import "image"

// Point 归一化坐标点（0~1 空间）。
type Point struct{ X, Y float64 }

// Descriptor 一张 icon 的多视角签名。
type Descriptor struct {
	ImageID string

	// 占据栅格（主判别力）：归一化前景掩码的多分辨率软栅格，
	// 直接刻画内部结构（网格十字、环 vs 实心、镂空等），弥补纯轮廓丢失内部信息。
	Occupancy64 []float64 // 64×64 软占据（高分辨率，保留细缝隙），4096 维
	Occupancy32 []float64 // 32×32 软占据，1024 维
	Occupancy16 []float64 // 16×16 软占据，256 维

	// 全局旋转/缩放/平移不变签名（外轮廓）。
	Fourier []float64 // 归一化 |FD|，K 维
	Radial  []float64  // 径向距离直方图（归一化），R 维

	// 局部形状签名。
	SC       [][]float64 // nPoints × nBins shape-context 直方图
	SCPoints []Point     // 采样轮廓点（归一化坐标）

	// 颜色无关纹理。
	LBP []float64 // uniform LBP 直方图（归一化）

	// 辅助元信息。
	Area int
	BBox image.Rectangle
	Valid bool
}

// Match 图像级匹配结果。
type Match struct {
	ImageID string
	Score   float64 // [0,1] 越大越相似
	FDSim   float64 // 全局签名相似度
	SCSim   float64 // shape-context 相似度
	LBPSim  float64 // LBP 纹理相似度
}

// Options 检索参数。
type Options struct {
	TopK          int     // 返回数量，<=0 取 5
	OccWeight     float64 // 占据栅格权重，<=0 取 0.6
	SCWeight      float64 // SC 权重，<=0 取 0.15
	FDWeight      float64 // Fourier 权重，<=0 取 0.1
	LBPWeight     float64 // LBP 权重，<=0 取 0.15
	FDPreFilter   int     // 占据+FD 召回后保留的候选数，<=0 取 200
	FDCut         float64 // 全局相似度截断（低于此值直接淘汰），<=0 取 0.4
	SCMaxPoints   int     // SC 匹配时每侧采样点数上限（控制匈牙利规模），<=0 取 48
}

// DefaultOptions 推荐默认参数。
func DefaultOptions() Options {
	return Options{
		TopK:        5,
		OccWeight:   0.6,
		SCWeight:    0.15,
		FDWeight:    0.1,
		LBPWeight:   0.15,
		FDPreFilter: 200,
		FDCut:       0.4,
		SCMaxPoints: 48,
	}
}

// 描述子维度常量。
const (
	// contourSamples 轮廓重采样点数（FFT 要求 2 的幂）。
	contourSamples = 128
	// fourierDims 保留的 Fourier 描述子个数（不含 DC，从 k=1 起）。
	fourierDims = 32
	// radialBins 径向距离直方图 bin 数。
	radialBins = 16
	// scAngularBins shape-context 角度 bin 数。
	scAngularBins = 5
	// scRadialBins shape-context 径向 bin 数（对数）。
	scRadialBins = 12
	// scBins 单点 shape-context 直方图总 bin 数。
	scBins = scAngularBins * scRadialBins
	// lbpDims uniform LBP(8,1) 直方图维度（58 个 uniform 模式 + 1 个"非 uniform"桶）。
	lbpDims = 59
	// normSize LBP/SC 计算用的归一化边长。
	normSize = 64
	// occBins 占据栅格边长常量。
	occBin16 = 16
	occBin32 = 32
)
