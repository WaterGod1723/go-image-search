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
	Radial  []float64 // 径向距离直方图（归一化），R 维

	// 局部形状签名。
	SC       [][]float64 // nPoints × nBins shape-context 直方图
	SCPoints []Point     // 采样轮廓点（归一化坐标）

	// NCC 归一化互相关模板：去均值 + L2 归一化的 64×64 灰度块。
	// 比二值化 DCT pHash 保留更多空间结构信息，且对亮度/对比度仿射变化不变
	// （query 截图 vs 图库原图的颜色差异不再破坏匹配）。
	Patch []float64

	// BBox 紧裁剪备份（非旋转精确匹配）：用轴对齐 bbox 紧裁剪到 64×64，
	// 内容填满整个栅格（无空白稀释），对非旋转变体匹配精度更高。
	// 搜索时 Occ/NCC 取 max(Rmax-旋转扫描, BBox-直接匹配)，
	// 旋转场景 Rmax 赢，非旋转场景 BBox 赢，无需决策树判断。
	OccBBox64 []float64 // bbox 紧裁剪 64×64 软占据（L2 归一化）
	PatchBBox []float64 // bbox 紧裁剪 NCC 模板（去均值+L2）

	// HOG 梯度方向直方图：捕捉边缘方向的空间分布，天然对填充色/背景色不变
	// （仅边界处梯度显著，填充色差异不影响）。替代脆弱的 LBP。
	HOG []float64

	// 掩码区域划分（多部件判别）：前景连通域逐个描述，
	// 检索时多区域匈牙利匹配，区分多部件图标与单块填充图标。
	Regions []RegionDesc

	// Solidity 实度 = 前景面积 / 凸包面积 [0,1]。低实度=镂空/碎片化轮廓，
	// 此时基于轮廓的 SC 信号不可靠，检索时对 SC 置零避免噪声拖分。
	Solidity float64

	// 辅助元信息。
	Area  int
	BBox  image.Rectangle
	Valid bool
}

// Match 图像级匹配结果。
type Match struct {
	ImageID string
	Score   float64 // [0,1] 越大越相似
	FDSim   float64 // 全局签名相似度
	SCSim   float64 // shape-context 相似度
	HOGSim  float64 // HOG 梯度方向相似度
}

// Options 检索参数。
type Options struct {
	TopK        int     // 返回数量，<=0 取 5
	OccWeight   float64 // 占据栅格权重，<=0 取 0.25
	NccWeight   float64 // NCC 互相关权重，<=0 取 0.25
	RegWeight   float64 // 区域匹配权重，<=0 取 0.15
	HogWeight   float64 // HOG 梯度方向权重，<=0 取 0.15
	SCWeight    float64 // SC 权重，<=0 取 0.10
	FDWeight    float64 // Fourier 权重，<=0 取 0.10
	FDPreFilter int     // 全局召回后保留的候选数，<=0 取 200
	FDCut       float64 // 全局相似度截断（低于此值直接淘汰），<=0 取 0.4
	SCMaxPoints int     // SC 匹配时每侧采样点数上限（控制匈牙利规模），<=0 取 48

	// RotSteps 旋转扫描步数（occ 软掩码 + SC 点云），均匀覆盖 [0, 2π)。
	// <=0 取 18（每 20°）；=1 时 ang=0 等效关闭旋转扫描（回退原旋转敏感行为）。
	// 旋转只发生在 query 侧（旋转其 occ64 / SCPoints），参考侧不动。
	// ang=0 那步即原匹配，故非旋转场景不会退化；旋转场景取最佳对齐恢复判别力。
	RotSteps int

	// 自适应权重：基于 query 在精排候选集上的各维度得分分布动态调整，
	// 而非写死 OccWeight..FDWeight。某维度"头部与主体分离越明显"
	// （top1 显著高于 top2..topK 均值）说明对当前 query 区分度越强，给更高权重。
	// final_w(d) = clip(α·prior_norm(d) + (1-α)·softmax(disc/T)(d), lo, hi)，再归一化。
	// 关闭（Adaptive=false）时退回 OccWeight..FDWeight 的固定先验权重。
	Adaptive      bool    // 是否启用自适应权重
	AdaptiveAlpha float64 // 先验权重占比，<=0 取 0.5
	AdaptiveTemp  float64 // softmax 温度（越大越平滑），<=0 取 1.0
	AdaptiveK     int     // 计算区分度时取前 K 个候选，<=0 取 10
	AdaptiveLo    float64 // 单维度权重下限，<=0 取 0.05
	AdaptiveHi    float64 // 单维度权重上限，<=0 取 0.5
}

// DefaultOptions 推荐默认参数。
func DefaultOptions() Options {
	return Options{
		TopK:          5,
		OccWeight:     0.20,
		NccWeight:     0.20,
		RegWeight:     0.20,
		HogWeight:     0.10,
		SCWeight:      0.10,
		FDWeight:      0.20,
		FDPreFilter:   200,
		FDCut:         0.0,
		SCMaxPoints:   48,
		RotSteps:      36,
		Adaptive:      true,
		AdaptiveAlpha: 0.5,
		AdaptiveTemp:  1.0,
		AdaptiveK:     10,
		AdaptiveLo:    0.05,
		AdaptiveHi:    0.5,
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
	// fillThr 实度低于此值时对占据栅格做孔洞填充（碎片化/镂空图标聚合）。
	fillThr = 0.5
)

// normFrame 旋转不变的归一化框架：质心为原点，scale 为尺度基准。
// 旋转后质心不变、距离不变 → frame 不变，使所有子描述子对旋转对齐一致。
type normFrame struct {
	CX, CY float64 // 前景质心（像素坐标）
	Scale  float64 // 归一化正方形边长（像素），R98*2.4 保险覆盖旋转后外接正方形
}
