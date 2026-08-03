// Package segment: shape.go
//
// 反转不变的结构哈希（polarity-canonical structural hash）。
//
// 许多检索用例是原图的"局部视图"，且背景/前景极性被翻转：例如原图是
// 黑底上的深灰网格，而查询图是白底上的黑色网格；或原图图标为透明底（在
// 灰度哈希里渲染为白底），查询图叠加在不透明深色底上。普通的 Otsu 二值
// 结构哈希对这类极性翻转极不稳健——前景/背景互换会让 DCT 交流系数整体
// 反号，使感知哈希近似取反、汉明距离接近满值。
//
// 解决办法：对 Otsu 二值掩码做"极性归一化"——当前景（值 0）像素占多数
// 时反转掩码，使前景恒为少数像素。这样一幅图与它的反相图（负片）会归
// 一化到同一张掩码，因而得到相同的结构哈希，使结构匹配对极性翻转不变。
// 仅在前景恰好占 50% 处存在测度零的不连续，实际图像几乎不触发。
//
// 该哈希用于 RegionInfo.Shape / MergedRegion.Shape，使颜色变化或背景反相
// 时仍可按稳定结构召回。
package segment

import (
	"image"

	"go-image-search/internal/phash"
)

// ShapeHash 计算图像的反相不变结构哈希。
//
// 流程：Otsu 二值化 → 极性归一化（前景占多数时反转掩码，使前景恒为少数）
// → 64-bit 感知哈希。归一化保证图与其负片哈希一致，从而对背景/前景极性
// 翻转稳健，适配"原图深色底 vs 查询图浅色底"的局部视图检索。
//
// 返回 0 表示无法计算（空图像等）。
func ShapeHash(src image.Image) uint64 {
	mask := structuralMask(src)
	if mask == nil {
		return 0
	}
	canonicalizePolarity(mask)
	return phash.Hash(mask)
}

// canonicalizePolarity 就地极性归一化掩码：当前景（值 0）像素严格过半时
// 反相，使前景恒为少数。归一化是 involutive 的——对 M 与 ~M 产生同一结果，
// 因此 phash.Hash(canonicalize(M)) == phash.Hash(canonicalize(~M))，
// 即图与其负片结构哈希相同。
func canonicalizePolarity(g *image.Gray) {
	if g == nil || len(g.Pix) == 0 {
		return
	}
	fg := 0
	for _, v := range g.Pix {
		if v == 0 {
			fg++
		}
	}
	if fg*2 > len(g.Pix) {
		for i := range g.Pix {
			g.Pix[i] = 255 - g.Pix[i]
		}
	}
}
