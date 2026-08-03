// 查询与索引的衍生表示生成：把形状压缩为单像素骨架，保留拓扑特征。
// 对原图生成：灰度化 → Otsu 二值化 → Zhang-Suen 细化，得到纯形状的骨架图。
// 颜色信息完全丢失，只描述形状拓扑。
package imageproc

import (
	"image"
	"image/color"
)

// SkeletonOptions 配置骨架生成。
type SkeletonOptions struct {
	Blur   bool // 是否在二值化前做 3×3 中值滤波去噪
	Invert bool // 是否反转前景/背景（false 取暗像素为前景）
}

// DefaultSkeletonOptions 返回默认骨架参数。
func DefaultSkeletonOptions() SkeletonOptions {
	return SkeletonOptions{Blur: true, Invert: false}
}

// QueryVariants 将源图像变换为衍生图列表，第一张始终为原图，
// 第二张为 Zhang-Suen 骨架图（纯形状描述）。
func QueryVariants(src image.Image) []image.Image {
	return QueryVariantsOpts(src, DefaultSkeletonOptions())
}

// QueryVariantsOpts 使用自定义参数生成衍生图。
func QueryVariantsOpts(src image.Image, o SkeletonOptions) []image.Image {
	out := []image.Image{src}
	if sk := Skeleton(src, o); sk != nil {
		out = append(out, sk)
	}
	return out
}

// Skeleton 生成图像的 Zhang-Suen 骨架图：
// 灰度化 → （可选中值滤波）→ Otsu 二值化 → Zhang-Suen 细化。
// 返回 *image.Gray，前景（骨架）为黑色，背景为白色。
func Skeleton(src image.Image, o SkeletonOptions) *image.Gray {
	g := Grayscale(src)
	if o.Blur {
		g = MedianFilter(g, 3)
	}
	bin := otsuBinarize(g, o)
	if bin == nil {
		return nil
	}
	b := bin.Bounds()
	fg := make([][]bool, b.Dx())
	for x := range fg {
		fg[x] = make([]bool, b.Dy())
	}
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			fg[x][y] = bin.GrayAt(b.Min.X+x, b.Min.Y+y).Y != 0
		}
	}
	zhangSuen(fg, b.Dx(), b.Dy())
	out := image.NewGray(b)
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			v := byte(255)
			if fg[x][y] {
				v = 0
			}
			out.SetGray(b.Min.X+x, b.Min.Y+y, color.Gray{Y: v})
		}
	}
	return out
}

// otsuBinarize 对灰度图做 Otsu 二值化。
// 返回 *image.Gray：前景为 0（黑）、背景为 255；空壳返回 nil。
func otsuBinarize(g *image.Gray, o SkeletonOptions) *image.Gray {
	b := g.Bounds()
	if b.Empty() {
		return nil
	}
	thr := otsuChoose(g.Pix)
	out := image.NewGray(b)
	for i, v := range g.Pix {
		if o.Invert {
			if v > thr {
				out.Pix[i] = 0
			} else {
				out.Pix[i] = 255
			}
		} else {
			if v < thr {
				out.Pix[i] = 0
			} else {
				out.Pix[i] = 255
			}
		}
	}
	return out
}

// StructuralMask 返回图像的颜色无关结构掩码：灰度化 → 3×3 均值平滑 → Otsu 二值化，
// 前景（暗）为 0，背景为 255。它丢弃颜色/亮度信息，只保留明暗分层的形状，
// 用于“颜色变化但形状稳定”的场景下作为不变量特征。空输入返回 nil。
//
// 二值前做 3×3 均值平滑：抑制裁剪边界/抗锯齿/纹理细节引入的逐像素噪声，
// 使同一图标的局部视图（裁剪+缩放）与原图分块得到稳定一致的二值结构，
// 让结构哈希（Shape）对裁剪/缩放稳健——这是局部视图检索召回的关键。
// 与 segment.structuralMask 语义保持一致（共享同一平滑策略）。
func StructuralMask(src image.Image) *image.Gray {
	return otsuBinarize(blurGray3(Grayscale(src)), DefaultSkeletonOptions())
}

// blurGray3 对灰度图做 3×3 均值平滑（边界采用镜像延拓），返回新图像。
// 用于 StructuralMask 二值化前抑制逐像素噪声，稳定结构哈希。
func blurGray3(src *image.Gray) *image.Gray {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	if w < 3 || h < 3 {
		return src
	}
	stride := w
	dst := image.NewGray(b)
	at := func(x, y int) int {
		if x < 0 {
			x = -x
		}
		if x >= w {
			x = 2*w - x - 2
			if x < 0 {
				x = 0
			}
		}
		if y < 0 {
			y = -y
		}
		if y >= h {
			y = 2*h - y - 2
			if y < 0 {
				y = 0
			}
		}
		return int(src.Pix[y*stride+x])
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			s := 0
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					s += at(x+dx, y+dy)
				}
			}
			dst.Pix[y*stride+x] = uint8((s + 4) / 9) // 四舍五入
		}
	}
	return dst
}

// otsuChoose 计算 Otsu 阈值。
func otsuChoose(src []uint8) uint8 {
	hist := make([]int, 256)
	for _, v := range src {
		hist[v]++
	}
	total := len(src)
	if total == 0 {
		return 0
	}
	sum := 0
	for i, n := range hist {
		sum += i * n
	}
	sumB, wB := 0, 0
	var bestThr uint8
	bestVar := float64(-1)
	for t := 0; t < 256; t++ {
		wB += hist[t]
		if wB == 0 {
			continue
		}
		wF := total - wB
		if wF == 0 {
			break
		}
		sumB += t * hist[t]
		mB := float64(sumB) / float64(wB)
		mF := float64(sum-sumB) / float64(wF)
		between := float64(wB) * float64(wF) * (mB - mF) * (mB - mF)
		if between >= bestVar {
			bestVar = between
			bestThr = uint8(t)
		}
	}
	return bestThr
}

// zhangSuen 就地提取骨架，fg 为前景（true 保留）。
func zhangSuen(fg [][]bool, w, h int) {
	for {
		var step1 [][2]int
		for y := 1; y < h-1; y++ {
			for x := 1; x < w-1; x++ {
				if !fg[x][y] {
					continue
				}
				if zsDel(fg, x, y, true) {
					step1 = append(step1, [2]int{x, y})
				}
			}
		}
		if len(step1) == 0 {
			break
		}
		for _, p := range step1 {
			fg[p[0]][p[1]] = false
		}

		var step2 [][2]int
		for y := 1; y < h-1; y++ {
			for x := 1; x < w-1; x++ {
				if !fg[x][y] {
					continue
				}
				if zsDel(fg, x, y, false) {
					step2 = append(step2, [2]int{x, y})
				}
			}
		}
		if len(step2) == 0 {
			break
		}
		for _, p := range step2 {
			fg[p[0]][p[1]] = false
		}
	}
}

// zsNeighbors 取 Zhang-Suen 的 8 邻域，顺序为 p2..p9（顺时针，从 12 点起）。
func zsNeighbors(fg [][]bool, x, y int) []bool {
	return []bool{
		fg[x][y-1],     // p2 北
		fg[x+1][y-1],   // p3 东北
		fg[x+1][y],     // p4 东
		fg[x+1][y+1],   // p5 东南
		fg[x][y+1],     // p6 南
		fg[x-1][y+1],   // p7 西南
		fg[x-1][y],     // p8 西
		fg[x-1][y-1],   // p9 西北
	}
}

// zsDel 判断像素 (x,y) 在 Zhang-Suen 一次迭代中是否应被删除。
func zsDel(fg [][]bool, x, y int, step1 bool) bool {
	n := zsNeighbors(fg, x, y)
	b := 0
	for _, v := range n {
		if v {
			b++
		}
	}
	if b < 2 || b > 6 {
		return false
	}
	// A(p1)=1：变量 p2..p9 中 0→1 跳变恰好一次
	a := 0
	for i := 0; i < 8; i++ {
		j := (i + 1) % 8
		if !n[i] && n[j] {
			a++
		}
	}
	if a != 1 {
		return false
	}
	p2, p3, p4, p5 := n[0], n[1], n[2], n[3]
	p6, p7, p8, p9 := n[4], n[5], n[6], n[7]
	_ = p3
	_ = p5
	_ = p7
	_ = p9
	if step1 {
		// 东北东南西南西北
		if !(p2 && p4 && p6) && !(p4 && p6 && p8) {
			return true
		}
	} else {
		if !(p2 && p4 && p8) && !(p2 && p6 && p8) {
			return true
		}
	}
	return false
}