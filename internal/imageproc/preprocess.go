// 查询侧截图归一化预处理。
//
// 问题背景：查询端经常收到带"统一背景色 + 图标 + 底部多余文字带"的截图，而图库
// 图标是透明底、只含图标本身。低分辨率截图的区域划分会碎裂，深色/彩色背景和多余
// 文字带会污染整图哈希，使其与图库透明底图标无法匹配。
//
// 方案：把截图还原为与图库一致的"白底 + 主内容"表示 —— 检测统一背景色并泛洪填充
// 为白，再把内容按行带切分、保留主体内容带、丢弃被空隙分隔的附属内容（如文字带）。
// 该衍生图只追加到查询侧（图库索引仍是干净的透明底图标，无需归一化）。
package imageproc

import (
	"image"
	"image/color"
)

// 归一化触发与内容带选择阈值。
const (
	minBorderCoverage = 0.5  // 边框主色覆盖率下限（相对全部边框采样），防止把图标误当背景
	bgColorDist       = 30   // 泛洪填充背景的颜色欧氏距离（0~255 空间）
	maxBgLum          = 240  // 近白背景无需归一化（图库透明底本身按白渲染）
	minGapFraction    = 0.12 // 内容带之间的空带至少占整图该比例才视为分隔
	minThinFraction   = 0.03 // 低于该比例高度的零散行带视为噪声，丢弃
	minBandFraction   = 0.10 // 主内容带高度至少占整图该比例，否则视为无有效内容
	minBandRatio      = 2.0  // 主带内容量需为其余各带最大者的该倍数，否则视为无明确主体
	bgNearWhiteLum    = 245  // 行分析中亮度低于该值才算前景像素
)

// QueryNormalizedVariant 检测并生成归一化裁剪衍生图；任一触发条件不满足时返回 nil。
// 触发条件：统一非近白背景（覆盖率 ≥ minBorderCoverage）且归一化后存在 ≥2 个被
// 显著空带分隔的内容带（即可丢弃的附属内容）。
func QueryNormalizedVariant(src image.Image) image.Image {
	bg, cov, ok := dominantBorderColor(src)
	if !ok || cov < minBorderCoverage || luminance(bg) >= maxBgLum {
		return nil
	}
	clean := removeBackground(src, bg)
	bands := splitContentBands(clean)
	if len(bands) < 2 {
		return nil
	}
	main, ok := pickMainBand(bands)
	if !ok {
		return nil
	}
	return cropBand(clean, main)
}

// contentBand 表示一个内容行带（半开区间 [y0,y1)），count 为带内前景像素数。
type contentBand struct {
	y0, y1, count int
}

// dominantBorderColor 统计边框像素（量化后的主色），返回主色、覆盖率（相对全部
// 边框采样，含透明像素）以及是否存在不透明边框。透明底图像覆盖率自然偏低。
func dominantBorderColor(src image.Image) (color.RGBA, float64, bool) {
	b := src.Bounds()
	count := make(map[uint32]int)
	sampled := 0
	add := func(x, y int) {
		r, g, bb, a := src.At(x, y).RGBA()
		sampled++
		if a>>8 == 0 {
			return
		}
		key := (uint32(r>>11) << 6) | (uint32(g>>11) << 3) | uint32(bb>>11)
		count[key]++
	}
	for x := b.Min.X; x < b.Max.X; x++ {
		add(x, b.Min.Y)
		add(x, b.Max.Y-1)
	}
	for y := b.Min.Y; y < b.Max.Y; y++ {
		add(b.Min.X, y)
		add(b.Max.X-1, y)
	}
	if sampled == 0 {
		return color.RGBA{}, 0, false
	}
	bestKey, bestN := uint32(0), 0
	for k, n := range count {
		if n > bestN {
			bestKey, bestN = k, n
		}
	}
	to8 := func(v uint32) uint8 { return uint8((v*255 + 15) / 31) }
	return color.RGBA{
		R: to8(bestKey >> 6),
		G: to8((bestKey >> 3) & 0x7),
		B: to8(bestKey & 0x7),
		A: 255,
	}, float64(bestN) / float64(sampled), true
}

// removeBackground 将图片中与边框连通、且颜色接近背景色的像素泛洪填充为白色。
// 非背景内容（图标/文字）颜色与背景差异大，不会被误填充。
func removeBackground(src image.Image, bg color.RGBA) image.Image {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return src
	}
	out := image.NewRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			out.Set(x, y, src.At(x, y))
		}
	}
	closeTo := func(c color.RGBA) bool {
		dr := float64(c.R) - float64(bg.R)
		dg := float64(c.G) - float64(bg.G)
		db := float64(c.B) - float64(bg.B)
		return dr*dr+dg*dg+db*db <= bgColorDist*bgColorDist
	}
	vis := make([]bool, w*h)
	stack := make([]int, 0, 64)
	push := func(x, y int) {
		if x < 0 || x >= w || y < 0 || y >= h {
			return
		}
		ni := y*w + x
		if vis[ni] {
			return
		}
		c := color.RGBAModel.Convert(out.At(b.Min.X+x, b.Min.Y+y)).(color.RGBA)
		if c.A >= 128 && closeTo(c) {
			vis[ni] = true
			stack = append(stack, ni)
		}
	}
	for x := 0; x < w; x++ {
		push(x, 0)
		push(x, h-1)
	}
	for y := 0; y < h; y++ {
		push(0, y)
		push(w-1, y)
	}
	for len(stack) > 0 {
		p := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		x, y := p%w, p/w
		out.Set(b.Min.X+x, b.Min.Y+y, color.RGBA{255, 255, 255, 255})
		push(x+1, y)
		push(x-1, y)
		push(x, y+1)
		push(x, y-1)
	}
	return out
}

// splitContentBands 在背景已归白的图像上，把内容行按空带切分为若干内容带：
// 先过滤太薄的噪声行带，再把被不足比例的空带分隔的行合并为同一带。
func splitContentBands(img image.Image) []contentBand {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return nil
	}
	rowFG := make([]int, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, bb, a := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			if a>>8 == 0 {
				continue
			}
			l := int(0.299*float64(r>>8) + 0.587*float64(g>>8) + 0.114*float64(bb>>8))
			if l < bgNearWhiteLum {
				rowFG[y]++
			}
		}
	}
	raw := make([]contentBand, 0, 4)
	start := -1
	acc := 0
	for y := 0; y < h; y++ {
		if rowFG[y] > 0 {
			if start < 0 {
				start = y
			}
			acc += rowFG[y]
		} else if start >= 0 {
			raw = append(raw, contentBand{y0: start, y1: y, count: acc})
			start = -1
			acc = 0
		}
	}
	if start >= 0 {
		raw = append(raw, contentBand{y0: start, y1: h, count: acc})
	}
	minThin := int(minThinFraction * float64(h))
	if minThin < 2 {
		minThin = 2
	}
	minGap := int(minGapFraction * float64(h))
	if minGap < 3 {
		minGap = 3
	}
	bands := make([]contentBand, 0, 4)
	for _, bd := range raw {
		if bd.y1-bd.y0 < minThin {
			continue
		}
		if n := len(bands); n > 0 && bd.y0-bands[n-1].y1 < minGap {
			bands[n-1].y1 = bd.y1
			bands[n-1].count += bd.count
		} else {
			bands = append(bands, bd)
		}
	}
	return bands
}

// pickMainBand 选出内容量最大的带作为主体；当主体内容量不足其余最大带的
// minBandRatio 倍时视为"无明确主体"，返回 false（避免猜测）。
func pickMainBand(bands []contentBand) (contentBand, bool) {
	if len(bands) < 2 {
		return contentBand{}, false
	}
	main := 0
	others := 0
	for i := range bands {
		if bands[i].count > bands[main].count {
			main = i
		}
	}
	for i := range bands {
		if i != main && bands[i].count > others {
			others = bands[i].count
		}
	}
	if float64(bands[main].count) < minBandRatio*float64(others) {
		return contentBand{}, false
	}
	return bands[main], true
}

// cropBand 把主内容带裁剪为"白底 + 居中内容"的归一化衍生图：按内容包围盒裁剪后，
// 四周补上白色边距，让内容浮于白底、与图库透明底图标同构。白色背景会聚合成完整的
// 边框区域，被 FilterFullFrame 过滤掉，因此整图辅助区域的包围盒就是内容本体，其
// 宽高比与图库一致。内容带高过小时返回 nil。
func cropBand(img image.Image, bd contentBand) image.Image {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return nil
	}
	contentH := bd.y1 - bd.y0
	if contentH < int(minBandFraction*float64(h)) {
		return nil
	}
	minX, maxX := w, -1
	for y := bd.y0; y < bd.y1; y++ {
		for x := 0; x < w; x++ {
			r, g, bb, a := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			if a>>8 == 0 {
				continue
			}
			l := int(0.299*float64(r>>8) + 0.587*float64(g>>8) + 0.114*float64(bb>>8))
			if l < bgNearWhiteLum {
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
			}
		}
	}
	if maxX < minX {
		return nil
	}
	contentW := maxX - minX + 1
	margin := int(0.08*float64(max(contentW, contentH)) + 0.5)
	if margin < 6 {
		margin = 6
	}
	canvas := image.Rect(0, 0, contentW+2*margin, contentH+2*margin)
	out := image.NewRGBA(canvas)
	white := color.RGBA{255, 255, 255, 255}
	for y := 0; y < canvas.Dy(); y++ {
		for x := 0; x < canvas.Dx(); x++ {
			out.Set(x, y, white)
		}
	}
	for y := bd.y0; y < bd.y1; y++ {
		for x := minX; x <= maxX; x++ {
			out.Set(x-minX+margin, y-bd.y0+margin, img.At(b.Min.X+x, b.Min.Y+y))
		}
	}
	return out
}

// luminance 计算颜色的感知亮度（Y 分量，0~255）。
func luminance(c color.RGBA) int {
	return int(0.299*float64(c.R) + 0.587*float64(c.G) + 0.114*float64(c.B))
}
