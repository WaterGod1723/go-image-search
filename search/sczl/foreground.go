package sczl

import (
	"image"
	"image/color"
	"math"
	"sort"
)

// foreground.go 前景提取：从截图分离 icon 主体，剔除背景/四周文字/填充区域。
// 思路：边框种子泛洪填充背景（颜色无关，纯色背景零影响）→ 形态学闭运算
// 桥接反锯齿间隙 → 连通域 → 丢弃小碎片（文字/线条干扰）→ 取主体区域。
// 不依赖颜色分割，对"背景色变化"鲁棒；不依赖颜色，对"icon 颜色变化"鲁棒。

// fgConfig 前景提取参数。
type fgConfig struct {
	bgTol     float64 // 泛洪背景颜色容差（0~255 欧氏距离，仅在不透明截图路径生效）
	noiseFrac float64 // 噪声碎片阈值（相对图像面积），丢弃小于此面积的连通域
}

func defaultFGConfig() fgConfig {
	return fgConfig{bgTol: 36, noiseFrac: 0.0008}
}

// fgMask 二值前景掩码：true 为前景。
type fgMask struct {
	cell []bool
	w, h int
}

func newFGMask(w, h int) fgMask { return fgMask{cell: make([]bool, w*h), w: w, h: h} }
func (m fgMask) at(x, y int) bool {
	if x < 0 || x >= m.w || y < 0 || y >= m.h {
		return false
	}
	return m.cell[y*m.w+x]
}
func (m fgMask) set(x, y int, v bool) {
	if x < 0 || x >= m.w || y < 0 || y >= m.h {
		return
	}
	m.cell[y*m.w+x] = v
}

// extractForeground 返回主体前景掩码与外接矩形。
// 保留全部前景像素（线稿图标的笔画多互不连通，按"最大连通域"选取会丢失整体形状）；
// 噪声/文字带由调用方 trimTextBands 处理。若无可提取前景返回 ok=false。
func extractForeground(pix []colorRGBA, w, h int, cfg fgConfig) (fgMask, image.Rectangle, bool) {
	if w == 0 || h == 0 {
		return fgMask{}, image.Rectangle{}, false
	}

	var bgMask fgMask
	var alphaPath bool
	transp := 0
	for _, c := range pix {
		if c.A == 0 {
			transp++
		}
	}
	if transp > w*h/20 {
		// 透明 PNG（图库图标常见）：alpha 通道即天然掩码，
		// 不依赖颜色泛洪，从根本上避免"深色图标被深色背景吃掉"。
		bgMask = floodFillTransparent(pix, w, h)
		alphaPath = true
	} else {
		bg := dominantBorder(pix, w, h)
		tol := adaptiveBGTol(pix, w, h, bg)
		bgMask = floodFillBG(pix, w, h, bg, tol)
	}

	// 前景 = 非背景 且 非透明。
	fg := newFGMask(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			if pix[i].A == 0 {
				continue
			}
			if !bgMask.cell[i] {
				fg.cell[i] = true
			}
		}
	}

	if !alphaPath {
		// 形态学闭运算：仅对不透明截图（反锯齿/泛洪边界）做桥接；
		// 透明 PNG 的 alpha 掩码本就干净，做形态学会破坏线稿图标。
		fg = dilate(fg, 1)
		fg = erode(fg, 1)
	}

	// 丢弃绝对噪声碎片（< 噪声阈值），保留其余全部前景。
	// 噪声阈值取图像规模的固定小值，不按"最大连通域比例"——否则线稿会被全量丢弃。
	noiseArea := int(float64(w*h) * cfg.noiseFrac)
	if noiseArea < 6 {
		noiseArea = 6
	}
	fg = dropSmallComponents(fg, noiseArea)

	minX, minY, maxX, maxY := w, h, -1, -1
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if fg.cell[y*w+x] {
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	if maxX < 0 {
		return fgMask{}, image.Rectangle{}, false
	}
	return fg, image.Rect(minX, minY, maxX+1, maxY+1), true
}

// dropSmallComponents 丢弃面积小于 minArea 的连通域，保留其余像素。
func dropSmallComponents(m fgMask, minArea int) fgMask {
	if minArea <= 1 {
		return m
	}
	labels, comps := connectedComponents(m)
	keep := make([]bool, len(comps))
	for i, c := range comps {
		if c.area >= minArea {
			keep[i] = true
		}
	}
	out := newFGMask(m.w, m.h)
	for i, l := range labels {
		if l >= 0 && keep[l] {
			out.cell[i] = true
		}
	}
	return out
}

// floodFillTransparent 把所有透明像素标记为背景，其余为前景候选。
// 透明 PNG 不需要颜色泛洪，alpha 即天然掩码。
func floodFillTransparent(pix []colorRGBA, w, h int) fgMask {
	m := newFGMask(w, h)
	for i, c := range pix {
		if c.A == 0 {
			m.cell[i] = true
		}
	}
	return m
}

// adaptiveBGTol 由边框像素相对主色的离散度自适应背景容差：
// 均匀背景 → 小容差（保护深色图标不被深色背景吞没）；渐变/噪声背景 → 较大容差。
func adaptiveBGTol(pix []colorRGBA, w, h int, bg colorRGBA) float64 {
	devs := make([]float64, 0, 2*(w+h))
	for x := 0; x < w; x++ {
		devs = append(devs, colorEuclid(pix[0*w+x], bg))
		devs = append(devs, colorEuclid(pix[(h-1)*w+x], bg))
	}
	for y := 0; y < h; y++ {
		devs = append(devs, colorEuclid(pix[y*w+0], bg))
		devs = append(devs, colorEuclid(pix[y*w+(w-1)], bg))
	}
	if len(devs) == 0 {
		return 16
	}
	sort.Float64s(devs)
	p90 := devs[(len(devs)-1)*9/10]
	tol := 1.5*p90 + 4
	if tol < 6 {
		tol = 6
	}
	if tol > 48 {
		tol = 48
	}
	return tol
}

// colorRGBA 别名为标准 color.RGBA，仅为本包内统一命名。
type colorRGBA = color.RGBA

// trimTextBands 剔除与主体前景分离的附属文字/填充带：
// 按行前景密度把图像切成水平带，带间由空行（前景稀疏）分隔；
// 保留前景量最大的主带，丢弃其余带（通常是文字标签、装饰条）。
// 对 "四周文字" / "填充区域" 干扰鲁棒。返回裁剪后的掩码与 bbox。
func trimTextBands(fg fgMask, w, h int, bbox image.Rectangle) (fgMask, image.Rectangle) {
	if w == 0 || h == 0 {
		return fg, bbox
	}
	// 行前景计数。
	rowCnt := make([]int, h)
	for y := 0; y < h; y++ {
		c := 0
		for x := 0; x < w; x++ {
			if fg.cell[y*w+x] {
				c++
			}
		}
		rowCnt[y] = c
	}
	// "非空"行：前景像素数 ≥ 行宽的 3%（抑制反锯齿噪声）。
	nonEmptyThr := int(float64(w) * 0.03)
	if nonEmptyThr < 1 {
		nonEmptyThr = 1
	}
	nonEmpty := make([]bool, h)
	for y := 0; y < h; y++ {
		nonEmpty[y] = rowCnt[y] >= nonEmptyThr
	}
	// 切成连续带。
	type band struct{ y0, y1, fg int } // [y0,y1)
	var bands []band
	i := 0
	for i < h {
		if !nonEmpty[i] {
			i++
			continue
		}
		j := i
		for j < h && nonEmpty[j] {
			j++
		}
		fgSum := 0
		for k := i; k < j; k++ {
			fgSum += rowCnt[k]
		}
		bands = append(bands, band{i, j, fgSum})
		i = j
	}
	if len(bands) <= 1 {
		return fg, bbox
	}
	// 主带 = 前景量最大者。
	best := 0
	for k := 1; k < len(bands); k++ {
		if bands[k].fg > bands[best].fg {
			best = k
		}
	}
	mb := bands[best]
	// 主带过窄则放弃裁剪（避免误删主体）。
	if mb.y1-mb.y0 < h/10 {
		return fg, bbox
	}
	// 构造裁剪后掩码。
	out := newFGMask(w, h)
	for y := mb.y0; y < mb.y1; y++ {
		for x := 0; x < w; x++ {
			if fg.cell[y*w+x] {
				out.cell[y*w+x] = true
			}
		}
	}
	// 重算 bbox。
	minX, minY, maxX, maxY := w, h, -1, -1
	for y := mb.y0; y < mb.y1; y++ {
		for x := 0; x < w; x++ {
			if out.cell[y*w+x] {
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	if maxX < 0 {
		return fg, bbox
	}
	return out, image.Rect(minX, minY, maxX+1, maxY+1)
}

// dominantBorder 取四条边像素的主色作为背景色估计。
func dominantBorder(pix []colorRGBA, w, h int) colorRGBA {
	if w == 0 || h == 0 {
		return colorRGBA{A: 255}
	}
	type cnt struct {
		c colorRGBA
		n int
	}
	bucket := func(c colorRGBA) uint32 {
		// 量化到 4 位每通道，抑制轻微噪声。
		return uint32(c.R>>4)<<16 | uint32(c.G>>4)<<8 | uint32(c.B>>4)
	}
	hist := make(map[uint32]cnt)
	add := func(c colorRGBA) {
		if c.A == 0 {
			c = colorRGBA{R: 255, G: 255, B: 255, A: 255} // 透明视作白
		}
		k := bucket(c)
		v := hist[k]
		v.c = c
		v.n++
		hist[k] = v
	}
	for x := 0; x < w; x++ {
		add(pix[0*w+x])
		add(pix[(h-1)*w+x])
	}
	for y := 0; y < h; y++ {
		add(pix[y*w+0])
		add(pix[y*w+(w-1)])
	}
	best := colorRGBA{R: 255, G: 255, B: 255, A: 255}
	bestN := -1
	for _, v := range hist {
		if v.n > bestN {
			bestN = v.n
			best = v.c
		}
	}
	return best
}

// floodFillBG 从四条边的非透明像素起泛洪填充颜色相近的区域。
func floodFillBG(pix []colorRGBA, w, h int, bg colorRGBA, tol float64) fgMask {
	visited := make([]bool, w*h)
	queue := make([]int, 0, w+h)
	push := func(x, y int) {
		i := y*w + x
		if visited[i] {
			return
		}
		c := pix[i]
		if c.A == 0 {
			visited[i] = true
			queue = append(queue, i)
			return
		}
		if colorEuclid(c, bg) <= tol {
			visited[i] = true
			queue = append(queue, i)
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
	// BFS（4 连通）。
	head := 0
	for head < len(queue) {
		i := queue[head]
		head++
		x := i % w
		y := i / w
		// 4 邻
		if x > 0 {
			push(x-1, y)
		}
		if x+1 < w {
			push(x+1, y)
		}
		if y > 0 {
			push(x, y-1)
		}
		if y+1 < h {
			push(x, y+1)
		}
	}
	return fgMask{cell: visited, w: w, h: h}
}

// colorEuclid 定义在 imageutil.go。

// dilate 3×3 膨胀，iters 次。
func dilate(m fgMask, iters int) fgMask {
	cur := m
	for k := 0; k < iters; k++ {
		next := newFGMask(m.w, m.h)
		for y := 0; y < m.h; y++ {
			for x := 0; x < m.w; x++ {
				if cur.at(x, y) {
					continue
				}
				if cur.at(x-1, y) || cur.at(x+1, y) || cur.at(x, y-1) || cur.at(x, y+1) ||
					cur.at(x-1, y-1) || cur.at(x+1, y-1) || cur.at(x-1, y+1) || cur.at(x+1, y+1) {
					next.set(x, y, true)
				}
			}
		}
		// 保留原前景
		for i := range next.cell {
			if cur.cell[i] {
				next.cell[i] = true
			}
		}
		cur = next
	}
	return cur
}

// erode 3×3 腐蚀，iters 次。
func erode(m fgMask, iters int) fgMask {
	cur := m
	for k := 0; k < iters; k++ {
		next := newFGMask(m.w, m.h)
		for y := 0; y < m.h; y++ {
			for x := 0; x < m.w; x++ {
				if !cur.at(x, y) {
					continue
				}
				// 任一 4 邻为空则腐蚀
				if cur.at(x-1, y) && cur.at(x+1, y) && cur.at(x, y-1) && cur.at(x, y+1) {
					next.set(x, y, true)
				}
			}
		}
		cur = next
	}
	return cur
}

// component 一个连通域的元信息（不单独存像素集，按需重扫 labels）。
type component struct {
	area int
	bbox image.Rectangle
}

// connectedComponents 4 连通连通域标记。返回标签图（-1=无）与各域元信息。
// 像素集合不单独保存：调用方按需重扫 labels 复用即可，避免大图内存膨胀。
func connectedComponents(m fgMask) ([]int, []component) {
	w, h := m.w, m.h
	labels := make([]int, w*h)
	for i := range labels {
		labels[i] = -1
	}
	var comps []component
	dir4 := [4][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	queue := make([][2]int, 0, 256)
	label := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			if !m.cell[i] || labels[i] != -1 {
				continue
			}
			queue = queue[:0]
			queue = append(queue, [2]int{x, y})
			labels[i] = label
			minX, minY, maxX, maxY := x, y, x, y
			area := 0
			head := 0
			for head < len(queue) {
				p := queue[head]
				head++
				area++
				if p[0] < minX {
					minX = p[0]
				}
				if p[0] > maxX {
					maxX = p[0]
				}
				if p[1] < minY {
					minY = p[1]
				}
				if p[1] > maxY {
					maxY = p[1]
				}
				for _, d := range dir4 {
					nx, ny := p[0]+d[0], p[1]+d[1]
					if nx < 0 || nx >= w || ny < 0 || ny >= h {
						continue
					}
					ni := ny*w + nx
					if !m.cell[ni] || labels[ni] != -1 {
						continue
					}
					labels[ni] = label
					queue = append(queue, [2]int{nx, ny})
				}
			}
			comps = append(comps, component{
				area: area,
				bbox: image.Rect(minX, minY, maxX+1, maxY+1),
			})
			label++
		}
	}
	return labels, comps
}

// fgCentroid 返回前景像素的质心（像素中心坐标系，即 (x+0.5, y+0.5)）。
// 与 Rmax 计算、轮廓射线追踪使用的同一坐标系，避免像素左上角/中心的 0.5 偏差
// 导致归一化后轮廓点超出 [0,1]^2。若前景为空返回 (0,0)。
func fgCentroid(m fgMask) (cx, cy float64) {
	var sx, sy float64
	n := 0
	for y := 0; y < m.h; y++ {
		for x := 0; x < m.w; x++ {
			if m.cell[y*m.w+x] {
				sx += float64(x) + 0.5
				sy += float64(y) + 0.5
				n++
			}
		}
	}
	if n == 0 {
		return 0, 0
	}
	return sx / float64(n), sy / float64(n)
}

// radialPercentile 返回前景像素到质心距离的 p 分位数（像素单位）。
// 用于构建旋转不变的归一化尺度基准（如 R98：98% 像素到此线内）。
func radialPercentile(m fgMask, cx, cy float64, p float64) float64 {
	if p <= 0 {
		p = 0.98
	}
	rs := make([]float64, 0, 256)
	for y := 0; y < m.h; y++ {
		for x := 0; x < m.w; x++ {
			if m.cell[y*m.w+x] {
				dx, dy := float64(x)-cx, float64(y)-cy
				rs = append(rs, math.Sqrt(dx*dx+dy*dy))
			}
		}
	}
	if len(rs) == 0 {
		return 0
	}
	sort.Float64s(rs)
	idx := int(float64(len(rs)-1) * p)
	if idx < 0 {
		idx = 0
	}
	if idx >= len(rs) {
		idx = len(rs) - 1
	}
	return rs[idx]
}

// makeNormFrame 从前景掩码构建归一化框架：质心 + Rmax*2.02 正方形。
// Rmax = 前景像素中心到质心的最大径向距离（像素中心按 (x+0.5, y+0.5) 计算，与射线追踪的子像素精度对齐）。
// 边长 = 2*Rmax*1.01 = Rmax*2.02，给旋转/插值/像素中心偏差留 1% 边距，保证最外围的像素
// 无论朝哪个方向都完整落入正方形内。
//
// 为何不用 bbox 或 R98：
//   - bbox（min/max）随旋转扭曲（长方形变斜 → 宽高变），非旋转不变；
//   - R98 对实心紧凑图（如 home_work 带一个大"圆"）可能把轮廓内凹当噪声丢掉，
//     导致 query/lib 尺度错位（尤其一张带边框另一张没有）。
//
// Rmax 虽然对单个离群噪点敏感，但 trimTextBands/连通域阈值已在 fg 阶段清走噪点。
func makeNormFrame(fg fgMask) (frame normFrame, ok bool) {
	cx, cy := fgCentroid(fg)
	rmax := 0.0
	n := 0
	for y := 0; y < fg.h; y++ {
		for x := 0; x < fg.w; x++ {
			if fg.cell[y*fg.w+x] {
				n++
				// 像素中心坐标（x+0.5, y+0.5），与射线追踪的子像素 lastX/lastY 对齐。
				dx := (float64(x) + 0.5) - cx
				dy := (float64(y) + 0.5) - cy
				d := math.Sqrt(dx*dx + dy*dy)
				if d > rmax {
					rmax = d
				}
			}
		}
	}
	if n == 0 || rmax <= 0 {
		return normFrame{}, false
	}
	frame.CX = cx
	frame.CY = cy
	// half = rmax*1.01 → scale = rmax*2.02。1% 应对射线 step=0.5 的最后一步超出（最多 0.5 像素级）。
	frame.Scale = rmax * 2.02
	return frame, true
}
