// Package segment 提供图像区域划分能力，分为两层：
//   - 算法层（color.go / merge.go / gravity.go / common.go）：像素颜色连通区域划分、
//     相似区域合并、引力聚合及共享工具；
//   - 策略层（strategy.go / builtin.go / run.go）：Strategy 接口 + 内置策略 + Pipeline，
//     统一入口 Run / RunDefault 支持可插拔、可组合的区域划分策略。
package segment

// color.go 像素颜色连通区域划分：基于颜色相似性与连通性把图像划分为连通区域，
// 并为区域去噪、裁剪提供支持。对应的策略见 builtin.go 的 ColorSegment。


import (
	"image"
	"image/color"
	"sort"
)

// Region 表示一个连通区域。
type Region struct {
	ID        int
	BBox      image.Rectangle
	Area      int
	MeanColor color.RGBA
}

// Result 保存整幅图像的区域划分结果。
type Result struct {
	Width   int
	Height  int
	Labels  []int32 // 每个像素的区域标签，-1 表示背景/噪声
	Regions []*Region
}

// Config 区域划分参数。
type Config struct {
	ThresholdFactor float64 // 自适应阈值 = 相邻色差分布的该分位数 × 因子
	ThresholdPct    float64 // 相邻色差分位数（0~1）
	MinThreshold    float64 // 阈值下限
	MaxThreshold    float64 // 阈值上限
	MinAreaRatio    float64 // 面积小于 该比例×总像素 的区域视为噪声丢弃
	MedianFilterK   int     // 预处理中值滤波核大小（0 关闭）
	Connectivity    int     // 4 或 8
	AlphaMin        uint8   // alpha 低于该值视为透明背景，不参与划分
}

// DefaultConfig 返回推荐默认参数。
func DefaultConfig() Config {
	return Config{
		ThresholdFactor: 1.0,
		ThresholdPct:    0.85,
		MinThreshold:    4.0,
		MaxThreshold:    60.0,
		MinAreaRatio:    0.0006,
		MedianFilterK:   3,
		Connectivity:    4,
		AlphaMin:        128,
	}
}

type pixel struct {
	x, y int
}

// Segment 对图像执行区域划分。
func Segment(src image.Image, cfg Config) (*Result, error) {
	if cfg.ThresholdPct <= 0 {
		cfg.ThresholdPct = 0.85
	}
	if cfg.MinThreshold <= 0 {
		cfg.MinThreshold = 4
	}
	if cfg.MaxThreshold <= 0 {
		cfg.MaxThreshold = 60
	}
	if cfg.ThresholdFactor <= 0 {
		cfg.ThresholdFactor = 1
	}
	if cfg.Connectivity != 8 {
		cfg.Connectivity = 4
	}

	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	rgba := toRGBA(src, b)

	if cfg.MedianFilterK > 1 {
		rgba = medianFilterRGBA(rgba, w, h, cfg.MedianFilterK)
	}

	idx := func(x, y int) int { return y*w + x }

	labels := make([]int32, w*h)
	for i := range labels {
		labels[i] = -1
	}

	// 自适应阈值：采样相邻像素色差分布
	thr := adaptiveThreshold(rgba, w, h, cfg)

	visited := make([]bool, w*h)
	var regions []*Region
	queue := make([]pixel, 0, 256)

	dir4 := [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	dir8 := [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}
	dirs := dir4
	if cfg.Connectivity == 8 {
		dirs = dir8
	}

	label := int32(0)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := idx(x, y)
			if labels[i] != -1 || visited[i] {
				continue
			}
			if rgba[i].A < cfg.AlphaMin {
				visited[i] = true
				continue
			}

			// 新区域 BFS
			label++
			queue = queue[:0]
			queue = append(queue, pixel{x, y})
			visited[i] = true
			labels[i] = label

			var sumR, sumG, sumB int64
			var cnt int64
			minX, minY, maxX, maxY := x, y, x, y

			for len(queue) > 0 {
				p := queue[len(queue)-1]
				queue = queue[:len(queue)-1]

				c := rgba[idx(p.x, p.y)]
				sumR += int64(c.R)
				sumG += int64(c.G)
				sumB += int64(c.B)
				cnt++
				if p.x < minX {
					minX = p.x
				}
				if p.x > maxX {
					maxX = p.x
				}
				if p.y < minY {
					minY = p.y
				}
				if p.y > maxY {
					maxY = p.y
				}

				for _, d := range dirs {
					nx, ny := p.x+d[0], p.y+d[1]
					if nx < 0 || nx >= w || ny < 0 || ny >= h {
						continue
					}
					ni := idx(nx, ny)
					if labels[ni] != -1 || visited[ni] {
						continue
					}
					nc := rgba[ni]
					if nc.A < cfg.AlphaMin {
						visited[ni] = true
						continue
					}
					// 与区域当前均值比较
					if colorDist(c, nc) > thr {
						continue
					}
					visited[ni] = true
					labels[ni] = label
					queue = append(queue, pixel{nx, ny})
				}
			}

			reg := &Region{
				ID:        int(label),
				BBox:      image.Rect(minX, minY, maxX+1, maxY+1),
				Area:      int(cnt),
				MeanColor: color.RGBA{R: uint8(sumR / cnt), G: uint8(sumG / cnt), B: uint8(sumB / cnt), A: 255},
			}
			regions = append(regions, reg)
		}
	}

	// 去噪：丢弃小区域
	minArea := int(cfg.MinAreaRatio * float64(w*h))
	if minArea < 1 {
		minArea = 1
	}
	kept := regions[:0]
	remap := make(map[int]int32)
	for _, reg := range regions {
		if reg.Area >= minArea {
			newID := int32(len(kept) + 1)
			remap[reg.ID] = newID
			reg.ID = int(newID)
			kept = append(kept, reg)
		} else {
			remap[reg.ID] = -1
		}
	}

	// 重写标签
	for i := range labels {
		if labels[i] != -1 {
			labels[i] = remap[int(labels[i])]
		}
	}

	return &Result{Width: w, Height: h, Labels: labels, Regions: kept}, nil
}

// Crop 从原图裁剪某个区域的 bbox 内容。
func (r *Result) Crop(src image.Image, id int) *image.RGBA {
	for _, reg := range r.Regions {
		if reg.ID == id {
			b := reg.BBox.Intersect(src.Bounds())
			out := image.NewRGBA(b)
			for y := b.Min.Y; y < b.Max.Y; y++ {
				for x := b.Min.X; x < b.Max.X; x++ {
					out.Set(x, y, src.At(x, y))
				}
			}
			return out
		}
	}
	return nil
}

// adaptiveThreshold 计算相邻像素色差分位点，作为合并阈值。
func adaptiveThreshold(rgba []color.RGBA, w, h int, cfg Config) float64 {
	idx := func(x, y int) int { return y*w + x }
	dists := make([]float64, 0, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := idx(x, y)
			if rgba[i].A < cfg.AlphaMin {
				continue
			}
			if x+1 < w {
				if j := idx(x+1, y); rgba[j].A >= cfg.AlphaMin {
					dists = append(dists, colorDist(rgba[i], rgba[j]))
				}
			}
			if y+1 < h {
				if j := idx(x, y+1); rgba[j].A >= cfg.AlphaMin {
					dists = append(dists, colorDist(rgba[i], rgba[j]))
				}
			}
		}
	}
	if len(dists) == 0 {
		return 16
	}
	sort.Float64s(dists)
	pos := int(cfg.ThresholdPct * float64(len(dists)-1))
	thr := dists[pos] * cfg.ThresholdFactor
	if thr < cfg.MinThreshold {
		thr = cfg.MinThreshold
	}
	if thr > cfg.MaxThreshold {
		thr = cfg.MaxThreshold
	}
	return thr
}

// colorDist 计算两个像素的 RGB 欧氏距离。
func colorDist(a, b color.RGBA) float64 {
	dr := float64(a.R) - float64(b.R)
	dg := float64(a.G) - float64(b.G)
	db := float64(a.B) - float64(b.B)
	return dr*dr + dg*dg + db*db
}

func toRGBA(src image.Image, b image.Rectangle) []color.RGBA {
	w, h := b.Dx(), b.Dy()
	out := make([]color.RGBA, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, bb, a := src.At(b.Min.X+x, b.Min.Y+y).RGBA()
			out[y*w+x] = color.RGBA{
				R: uint8(r >> 8),
				G: uint8(g >> 8),
				B: uint8(bb >> 8),
				A: uint8(a >> 8),
			}
		}
	}
	return out
}

func medianFilterRGBA(src []color.RGBA, w, h, k int) []color.RGBA {
	if k < 3 {
		return src
	}
	k |= 1
	half := k / 2
	dst := make([]color.RGBA, len(src))
	rs := make([]uint8, 0, k*k)
	gs := make([]uint8, 0, k*k)
	bs := make([]uint8, 0, k*k)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			rs = rs[:0]
			gs = gs[:0]
			bs = bs[:0]
			for dy := -half; dy <= half; dy++ {
				for dx := -half; dx <= half; dx++ {
					cx, cy := x+dx, y+dy
					if cx < 0 || cx >= w || cy < 0 || cy >= h {
						continue
					}
					c := src[cy*w+cx]
					rs = append(rs, c.R)
					gs = append(gs, c.G)
					bs = append(bs, c.B)
				}
			}
			dst[y*w+x] = color.RGBA{
				R: medianU8(rs),
				G: medianU8(gs),
				B: medianU8(bs),
				A: src[y*w+x].A,
			}
		}
	}
	return dst
}

func medianU8(v []uint8) uint8 {
	for i := 1; i < len(v); i++ {
		for j := i; j > 0 && v[j] < v[j-1]; j-- {
			v[j], v[j-1] = v[j-1], v[j]
		}
	}
	return v[len(v)/2]
}
