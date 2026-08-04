package sczl

import (
	"image"
	"math"
)

// region.go 掩码区域划分 + 多区域描述。
// 与原方案的颜色连通区域不同，这里以前景掩码的连通域为"区域"：
// 颜色无关（免疫 icon 颜色变化/背景色变化），且天然刻画图标内部多部件结构
// （dashboard 的 4 个方块、天平的横梁与两盘、勾选框的边框与勾）。
// 检索时多区域匈牙利一对一匹配，部件数量与位置一致才得高分，
// 直接区分"多部件图标"与"单块填充图标"。

// RegionDesc 单个掩码区域的描述子。
type RegionDesc struct {
	Occ    []float64 // 8×8 软占据（L2 归一化），64 维
	Area   int
	NX, NY float64 // 区域质心相对 icon bbox 的归一化坐标 [0,1]
	Fill   float64 // 区域面积 / 其 bbox 面积
	Aspect float64 // 区域 bbox 宽/高
}

// regionOccBins 单区域占据栅格边长。
const regionOccBins = 8

// extractRegions 对前景掩码做连通域划分，返回每个有效区域的描述子。
// 噪声阈值相对最大连通域，避免大画布小线稿被全量丢弃；同时保留下限防止碎片噪声。
func extractRegions(fg fgMask, bbox image.Rectangle) []RegionDesc {
	labels, comps := connectedComponents(fg)
	if len(comps) == 0 {
		return nil
	}
	maxArea := 0
	for _, c := range comps {
		if c.area > maxArea {
			maxArea = c.area
		}
	}
	thr := int(0.08 * float64(maxArea))
	if thr < 6 {
		thr = 6
	}
	// 选中保留的标签。
	keep := make(map[int]bool, len(comps))
	for ci, c := range comps {
		if c.area >= thr {
			keep[ci] = true
		}
	}
	if len(keep) == 0 {
		keep[0] = true
	}
	// 单遍扫描累加各保留区域的质心。
	type acc struct{ sx, sy float64; n int }
	stats := make(map[int]acc, len(keep))
	for y := 0; y < fg.h; y++ {
		for x := 0; x < fg.w; x++ {
			l := labels[y*fg.w+x]
			if l < 0 || !keep[l] {
				continue
			}
			a := stats[l]
			a.sx += float64(x)
			a.sy += float64(y)
			a.n++
			stats[l] = a
		}
	}
	var regs []RegionDesc
	for ci, c := range comps {
		if !keep[ci] {
			continue
		}
		occ := regionOccupancy(fg, labels, ci, c.bbox, regionOccBins)
		st := stats[ci]
		var cx, cy float64
		if st.n > 0 {
			cx = st.sx / float64(st.n)
			cy = st.sy / float64(st.n)
		} else {
			cx = float64(c.bbox.Min.X+c.bbox.Max.X) / 2
			cy = float64(c.bbox.Min.Y+c.bbox.Max.Y) / 2
		}
		bw, bh := c.bbox.Dx(), c.bbox.Dy()
		fill := 0.0
		if bw > 0 && bh > 0 {
			fill = float64(c.area) / float64(bw*bh)
		}
		aspect := 1.0
		if bh > 0 {
			aspect = float64(bw) / float64(bh)
		}
		// 质心归一化到 icon bbox [0,1]。
		nx, ny := 0.5, 0.5
		if bbox.Dx() > 0 {
			nx = (cx - float64(bbox.Min.X)) / float64(bbox.Dx())
		}
		if bbox.Dy() > 0 {
			ny = (cy - float64(bbox.Min.Y)) / float64(bbox.Dy())
		}
		regs = append(regs, RegionDesc{
			Occ:    occ,
			Area:   c.area,
			NX:     nx,
			NY:     ny,
			Fill:   fill,
			Aspect: aspect,
		})
	}
	return regs
}

// regionOccupancy 计算某标签区域在其 bbox 内的 n×n 软占据（L2 归一化）。
func regionOccupancy(fg fgMask, labels []int, label int, bbox image.Rectangle, n int) []float64 {
	bw, bh := bbox.Dx(), bbox.Dy()
	out := make([]float64, n*n)
	if bw <= 0 || bh <= 0 {
		return out
	}
	for gy := 0; gy < n; gy++ {
		y0 := bbox.Min.Y + gy*bh/n
		y1 := bbox.Min.Y + (gy+1)*bh/n
		if y1 <= y0 {
			y1 = y0 + 1
		}
		if y1 > bbox.Min.Y+bh {
			y1 = bbox.Min.Y + bh
		}
		for gx := 0; gx < n; gx++ {
			x0 := bbox.Min.X + gx*bw/n
			x1 := bbox.Min.X + (gx+1)*bw/n
			if x1 <= x0 {
				x1 = x0 + 1
			}
			if x1 > bbox.Min.X+bw {
				x1 = bbox.Min.X + bw
			}
			cnt, tot := 0, 0
			for yy := y0; yy < y1; yy++ {
				for xx := x0; xx < x1; xx++ {
					tot++
					if labels[yy*fg.w+xx] == label {
						cnt++
					}
				}
			}
			if tot > 0 {
				out[gy*n+gx] = float64(cnt) / float64(tot)
			}
		}
	}
	norm := 0.0
	for _, v := range out {
		norm += v * v
	}
	if norm > 0 {
		norm = math.Sqrt(norm)
		for i := range out {
			out[i] /= norm
		}
	}
	return out
}
