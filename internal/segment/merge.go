// 相似区域合并：将感知哈希相近且空间邻近/相邻的区域合并为组合区域。
package segment

import (
	"image"
	"image/color"
	"sort"

	"go-image-search/internal/phash"
)

// MergeConfig 相似区域合并参数。
type MergeConfig struct {
	Enabled      bool    // 是否启用合并
	HashDist     int     // 区域感知哈希汉明距离阈值（≤0 使用默认 16）
	ColorDist    float64 // 区域平均色归一化距离阈值（≤0 使用默认 0.25）
	Connectivity int     // 邻接判定连通性（4 或 8，默认 4）
	GapFactor    float64 // 空间邻近：bbox 间距 ≤ GapFactor×min(区域尺寸) 视为邻近（0 表示仅严格邻接）
	Additive     bool    // 组合区域是否与原始区域共存（true 时保留全部原区域，仅追加组合区域）
}

// DefaultMergeConfig 返回推荐默认合并参数。
func DefaultMergeConfig() MergeConfig {
	return MergeConfig{
		Enabled:      true,
		HashDist:     16,
		ColorDist:    0.25,
		Connectivity: 4,
		GapFactor:    0.4,
	}
}

// RegionInfo 供合并使用的单区域信息。
type RegionInfo struct {
	ID    int
	Hash  uint64
	Shape uint64 // 颜色无关结构哈希（Otsu 二值掩码），合并时传递
	Area  int
	Color color.RGBA
	BBox  image.Rectangle
}

// MergedRegion 合并后的区域：组合区域或未合并的单区域。
type MergedRegion struct {
	ID        int             // 输出区域 ID（重新编号，1 起）
	Members   []int           // 构成该区域的原始区域 ID（组合区域为多个）
	Hash      uint64          // 组合区域裁剪块的感知哈希
	Shape     uint64          // 组合区域裁剪块的结构哈希
	Area      int             // 面积（组合区域为成员面积之和）
	BBox      image.Rectangle // 组合区域为成员 bbox 之并
	MeanColor color.RGBA      // 组合区域为面积加权平均色
}

// MergeSimilar 将感知哈希相近且空间邻近/相邻的区域合并为组合区域。
// 判据：两区域空间接近（严格邻接，或当 GapFactor>0 时 bbox 间距
// ≤ GapFactor×min(区域尺寸)），且 pHash 汉明距离 ≤ HashDist、
// 平均色归一化距离 ≤ ColorDist。合并是传递的（A~B、B~C ⇒ A、B、C 合并）。
// 未合并的单区域原样保留；组合区域对 src 的并集 bbox 重新计算感知哈希。
func MergeSimilar(src image.Image, res *Result, infos []RegionInfo, cfg MergeConfig) []MergedRegion {
	if !cfg.Enabled || len(infos) < 2 {
		out := make([]MergedRegion, 0, len(infos))
		for i, in := range infos {
			out = append(out, MergedRegion{
				ID: i + 1, Members: []int{in.ID},
				Hash: in.Hash, Shape: in.Shape, Area: in.Area, BBox: in.BBox, MeanColor: in.Color,
			})
		}
		return out
	}
	if cfg.HashDist <= 0 {
		cfg.HashDist = 16
	}
	if cfg.ColorDist <= 0 {
		cfg.ColorDist = 0.25
	}
	if cfg.Connectivity != 8 {
		cfg.Connectivity = 4
	}

	byID := make(map[int]RegionInfo, len(infos))
	for _, in := range infos {
		byID[in.ID] = in
	}

	// 1) 由标签图建立区域邻接表
	edges := regionAdjacency(res, cfg.Connectivity)

	// 2) 相似 + 空间邻近的区域之间建边
	uf := &unionFind{parent: make(map[int]int, len(infos))}
	consider := func(a, b int) {
		if b <= a {
			return
		}
		ia, okA := byID[a]
		if !okA {
			return
		}
		ib, okB := byID[b]
		if !okB {
			return
		}
		if phash.Hamming(ia.Hash, ib.Hash) <= cfg.HashDist &&
			colorDistNorm01(ia.Color, ib.Color) <= cfg.ColorDist &&
			spatiallyClose(ia.BBox, ib.BBox, cfg.GapFactor) {
			uf.union(a, b)
		}
	}
	if cfg.GapFactor <= 0 {
		var keys []int
		for a := range edges {
			keys = append(keys, a)
		}
		sort.Ints(keys)
		for _, a := range keys {
			var nb []int
			for b := range edges[a] {
				nb = append(nb, b)
			}
			sort.Ints(nb)
			for _, b := range nb {
				consider(a, b)
			}
		}
	} else {
		ids := make([]int, 0, len(byID))
		for id := range byID {
			ids = append(ids, id)
		}
		sort.Ints(ids)
		for i := 0; i < len(ids); i++ {
			for j := i + 1; j < len(ids); j++ {
				consider(ids[i], ids[j])
			}
		}
	}

	// 3) 按并查集根聚类，保持确定顺序（按最小成员 ID 排序）
	clusters := make(map[int][]int)
	for _, in := range infos {
		root := uf.find(in.ID)
		clusters[root] = append(clusters[root], in.ID)
	}
	keys := make([]int, 0, len(clusters))
	for root := range clusters {
		keys = append(keys, root)
	}
	sort.Slice(keys, func(i, j int) bool {
		return minID(clusters[keys[i]]) < minID(clusters[keys[j]])
	})

	// 4) 生成输出区域。
	//    Additive 模式下保留全部原始区域，仅追加组合区域（size≥2 的聚类）。
	//    否则组合区域替换其成员（replace 模式）。
	var out []MergedRegion
	if cfg.Additive {
		out = make([]MergedRegion, 0, len(infos)+len(keys))
		for i, in := range infos {
			out = append(out, MergedRegion{
				ID: i + 1, Members: []int{in.ID},
				Hash: in.Hash, Shape: in.Shape, Area: in.Area, BBox: in.BBox, MeanColor: in.Color,
			})
		}
		id := len(infos)
		for _, root := range keys {
			members := clusters[root]
			sort.Ints(members)
			if len(members) < 2 {
				continue
			}
			id++
			mr := mergeCluster(src, members, byID)
			mr.ID = id
			out = append(out, mr)
		}
		return out
	}

	out = make([]MergedRegion, 0, len(clusters))
	id := 0
	for _, root := range keys {
		members := clusters[root]
		sort.Ints(members)
		id++
		if len(members) == 1 {
			in := byID[members[0]]
			out = append(out, MergedRegion{
				ID: id, Members: members,
				Hash: in.Hash, Shape: in.Shape, Area: in.Area, BBox: in.BBox, MeanColor: in.Color,
			})
			continue
		}
		mr := mergeCluster(src, members, byID)
		mr.ID = id
		out = append(out, mr)
	}
	return out
}

// mergeCluster 合并一个成员集合为组合区域。
func mergeCluster(src image.Image, members []int, byID map[int]RegionInfo) MergedRegion {
	var area int64
	var sr, sg, sb int64
	bbox := byID[members[0]].BBox
	for _, m := range members {
		in := byID[m]
		area += int64(in.Area)
		sr += int64(in.Color.R) * int64(in.Area)
		sg += int64(in.Color.G) * int64(in.Area)
		sb += int64(in.Color.B) * int64(in.Area)
		bbox = bbox.Union(in.BBox)
	}
	mean := color.RGBA{}
	if area > 0 {
		mean = color.RGBA{R: uint8(sr / area), G: uint8(sg / area), B: uint8(sb / area), A: 255}
	}
	mr := MergedRegion{
		Members:   members,
		Area:      int(area),
		BBox:      bbox,
		MeanColor: mean,
	}
	if crop := cropRect(src, bbox); crop != nil {
		mr.Hash = phash.Hash(crop)
		mr.Shape = phash.Hash(structuralMask(crop))
	}
	return mr
}

// structuralMask 生成颜色无关的结构掩码（灰度 → Otsu 二值，前景为黑）。
// 与 imageproc.StructuralMask 语义一致，供合并区域计算结构哈希使用，
// 避免 segment 引入对 imageproc 的依赖。
func structuralMask(src image.Image) *image.Gray {
	b := src.Bounds()
	g := image.NewGray(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			g.Set(x, y, color.GrayModel.Convert(src.At(x, y)))
		}
	}
	hist := make([]int, 256)
	for _, v := range g.Pix {
		hist[v]++
	}
	total := len(g.Pix)
	if total == 0 {
		return g
	}
	sum := 0
	for i, n := range hist {
		sum += i * n
	}
	sumB, wB := 0, 0
	bestThr, bestVar := uint8(0), float64(-1)
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
	out := image.NewGray(b)
	for i, v := range g.Pix {
		if v < bestThr {
			out.Pix[i] = 0
		} else {
			out.Pix[i] = 255
		}
	}
	return out
}

// regionAdjacency 从标签图提取相邻区域对（无向、去重）。
func regionAdjacency(res *Result, connectivity int) map[int]map[int]bool {
	w, h := res.Width, res.Height
	labels := res.Labels
	edges := make(map[int]map[int]bool)
	add := func(a, b int32) {
		if a <= 0 || b <= 0 || a == b {
			return
		}
		if a > b {
			a, b = b, a
		}
		m := edges[int(a)]
		if m == nil {
			m = make(map[int]bool)
			edges[int(a)] = m
		}
		m[int(b)] = true
	}
	idx := func(x, y int) int { return y*w + x }
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			l := labels[idx(x, y)]
			if l <= 0 {
				continue
			}
			if x+1 < w {
				add(l, labels[idx(x+1, y)])
			}
			if y+1 < h {
				add(l, labels[idx(x, y+1)])
			}
			if connectivity == 8 {
				if x+1 < w && y+1 < h {
					add(l, labels[idx(x+1, y+1)])
				}
				if x+1 < w && y-1 >= 0 {
					add(l, labels[idx(x+1, y-1)])
				}
			}
		}
	}
	return edges
}

// cropRect 裁剪出矩形区域的内容。
func cropRect(src image.Image, b image.Rectangle) *image.RGBA {
	bb := b.Intersect(src.Bounds())
	if bb.Empty() {
		return nil
	}
	out := image.NewRGBA(bb)
	for y := bb.Min.Y; y < bb.Max.Y; y++ {
		for x := bb.Min.X; x < bb.Max.X; x++ {
			out.Set(x, y, src.At(x, y))
		}
	}
	return out
}

// colorDistNorm01 返回两个颜色的归一化 RGB 距离 [0,1]。
func colorDistNorm01(a, b color.RGBA) float64 {
	dr := float64(a.R) - float64(b.R)
	dg := float64(a.G) - float64(b.G)
	db := float64(a.B) - float64(b.B)
	return (dr*dr + dg*dg + db*db) / (3 * 255 * 255)
}

// spatiallyClose 判定两个区域是否空间邻近。
// factor ≤ 0 表示仅允许严格邻接（间距 0）；factor > 0 时允许 bbox 间距
// ≤ factor × min(两区域最大边长)。间距取两个方向间距的较大值。
func spatiallyClose(a, b image.Rectangle, factor float64) bool {
	if factor <= 0 {
		return gap1D(a.Min.X, a.Max.X, b.Min.X, b.Max.X) == 0 &&
			gap1D(a.Min.Y, a.Max.Y, b.Min.Y, b.Max.Y) == 0
	}
	gx := gap1D(a.Min.X, a.Max.X, b.Min.X, b.Max.X)
	gy := gap1D(a.Min.Y, a.Max.Y, b.Min.Y, b.Max.Y)
	dimA := float64(a.Dx())
	if float64(a.Dy()) > dimA {
		dimA = float64(a.Dy())
	}
	dimB := float64(b.Dx())
	if float64(b.Dy()) > dimB {
		dimB = float64(b.Dy())
	}
	scale := dimA
	if dimB < scale {
		scale = dimB
	}
	gap := gx
	if gy > gap {
		gap = gy
	}
	return gap <= factor*scale
}

// gap1D 返回两条线段在一维上的间距（重叠或相接为 0）。
func gap1D(aMin, aMax, bMin, bMax int) float64 {
	if aMax <= bMin {
		return float64(bMin - aMax)
	}
	if bMax <= aMin {
		return float64(aMin - bMax)
	}
	return 0
}

// unionFind 简单整数并查集。
type unionFind struct {
	parent map[int]int
}

func (u *unionFind) find(x int) int {
	p, ok := u.parent[x]
	if !ok {
		u.parent[x] = x
		return x
	}
	if p != x {
		u.parent[x] = u.find(p)
	}
	return u.parent[x]
}

func (u *unionFind) union(a, b int) {
	ra, rb := u.find(a), u.find(b)
	if ra != rb {
		u.parent[rb] = ra
	}
}

func minID(ids []int) int {
	m := ids[0]
	for _, v := range ids[1:] {
		if v < m {
			m = v
		}
	}
	return m
}
