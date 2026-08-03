// Package index 提供基于区域感知哈希的倒排索引构建与检索。
package index

import (
	"encoding/gob"
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"
	"os"
	"sort"

	"go-image-search/internal/phash"
)

// 索引分段参数：64-bit 哈希拆成 8 个 8-bit 段，每段翻转 ≤2 位生成变体，
// 可保证召回汉明距离 ≤ 8×3 = 24 的全部条目（桶内再按 MaxDist 过滤）。
const (
	segCount = 8
	segBits  = 8
	maxFlips = 3
	// 整图辅助区域（绿框）探针允许更多翻转：文字干扰/背景色变化会使哈希漂移，
	// 需要更大的召回半径。
	maxWholeFlips = 3
)

// RegionHash 表示一个待入索引的区域及其感知哈希。
type RegionHash struct {
	RegionID int
	Hash     uint64 // 颜色感知哈希（对光照/颜色敏感）
	Shape    uint64 // 颜色无关结构哈希（Otsu 二值掩码），颜色变化时仍稳定
	Area     int
	BBox     image.Rectangle
	Color    color.RGBA // 区域平均色，用于颜色相似度加权
	NX, NY   float64    // 归一化质心（相对整幅图，0~1），用于布局一致性
	Fill     float64    // 填充率：区域面积 / bbox 面积
	Aspect   float64    // 区域宽高比
	Global   float64    // 全局信息权重（整图辅助最高、组合区域次之，≥1）
}

// RegionEntry 索引中的一条区域记录。
type RegionEntry struct {
	ImageID  string
	RegionID int
	Hash     uint64
	Shape    uint64
	Area     int
	BBox     image.Rectangle
	Color    color.RGBA
	NX, NY   float64
	Fill     float64
	Aspect   float64
	Global   float64 // 全局信息权重
}

// Index 倒排索引：64-bit 哈希拆成 8 个 8-bit 分段。
type Index struct {
	Segments [segCount]map[uint8][]int
	Shapes   [segCount]map[uint8][]int
	Entries  []RegionEntry
	Images   map[string]bool
}

// New 创建一个空索引。
func New() *Index {
	ix := &Index{
		Images: make(map[string]bool),
	}
	for s := range ix.Segments {
		ix.Segments[s] = make(map[uint8][]int)
	}
	for s := range ix.Shapes {
		ix.Shapes[s] = make(map[uint8][]int)
	}
	return ix
}

// AddImage 将一个图像的全部区域加入索引。
func (ix *Index) AddImage(imageID string, regions []RegionHash) {
	for _, r := range regions {
		pos := len(ix.Entries)
		ix.Entries = append(ix.Entries, RegionEntry{
			ImageID:  imageID,
			RegionID: r.RegionID,
			Hash:     r.Hash,
			Shape:    r.Shape,
			Area:     r.Area,
			BBox:     r.BBox,
			Color:    r.Color,
			NX:       r.NX,
			NY:       r.NY,
			Fill:     r.Fill,
			Aspect:   r.Aspect,
			Global:   r.Global,
		})
		for s := 0; s < segCount; s++ {
			key := segment8(r.Hash, s)
			ix.Segments[s][key] = append(ix.Segments[s][key], pos)
		}
		if r.Shape != 0 {
			for s := 0; s < segCount; s++ {
				key := segment8(r.Shape, s)
				ix.Shapes[s][key] = append(ix.Shapes[s][key], pos)
			}
		}
	}
	ix.Images[imageID] = true
}

// Len 返回区域条目总数。
func (ix *Index) Len() int { return len(ix.Entries) }

func segment8(h uint64, s int) uint8 {
	return uint8(h >> (8 * uint(s)) & 0xff)
}

// QueryRegion 查询图像的一个区域。
type QueryRegion struct {
	Hash   uint64
	Shape  uint64
	Area   int
	Color  color.RGBA
	NX, NY float64
	Fill   float64
	Aspect float64
	Global float64 // 全局信息权重
}

// RegionMatch 单个区域命中。
type RegionMatch struct {
	Entry RegionEntry
	Dist  int
	QI    int  // 匹配的查询区域下标
	W     float64 // 该查询区域的 idf 权重（调试用）
}

// Match 图像级匹配结果。
type Match struct {
	ImageID    string
	Score      float64 // [0,1] 越大越相似
	CoverRatio float64 // 匹配区域面积占查询总面积的比重
	Matches    []RegionMatch
}

// SearchOptions 检索参数。
type SearchOptions struct {
	MaxDist      int     // 区域哈希最大汉明距离，0 表示默认 16（与召回保证一致）
	ColorWeight  float64 // 颜色相似度权重(0~1)，0 不使用；负数表示默认 0.8
	ShapeWeight  float64 // 形状(填充率/宽高比)权重，0 不使用；负数表示默认 0.4
	LayoutWeight float64 // 布局一致性权重，负数关闭；0 表示默认 0.4
}

// pairHit 表示查询区域 qi 与条目 pos 之间一个可行的候选对。
type pairHit struct {
	qi  int
	pos int
	d   float64
}

type pt struct{ x, y float64 }

// 查询区域权重：idf × √面积。idf 保留"稀有区域更关键"的语义；
// √面积 抑制查询图像中大量抗锯齿产生的小碎片（它们在很多图里都有、
// df 高、idf 低，但数量多会让"匹配碎片最多"的图像虚高）。
const scoreAreaPow = 0.5

// Search 用查询区域的哈希集合检索，返回按得分降序的图像列表。
// 流程：8×8-bit 分段变体探针收集候选 → 按图像聚合 → 二分图一对一分配
// （避免多个相似查询区域绑定同一物理区域）→ 按“匹配权重占比 × 平均相似度”
// 打分，并用匹配区域的空间布局一致性做二次排序。
func (ix *Index) Search(query []QueryRegion, opts SearchOptions) []Match {
	if opts.MaxDist <= 0 {
		opts.MaxDist = 16
	}
	if opts.ColorWeight < 0 {
		opts.ColorWeight = 0.1
	}
	if opts.ShapeWeight == 0 {
		opts.ShapeWeight = 0.4
	} else if opts.ShapeWeight < 0 {
		opts.ShapeWeight = 0
	}
	layoutW := 0.4
	if opts.LayoutWeight < 0 {
		layoutW = 0
	} else if opts.LayoutWeight > 0 {
		layoutW = opts.LayoutWeight
	}

	totalArea := 0
	for _, q := range query {
		totalArea += q.Area
	}
	if totalArea == 0 {
		return nil
	}
	nImg := float64(len(ix.Images))
	if nImg < 1 {
		nImg = 1
	}

	// 1) 候选收集：每查询区域对 8 个分段做 ≤2 位翻转探针。
	//    同一 (qi,pos) 可能经多个分段命中，按查询区域去重。
	imgPairs := make(map[string][]pairHit)
	df := make([]int, len(query)) // 查询区域命中的图像数（用于 idf）
	for qi, q := range query {
		seenPos := make(map[int]struct{})
		byImg := make(map[string]float64)
		// 整图辅助区域（绿框）的哈希漂移较大（文字干扰/背景色变化），
		// 探针允许更多翻转以召回，其余区域仍按 ≤2 翻转以控制候选量。
		// 仅当填充率足够（真实完整图标而非稀疏骨架）时启用整图探针。
		flips := maxFlips
		if q.Global >= 2.5 && q.Fill >= 0.15 {
			flips = maxWholeFlips
		}
		// 颜色哈希与结构哈希各按 8 个分段做翻转探针；
		// 两者各有独立的倒排桶（Shapes），命中任一即成为该区域的候选。
		for s := 0; s < segCount; s++ {
			key := segment8(q.Hash, s)
			for _, v := range variantsN(key, flips) {
				for _, pos := range ix.Segments[s][v] {
					ix.considerHit(qi, q, pos, &seenPos, &byImg, imgPairs, opts)
				}
			}
			if q.Shape != 0 {
				key := segment8(q.Shape, s)
				for _, v := range variantsN(key, flips) {
					for _, pos := range ix.Shapes[s][v] {
						ix.considerHit(qi, q, pos, &seenPos, &byImg, imgPairs, opts)
					}
				}
			}
		}
		df[qi] = len(byImg)
	}

	// 2) 每个查询区域按稀有度（idf）× √面积 加权：
	//    idf 保留"稀有区域更关键"的语义；√面积 抑制查询图像中大量
	//    抗锯齿产生的小碎片（它们在很多图里都有，df 高、idf 低，
	//    但数量多会让"匹配碎片最多"的图像虚高）。
	idf := make([]float64, len(query))
	totalW := 0.0
	for qi := range query {
		// df=0 表示该查询区域在索引中无任何命中：仍按 df=1（最大稀有度）计入分母，
		// 避免"只有一个罕见区域命中"导致 countRatio 虚高。
		d := df[qi]
		if d < 1 {
			d = 1
		}
		base := math.Log(1+nImg) / math.Log(1+float64(d))
		g := query[qi].Global
		if g < 1 {
			g = 1
		}
		idf[qi] = base * math.Pow(float64(query[qi].Area), scoreAreaPow) * g
		totalW += idf[qi]
	}
	if totalW <= 0 {
		return nil
	}

	// 3) 逐图一对一分配与打分。
	matches := make([]Match, 0, len(imgPairs))
	const hungarianLimit = 4096
	for imgID, pairs := range imgPairs {
		rows := orderedUniqueQI(pairs)
		cols := orderedUniquePos(pairs)
		if len(rows) == 0 || len(cols) == 0 {
			continue
		}
		rowIdx := make(map[int]int, len(rows))
		for i, qi := range rows {
			rowIdx[qi] = i
		}
		colIdx := make(map[int]int, len(cols))
		for i, pos := range cols {
			colIdx[pos] = i
		}
		cost := make([][]float64, len(rows))
		for i := range cost {
			cost[i] = make([]float64, len(cols))
			for j := range cost[i] {
				cost[i][j] = math.Inf(1)
			}
		}
		for _, p := range pairs {
			cost[rowIdx[p.qi]][colIdx[p.pos]] = p.d
		}

		var assign []int
		if len(rows)*len(cols) > hungarianLimit {
			assign = greedyAssign(cost)
		} else {
			assign = hungarian(cost)
		}

		matchedW := 0.0
		scoreSum := 0.0
		matchedArea := 0
		qCen := make([]pt, 0, len(rows))
		eCen := make([]pt, 0, len(rows))
		var regMatches []RegionMatch
		for ri, qi := range rows {
			if assign[ri] < 0 {
				continue
			}
			d := cost[ri][assign[ri]]
			if math.IsInf(d, 1) {
				continue
			}
			sim := 1 - d/64
			if sim < 0 {
				sim = 0
			}
			matchedW += idf[qi]
			scoreSum += idf[qi] * sim
			matchedArea += query[qi].Area
			qCen = append(qCen, pt{query[qi].NX, query[qi].NY})
			eCen = append(eCen, pt{ix.Entries[cols[assign[ri]]].NX, ix.Entries[cols[assign[ri]]].NY})
			regMatches = append(regMatches, RegionMatch{
				Entry: ix.Entries[cols[assign[ri]]],
				Dist:  int(d),
				QI:    qi,
				W:     idf[qi],
			})
		}
		if matchedW <= 0 {
			continue
		}
		avgSim := scoreSum / matchedW
		countRatio := matchedW / totalW
		score := avgSim * countRatio
		if layoutW > 0 && len(qCen) > 1 {
			ls := layoutScore(qCen, eCen)
			score *= 1 + layoutW*(ls-1)
		}
		matches = append(matches, Match{
			ImageID:    imgID,
			Score:      score,
			CoverRatio: float64(matchedArea) / float64(totalArea),
			Matches:    regMatches,
		})
	}

	sort.SliceStable(matches, func(i, j int) bool {
		if matches[i].Score != matches[j].Score {
			return matches[i].Score > matches[j].Score
		}
		if matches[i].CoverRatio != matches[j].CoverRatio {
			return matches[i].CoverRatio > matches[j].CoverRatio
		}
		return matches[i].ImageID < matches[j].ImageID
	})
	return matches
}

// considerHit 处理一次探针命中：若 (qi,pos) 哈希或结构的汉明距离在门槛内，
// 则记录候选。结构哈希与颜色哈希取较近者作为基础距离门限。
// 颜色哈希与结构哈希各按 8 个分段做 ≤2 位翻转探针；二者独立倒排，命中任一
// 即为该区域的候选。颜色变化导致颜色哈希漂移时，仍可按稳定结构召回。
func (ix *Index) considerHit(qi int, q QueryRegion, pos int, seen *map[int]struct{}, byImg *map[string]float64, imgPairs map[string][]pairHit, opts SearchOptions) {
	if _, ok := (*seen)[pos]; ok {
		return
	}
	(*seen)[pos] = struct{}{}
	e := ix.Entries[pos]
	gd := phash.Hamming(q.Hash, e.Hash)
	// 整图辅助区域（绿框）之间的匹配：文字干扰/背景色变化会使哈希显著漂移，
	// 但整图结构仍应保留，故放宽距离门限。仅当查询区域与条目均为整图辅助区域
	// 且填充率足够（真实完整图标，而非稀疏骨架）时生效。
	limit := opts.MaxDist
	if q.Global >= 2.5 && e.Global >= 2.5 && q.Fill >= 0.15 && e.Fill >= 0.15 && limit < 24 {
		limit = 24
	}
	if gd > limit {
		if q.Shape == 0 || e.Shape == 0 {
			return
		}
		sd := phash.Hamming(q.Shape, e.Shape)
		if sd > limit {
			return
		}
	}
	d := pairDist(q, e, opts)
	if d >= 64 {
		return
	}
	if prev, ok := (*byImg)[e.ImageID]; !ok || d < prev {
		(*byImg)[e.ImageID] = d
	}
	imgPairs[e.ImageID] = append(imgPairs[e.ImageID], pairHit{qi: qi, pos: pos, d: d})
}

// rankLess 是确定性的图像排序比较：得分降序，其次覆盖比降序，最后按图像 ID 升序。
// 这里特意在最后用 ImageID 决胜，避免 map 迭代顺序导致的非确定排序。
func rankLess(a, b Match) bool {
	if a.Score != b.Score {
		return a.Score > b.Score
	}
	if a.CoverRatio != b.CoverRatio {
		return a.CoverRatio > b.CoverRatio
	}
	return a.ImageID < b.ImageID
}

// SearchMulti 对多组查询区域（原图及其衍生图）依次检索并按图像合并结果。
// 每张图像取各衍生图检索得分中的最高分，并统计被多少个衍生图命中；
// 被多个衍生图同时命中的图像获得小幅加成（多视角确认更可信）。
func (ix *Index) SearchMulti(querySets [][]QueryRegion, opts SearchOptions) []Match {
	const multiBoost = 0.08
	best := make(map[string]Match)
	hits := make(map[string]int)
	for _, qs := range querySets {
		if len(qs) == 0 {
			continue
		}
		for _, m := range ix.Search(qs, opts) {
			if prev, ok := best[m.ImageID]; !ok || m.Score > prev.Score {
				best[m.ImageID] = m
			}
			hits[m.ImageID]++
		}
	}
	out := make([]Match, 0, len(best))
	for id, m := range best {
		if k := hits[id]; k > 1 {
			m.Score *= 1 + multiBoost*float64(k-1)
		}
		out = append(out, m)
	}
	sort.SliceStable(out, func(i, j int) bool { return rankLess(out[i], out[j]) })
	return out
}

// pairDist 组合哈希、颜色、形状三个维度的相似度距离。
// 哈希维度取“颜色哈希”与“结构哈希”中较近者——当颜色变化导致颜色哈希
// 漂移时，仍可用稳定的结构哈希度量形状相似，从而容忍如 TEST8 的颜色变动。
//
// 整图辅助区域（双方 Global≥2.5）跳过形状权重项：整图的填充率/宽高比
// 随裁剪/缩放而变（局部视图的整图区域填充率与原图差异大），不具备判别力，
// 反而会惩罚正确的整图-整图匹配。颜色哈希已做极性归一化，足以度量整图相似。
func pairDist(q QueryRegion, e RegionEntry, opts SearchOptions) float64 {
	hd := phash.Hamming(q.Hash, e.Hash)
	if q.Shape != 0 && e.Shape != 0 {
		if sd := phash.Hamming(q.Shape, e.Shape); sd < hd {
			hd = sd
		}
	}
	d := float64(hd)
	if opts.ColorWeight > 0 {
		d += opts.ColorWeight * colorWeight01(q, e) * colorDist01(q.Color, e.Color) * 64
	}
	if opts.ShapeWeight > 0 && !(q.Global >= 2.5 && e.Global >= 2.5) {
		d += opts.ShapeWeight * shapeDist01(q, e) * 64
	}
	return d
}

// colorWeight01 对大面积、高填充的背景类区域降低颜色权重。
// 这类区域常是整块底色/底图，颜色多随主题而变化（如 TEST2/图标卡片），
// 而其形状又已被结构哈希刻画；高填充实心块的颜色信息价值低，
// 予以衰减，保留颜色用于匹配结构鲜明的图标区域。
func colorWeight01(q QueryRegion, e RegionEntry) float64 {
	return fillScale(q.Fill) * fillScale(e.Fill)
}

// fillScale 把填充率映射为颜色权重系数：低填充（轮廓/图形）保持权重，
// 高填充（实心大块）线性衰减到下限。
func fillScale(fill float64) float64 {
	const lo, hi, minW = 0.3, 0.8, 0.15
	if fill <= 0 {
		return 1
	}
	if fill >= hi {
		return minW
	}
	return 1 - (1-minW)*((fill-lo)/(hi-lo))
}

// colorDist01 返回两个颜色的归一化 RGB 距离 [0,1]。
func colorDist01(a, b color.RGBA) float64 {
	dr := float64(a.R) - float64(b.R)
	dg := float64(a.G) - float64(b.G)
	db := float64(a.B) - float64(b.B)
	dist := dr*dr + dg*dg + db*db
	return dist / (3 * 255 * 255)
}

// shapeDist01 返回填充率与宽高比差异的归一化距离 [0,1]。
func shapeDist01(q QueryRegion, e RegionEntry) float64 {
	fill := math.Abs(q.Fill - e.Fill)
	aspect := math.Log((q.Aspect + 1e-6) / (e.Aspect + 1e-6))
	if aspect < 0 {
		aspect = -aspect
	}
	if aspect > 1 {
		aspect = 1
	}
	return 0.5*fill + 0.5*aspect
}

// layoutScore 度量匹配区域的整体空间布局一致性（0~1，越大越一致）。
// 对两套质心分别做“匹配集内按轴归一化”，排序后逐点比较——
// 该度量对平移/缩放（含各向异性）不变，且不受匹配对应次序影响。
func layoutScore(q, e []pt) float64 {
	n := len(q)
	if n < 2 {
		return 1
	}
	norm := func(pts []pt) []pt {
		out := make([]pt, len(pts))
		minX, maxX := pts[0].x, pts[0].x
		minY, maxY := pts[0].y, pts[0].y
		for _, p := range pts[1:] {
			minX = math.Min(minX, p.x)
			maxX = math.Max(maxX, p.x)
			minY = math.Min(minY, p.y)
			maxY = math.Max(maxY, p.y)
		}
		rx, ry := maxX-minX, maxY-minY
		for i, p := range pts {
			x, y := 0.5, 0.5
			if rx > 0 {
				x = (p.x - minX) / rx
			}
			if ry > 0 {
				y = (p.y - minY) / ry
			}
			out[i] = pt{x, y}
		}
		return out
	}
	qn := norm(q)
	en := norm(e)
	sort.Slice(qn, func(a, b int) bool {
		if qn[a].x != qn[b].x {
			return qn[a].x < qn[b].x
		}
		return qn[a].y < qn[b].y
	})
	sort.Slice(en, func(a, b int) bool {
		if en[a].x != en[b].x {
			return en[a].x < en[b].x
		}
		return en[a].y < en[b].y
	})
	sum := 0.0
	for i := 0; i < n; i++ {
		dx := qn[i].x - en[i].x
		dy := qn[i].y - en[i].y
		sum += math.Sqrt(dx*dx + dy*dy)
	}
	ls := 1 - sum/float64(n)/math.Sqrt2
	if ls < 0 {
		ls = 0
	}
	return ls
}

func orderedUniqueQI(pairs []pairHit) []int {
	seen := make(map[int]bool)
	out := make([]int, 0, len(pairs))
	for _, p := range pairs {
		if !seen[p.qi] {
			seen[p.qi] = true
			out = append(out, p.qi)
		}
	}
	return out
}

func orderedUniquePos(pairs []pairHit) []int {
	seen := make(map[int]bool)
	out := make([]int, 0, len(pairs))
	for _, p := range pairs {
		if !seen[p.pos] {
			seen[p.pos] = true
			out = append(out, p.pos)
		}
	}
	return out
}

// variants 返回 8-bit 值翻转 ≤maxFlips 位后的全部变体。
func variants(key uint8) []uint8 {
	return variantsN(key, maxFlips)
}

// variantsN 返回 8-bit 值翻转 ≤n 位后的全部变体。
func variantsN(key uint8, n int) []uint8 {
	if n < 0 {
		n = 0
	}
	if n > 8 {
		n = 8
	}
	out := make([]uint8, 0, 128)
	out = append(out, key)
	// 用组合位掩码生成翻转 ≤n 位的变体
	for flip := 1; flip <= n; flip++ {
		comb(key, 0, uint8(flip), 0, &out)
	}
	return out
}

// comb 递归生成翻转 flip 位的掩码组合。
func comb(key uint8, start, flip, mask uint8, out *[]uint8) {
	if flip == 0 {
		*out = append(*out, key^mask)
		return
	}
	for b := start; b <= 8-flip; b++ {
		comb(key, b+1, flip-1, mask|(uint8(1)<<b), out)
	}
}

// hungarianPad 匈牙利算法的哑元成本，远大于真实成本总和（≤64×64）。
const hungarianPad = 1e7

// hungarian 求解矩形成本矩阵的最小成本二分图匹配，返回每行分配的列下标，
// 未分配到可行列的行返回 -1。内部填充成方阵以复用 O(n^3) 匈牙利算法。
func hungarian(cost [][]float64) []int {
	r := len(cost)
	res := make([]int, r)
	for i := range res {
		res[i] = -1
	}
	if r == 0 {
		return res
	}
	c := len(cost[0])
	if c == 0 {
		return res
	}
	n := r
	if c > n {
		n = c
	}
	a := make([][]float64, n)
	for i := 0; i < n; i++ {
		a[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			a[i][j] = hungarianPad
		}
	}
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if !math.IsInf(cost[i][j], 1) {
				a[i][j] = cost[i][j]
			}
		}
	}

	u := make([]float64, n+1)
	v := make([]float64, n+1)
	p := make([]int, n+1)
	way := make([]int, n+1)
	for i := 1; i <= n; i++ {
		p[0] = i
		j0 := 0
		minv := make([]float64, n+1)
		used := make([]bool, n+1)
		for j := 1; j <= n; j++ {
			minv[j] = math.Inf(1)
		}
		for {
			used[j0] = true
			i0 := p[j0]
			delta := math.Inf(1)
			j1 := 0
			for j := 1; j <= n; j++ {
				if used[j] {
					continue
				}
				cur := a[i0-1][j-1] - u[i0] - v[j]
				if cur < minv[j] {
					minv[j] = cur
					way[j] = j0
				}
				if minv[j] < delta {
					delta = minv[j]
					j1 = j
				}
			}
			for j := 0; j <= n; j++ {
				if used[j] {
					u[p[j]] += delta
					v[j] -= delta
				} else {
					minv[j] -= delta
				}
			}
			j0 = j1
			if p[j0] == 0 {
				break
			}
		}
		for {
			j1 := way[j0]
			p[j0] = p[j1]
			j0 = j1
			if j0 == 0 {
				break
			}
		}
	}
	for j := 1; j <= n; j++ {
		row := p[j] - 1
		if row >= 0 && row < r && j-1 < c && !math.IsInf(cost[row][j-1], 1) {
			res[row] = j - 1
		}
	}
	return res
}

// greedyAssign 贪心一对一分配（大矩阵降级用）：按成本升序逐个占用行列。
func greedyAssign(cost [][]float64) []int {
	r := len(cost)
	c := len(cost[0])
	res := make([]int, r)
	for i := range res {
		res[i] = -1
	}
	type cand struct {
		i, j int
		d    float64
	}
	all := make([]cand, 0, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if !math.IsInf(cost[i][j], 1) {
				all = append(all, cand{i, j, cost[i][j]})
			}
		}
	}
	sort.Slice(all, func(a, b int) bool { return all[a].d < all[b].d })
	rowUsed := make([]bool, r)
	colUsed := make([]bool, c)
	for _, cd := range all {
		if rowUsed[cd.i] || colUsed[cd.j] {
			continue
		}
		rowUsed[cd.i] = true
		colUsed[cd.j] = true
		res[cd.i] = cd.j
	}
	return res
}

// Save 将索引以 gob 编码写入文件。
func (ix *Index) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(ix)
}

// Load 从文件加载 gob 编码的索引。
func Load(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	ix := &Index{}
	if err := gob.NewDecoder(f).Decode(ix); err != nil {
		return nil, fmt.Errorf("decode index: %w", err)
	}
	if ix.Images == nil {
		return nil, errors.New("invalid index file")
	}
	return ix, nil
}
