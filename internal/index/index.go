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

// RegionHash 表示一个待入索引的区域及其感知哈希。
type RegionHash struct {
	RegionID int
	Hash     uint64
	Area     int
	BBox     image.Rectangle
	Color    color.RGBA // 区域平均色，用于颜色相似度加权
}

// RegionEntry 索引中的一条区域记录。
type RegionEntry struct {
	ImageID  string
	RegionID int
	Hash     uint64
	Area     int
	BBox     image.Rectangle
	Color    color.RGBA
}

// Index 倒排索引：64-bit 哈希拆成 4 个 16-bit 分段。
type Index struct {
	Segments [4]map[uint16][]int
	Entries  []RegionEntry
	Images   map[string]bool
}

// New 创建一个空索引。
func New() *Index {
	ix := &Index{
		Images: make(map[string]bool),
	}
	for s := range ix.Segments {
		ix.Segments[s] = make(map[uint16][]int)
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
			Area:     r.Area,
			BBox:     r.BBox,
			Color:    r.Color,
		})
		for s := 0; s < 4; s++ {
			key := segment16(r.Hash, s)
			ix.Segments[s][key] = append(ix.Segments[s][key], pos)
		}
	}
	ix.Images[imageID] = true
}

// Len 返回区域条目总数。
func (ix *Index) Len() int { return len(ix.Entries) }

func segment16(h uint64, s int) uint16 {
	return uint16((h >> (16 * uint(s))) & 0xffff)
}

// QueryRegion 查询图像的一个区域。
type QueryRegion struct {
	Hash  uint64
	Area  int
	Color color.RGBA
}

// RegionMatch 单个区域命中。
type RegionMatch struct {
	Entry RegionEntry
	Dist  int
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
	MaxDist     int     // 区域灰度哈希最大汉明距离，0 表示默认 12
	ColorWeight float64 // 颜色相似度权重(0~1)，0 表示不使用颜色；负数表示默认 0.6
}

// Search 用查询区域的哈希集合检索，返回按得分降序的图像列表。
func (ix *Index) Search(query []QueryRegion, opts SearchOptions) []Match {
	if opts.MaxDist <= 0 {
		opts.MaxDist = 12
	}
	if opts.ColorWeight < 0 {
		opts.ColorWeight = 0.8
	}

	totalArea := 0
	for _, q := range query {
		totalArea += q.Area
	}
	if totalArea == 0 {
		return nil
	}

	type qResult struct {
		best map[string]float64 // imageID -> 最优组合距离
		pos  map[string]int     // imageID -> 达到最优的条目下标
		idf  float64            // 逆文档频率，压制公共区域（如白色背景）
	}
	qr := make([]qResult, len(query))

	nImg := float64(len(ix.Images))
	if nImg < 1 {
		nImg = 1
	}

	for qi, q := range query {
		// 候选收集：4 个 16-bit 分段，每段翻转 ≤2 bit 生成变体
		cand := make(map[int]struct{})
		for s := 0; s < 4; s++ {
			key := segment16(q.Hash, s)
			for _, v := range variants(key) {
				for _, pos := range ix.Segments[s][v] {
					cand[pos] = struct{}{}
				}
			}
		}
		best := make(map[string]float64)
		bestPos := make(map[string]int)
		for pos := range cand {
			e := ix.Entries[pos]
			gd := phash.Hamming(q.Hash, e.Hash)
			if gd > opts.MaxDist {
				continue
			}
			d := float64(gd) + opts.ColorWeight*colorDist01(q.Color, e.Color)*64
			if prev, ok := best[e.ImageID]; !ok || d < prev {
				best[e.ImageID] = d
				bestPos[e.ImageID] = pos
			}
		}
		// idf：区域匹配到的图像越多（如白色背景），权重越低
		qr[qi].idf = 1
		if len(best) > 0 {
			qr[qi].idf = math.Log(1+nImg) / math.Log(1+float64(len(best)))
		}
		qr[qi].best = best
		qr[qi].pos = bestPos
	}

	// 计算查询区域归一化权重（面积 × idf）
	totalW := 0.0
	wa := make([]float64, len(query))
	for qi, q := range query {
		if len(qr[qi].best) == 0 {
			continue
		}
		w := float64(q.Area) * qr[qi].idf
		wa[qi] = w
		totalW += w
	}
	if totalW <= 0 {
		return nil
	}

	// 聚合到图像
	imgScore := make(map[string]float64)
	imgCover := make(map[string]int)
	imgMatches := make(map[string][]RegionMatch)

	for qi, q := range query {
		for imgID, d := range qr[qi].best {
			sc := 1 - d/64
			if sc < 0 {
				sc = 0
			}
			imgScore[imgID] += (wa[qi] / totalW) * sc
			imgCover[imgID] += q.Area
			imgMatches[imgID] = append(imgMatches[imgID], RegionMatch{
				Entry: ix.Entries[qr[qi].pos[imgID]],
				Dist:  int(d),
			})
		}
	}

	matches := make([]Match, 0, len(imgScore))
	for imgID, sc := range imgScore {
		matches = append(matches, Match{
			ImageID:    imgID,
			Score:      sc,
			CoverRatio: float64(imgCover[imgID]) / float64(totalArea),
			Matches:    imgMatches[imgID],
		})
	}
	sort.SliceStable(matches, func(i, j int) bool {
		if matches[i].Score != matches[j].Score {
			return matches[i].Score > matches[j].Score
		}
		return matches[i].CoverRatio > matches[j].CoverRatio
	})
	return matches
}

// colorDist01 返回两个颜色的归一化 RGB 距离 [0,1]。
func colorDist01(a, b color.RGBA) float64 {
	dr := float64(a.R) - float64(b.R)
	dg := float64(a.G) - float64(b.G)
	db := float64(a.B) - float64(b.B)
	dist := dr*dr + dg*dg + db*db
	return dist / (3 * 255 * 255)
}

// variants 返回 16-bit 值翻转 ≤2 位后的全部变体。
func variants(key uint16) []uint16 {
	out := make([]uint16, 0, 137)
	out = append(out, key)
	for i := 0; i < 16; i++ {
		out = append(out, key^(1<<i))
	}
	for i := 0; i < 16; i++ {
		for j := i + 1; j < 16; j++ {
			out = append(out, key^(1<<i)^(1<<j))
		}
	}
	return out
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
