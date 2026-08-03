package index

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"go-image-search/internal/phash"
	"go-image-search/internal/segment"
)

func mkRegionHash(regionID int, area int, bbox image.Rectangle, fill color.RGBA) RegionHash {
	img := image.NewRGBA(bbox)
	for y := bbox.Min.Y; y < bbox.Max.Y; y++ {
		for x := bbox.Min.X; x < bbox.Max.X; x++ {
			img.Set(x, y, fill)
		}
	}
	return RegionHash{RegionID: regionID, Hash: phash.Hash(img), Area: area, BBox: bbox}
}

func TestIndexSearch(t *testing.T) {
	ix := New()
	ix.AddImage("a.png", []RegionHash{
		mkRegionHash(1, 100, image.Rect(0, 0, 10, 10), color.RGBA{255, 0, 0, 255}),
		mkRegionHash(2, 300, image.Rect(10, 10, 30, 30), color.RGBA{0, 0, 255, 255}),
	})
	ix.AddImage("b.png", []RegionHash{
		mkRegionHash(1, 100, image.Rect(0, 0, 10, 10), color.RGBA{0, 255, 0, 255}),
		mkRegionHash(2, 300, image.Rect(10, 10, 30, 30), color.RGBA{255, 255, 0, 255}),
	})

	// 查询与 a 相同的红色块 + 蓝色块
	query := []QueryRegion{
		{Hash: mkRegionHash(9, 100, image.Rect(0, 0, 10, 10), color.RGBA{255, 0, 0, 255}).Hash, Area: 100},
		{Hash: mkRegionHash(9, 300, image.Rect(10, 10, 30, 30), color.RGBA{0, 0, 255, 255}).Hash, Area: 300},
	}
	matches := ix.Search(query, SearchOptions{MaxDist: 12})
	if len(matches) == 0 {
		t.Fatal("无匹配结果")
	}
	if matches[0].ImageID != "a.png" {
		t.Fatalf("期望匹配 a.png，得到 %s", matches[0].ImageID)
	}
}

// TestSearchMultiMergesVariants 验证 SearchMulti 合并多组衍生图检索结果：
// 某图像仅被某一个衍生图命中时也能被返回（取最高分）。
func TestSearchMultiMergesVariants(t *testing.T) {
	ix := New()
	ix.AddImage("a.png", []RegionHash{
		mkRegionHash(1, 100, image.Rect(0, 0, 10, 10), color.RGBA{255, 0, 0, 255}),
	})
	ix.AddImage("b.png", []RegionHash{
		mkRegionHash(1, 100, image.Rect(0, 0, 10, 10), color.RGBA{0, 0, 255, 255}),
	})

	ha := mkRegionHash(9, 100, image.Rect(0, 0, 10, 10), color.RGBA{255, 0, 0, 255}).Hash
	hNb := mkRegionHash(9, 100, image.Rect(0, 0, 10, 10), color.RGBA{0, 0, 255, 255}).Hash

	// 第一组（原图）：仅命中 a；第二组（衍生图）：命中 a 与 b。
	sets := [][]QueryRegion{
		{{Hash: ha, Area: 100}},
		{{Hash: ha, Area: 100}, {Hash: hNb, Area: 100}},
	}

	res := ix.SearchMulti(sets, SearchOptions{MaxDist: 12})
	if len(res) == 0 {
		t.Fatal("SearchMulti 无结果")
	}
	found := map[string]float64{}
	for _, m := range res {
		found[m.ImageID] = m.Score
	}
	if _, ok := found["a.png"]; !ok {
		t.Fatal("应命中 a.png")
	}
	if _, ok := found["b.png"]; !ok {
		t.Fatal("衍生图命中 b.png 应被返回")
	}
	if found["a.png"] <= 0 {
		t.Fatal("a.png 得分异常")
	}
}

func TestSaveLoadRoundtrip(t *testing.T) {
	ix := New()
	ix.AddImage("a.png", []RegionHash{
		mkRegionHash(1, 100, image.Rect(0, 0, 10, 10), color.RGBA{255, 0, 0, 255}),
	})
	path := filepath.Join(t.TempDir(), "idx.bin")
	if err := ix.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Len() != 1 {
		t.Fatalf("加载后条目数错误: %d", loaded.Len())
	}
	if !loaded.Images["a.png"] {
		t.Fatal("加载后丢失图像记录")
	}
	matches := loaded.Search([]QueryRegion{{Hash: ix.Entries[0].Hash, Area: 100}}, SearchOptions{})
	if len(matches) == 0 || matches[0].ImageID != "a.png" {
		t.Fatal("加载后检索失败")
	}
}

func TestVariants(t *testing.T) {
	v := variants(0x00)
	// 1 + 8 + 28 = 37
	if len(v) != 37 {
		t.Fatalf("变体数量错误: %d", len(v))
	}
	seen := map[uint8]bool{}
	for _, x := range v {
		seen[x] = true
	}
	if !seen[0x01] || !seen[0x80] || !seen[0x03] || !seen[0x05] {
		t.Fatal("变体缺少预期值")
	}
	if len(seen) != 37 {
		t.Fatal("变体存在重复")
	}
}

// TestRecallDistance16 验证 8×8-bit 分段 f=2 能保证召回汉明距离 16 的条目。
// 旧实现（4×16-bit 段 f=2）对每段差异 >2 位的条目完全漏检。
func TestRecallDistance16(t *testing.T) {
	base := uint64(0x0F0F0F0F0F0F0F0F)
	var mask uint64
	for s := 0; s < 8; s++ {
		mask |= 1 << uint(8*s)
		mask |= 1 << uint(8*s+1)
	}
	qh := base ^ mask
	if phash.Hamming(base, qh) != 16 {
		t.Fatalf("构造的汉明距离错误: %d", phash.Hamming(base, qh))
	}
	ix := New()
	ix.AddImage("a.png", []RegionHash{{RegionID: 1, Hash: base, Area: 100}})
	ix.AddImage("b.png", []RegionHash{{RegionID: 1, Hash: qh, Area: 100}})

	matches := ix.Search([]QueryRegion{{Hash: qh, Area: 100}}, SearchOptions{MaxDist: 16})
	for _, m := range matches {
		if m.ImageID == "a.png" {
			return
		}
	}
	t.Fatal("汉明距离 16 的条目未被召回（8×8 分段 f=2 应保证召回 ≤16）")
}

// TestOneToOneSimilarRegions 验证一对一分配：
// 查询有 6 个完全相同的小区域，full 图含 6 个、half 图仅含 3 个。
// 旧实现中 6 个查询区域会同时绑定到 half 的同一批区域导致误报高分；
// 一对一分配后 half 只能匹配 3 个。
func TestOneToOneSimilarRegions(t *testing.T) {
	H := uint64(0x0123456789abcdef)
	positions := []pt{
		{0.1, 0.1}, {0.3, 0.1}, {0.5, 0.1},
		{0.1, 0.5}, {0.3, 0.5}, {0.5, 0.5},
	}
	full := make([]RegionHash, len(positions))
	for i, p := range positions {
		full[i] = RegionHash{RegionID: i + 1, Hash: H, Area: 100, NX: p.x, NY: p.y, Fill: 1, Aspect: 1}
	}
	half := append([]RegionHash(nil), full[:3]...)

	ix := New()
	ix.AddImage("full.png", full)
	ix.AddImage("half.png", half)

	query := make([]QueryRegion, len(positions))
	for i, p := range positions {
		query[i] = QueryRegion{Hash: H, Area: 100, NX: p.x, NY: p.y, Fill: 1, Aspect: 1}
	}

	matches := ix.Search(query, SearchOptions{MaxDist: 12})
	if len(matches) == 0 {
		t.Fatal("无匹配结果")
	}
	if matches[0].ImageID != "full.png" {
		t.Fatalf("期望 top1=full.png，得到 %s", matches[0].ImageID)
	}
	for _, m := range matches {
		switch m.ImageID {
		case "full.png":
			if len(m.Matches) != 6 {
				t.Fatalf("full.png 应匹配 6 个区域，得到 %d", len(m.Matches))
			}
			if m.Score < 0.9 {
				t.Fatalf("full.png 得分过低: %f", m.Score)
			}
		case "half.png":
			if len(m.Matches) != 3 {
				t.Fatalf("half.png 应只匹配 3 个区域(一对一)，得到 %d", len(m.Matches))
			}
			if m.Score > 0.6 {
				t.Fatalf("half.png 得分虚高: %f", m.Score)
			}
		}
	}
}

// iconDots 绘制透明底、固定大小同色圆点阵的图标。
func iconDots(w, h, d int, centers [][2]int, fg color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h)) // 默认全透明
	half := d / 2
	for _, c := range centers {
		for y := c[1] - half; y < c[1]+d-half; y++ {
			for x := c[0] - half; x < c[0]+d-half; x++ {
				if x >= 0 && x < w && y >= 0 && y < h {
					img.Set(x, y, fg)
				}
			}
		}
	}
	return img
}

// hashRegions 走真实管线：分割 + 感知哈希 + 布局/形状特征。
func hashRegions(img image.Image, cfg segment.Config) []RegionHash {
	res, err := segment.Segment(img, cfg)
	if err != nil {
		panic(err)
	}
	var out []RegionHash
	for _, reg := range res.Regions {
		crop := res.Crop(img, reg.ID)
		if crop == nil {
			continue
		}
		bw, bh := reg.BBox.Dx(), reg.BBox.Dy()
		fill, aspect := 0.0, 0.0
		if bw > 0 && bh > 0 {
			fill = float64(reg.Area) / float64(bw*bh)
			aspect = float64(bw) / float64(bh)
		}
		nx, ny := 0.0, 0.0
		if res.Width > 0 {
			nx = float64(reg.BBox.Min.X+reg.BBox.Max.X) / (2 * float64(res.Width))
		}
		if res.Height > 0 {
			ny = float64(reg.BBox.Min.Y+reg.BBox.Max.Y) / (2 * float64(res.Height))
		}
		out = append(out, RegionHash{
			RegionID: reg.ID, Hash: phash.Hash(crop), Area: reg.Area, BBox: reg.BBox,
			Color: reg.MeanColor, NX: nx, NY: ny, Fill: fill, Aspect: aspect,
		})
	}
	return out
}

// TestLayoutSimilarRegions 端到端验证布局区分：
// 图标 A 与 B 都是 6 个相同圆点但排列不同，C 只有 3 个圆点。
// 期望：A 命中 top1，A 得分 > B（布局区分）> C（数量区分）。
func TestLayoutSimilarRegions(t *testing.T) {
	cfg := segment.DefaultConfig()
	fg := color.RGBA{220, 60, 60, 255}

	a := iconDots(360, 360, 40, [][2]int{
		{90, 90}, {180, 90}, {270, 90},
		{90, 270}, {180, 270}, {270, 270},
	}, fg)
	b := iconDots(360, 360, 40, [][2]int{
		{90, 60}, {270, 60},
		{90, 180}, {270, 180},
		{90, 300}, {270, 300},
	}, fg)
	c := iconDots(360, 360, 40, [][2]int{
		{90, 90}, {180, 180}, {270, 270},
	}, fg)

	ix := New()
	ix.AddImage("a.png", hashRegions(a, cfg))
	ix.AddImage("b.png", hashRegions(b, cfg))
	ix.AddImage("c.png", hashRegions(c, cfg))

	query := make([]QueryRegion, 0)
	for _, h := range hashRegions(a, cfg) {
		query = append(query, QueryRegion{
			Hash: h.Hash, Area: h.Area, Color: h.Color,
			NX: h.NX, NY: h.NY, Fill: h.Fill, Aspect: h.Aspect,
		})
	}

	matches := ix.Search(query, SearchOptions{MaxDist: 12})
	if len(matches) == 0 {
		t.Fatal("无匹配结果")
	}
	if matches[0].ImageID != "a.png" {
		t.Fatalf("期望 top1=a.png，得到 %s", matches[0].ImageID)
	}
	scores := map[string]float64{}
	for _, m := range matches {
		scores[m.ImageID] = m.Score
	}
	if scores["a.png"] <= scores["b.png"] {
		t.Fatalf("布局一致性未生效: a=%f b=%f", scores["a.png"], scores["b.png"])
	}
	if scores["a.png"] <= scores["c.png"] {
		t.Fatalf("数量(一对一)未生效: a=%f c=%f", scores["a.png"], scores["c.png"])
	}
}

// bruteForceAssign 穷举矩形分配（未匹配行计 pad 成本），作为 Hungarian 的参照。
func bruteForceAssign(cost [][]float64, pad float64) ([]int, float64) {
	r, c := len(cost), len(cost[0])
	best := math.Inf(1)
	var bestAssign []int
	colsUsed := make([]bool, c)
	var rec func(i int, sum float64, cur []int)
	rec = func(i int, sum float64, cur []int) {
		if i == r {
			if sum < best {
				best = sum
				bestAssign = append([]int(nil), cur...)
			}
			return
		}
		for j := 0; j < c; j++ {
			if colsUsed[j] || math.IsInf(cost[i][j], 1) {
				continue
			}
			colsUsed[j] = true
			rec(i+1, sum+cost[i][j], append(cur, j))
			colsUsed[j] = false
		}
		rec(i+1, sum+pad, append(cur, -1))
	}
	rec(0, 0, make([]int, 0, r))
	if math.IsInf(best, 1) {
		return nil, best
	}
	return bestAssign, best
}

func TestHungarian(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for iter := 0; iter < 300; iter++ {
		r := rng.Intn(5) + 1
		c := rng.Intn(5) + 1
		cost := make([][]float64, r)
		for i := range cost {
			cost[i] = make([]float64, c)
			for j := range cost[i] {
				if rng.Float64() < 0.3 {
					cost[i][j] = math.Inf(1)
				} else {
					cost[i][j] = float64(rng.Intn(50))
				}
			}
		}
		got := hungarian(cost)
		wantAssign, wantTotal := bruteForceAssign(cost, hungarianPad)

		gotTotal := 0.0
		for i, j := range got {
			if j >= 0 {
				gotTotal += cost[i][j]
			} else {
				gotTotal += hungarianPad
			}
		}
		if math.Abs(gotTotal-wantTotal) > 1e-6 {
			t.Fatalf("iter=%d 成本不一致: got=%v total=%f want=%v total=%f",
				iter, got, gotTotal, wantAssign, wantTotal)
		}
	}
}

// TestBuildParallelDeterministic 验证并行构建（BuildParallel）与串行构建产生完全
// 一致的索引：条目顺序、分段表、跳过行为与保存结果均相同，确保并发改造不改变
// 检索结果与索引文件内容。同时验证进度回调按输入顺序且覆盖全部文件。
func TestBuildParallelDeterministic(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	files := make([]string, 60)
	gen := make(map[string][]RegionHash, len(files))
	for i := range files {
		f := fmt.Sprintf("img%02d.png", i)
		files[i] = f
		n := rng.Intn(6) + 1
		regs := make([]RegionHash, n)
		for j := range regs {
			regs[j] = mkRegionHash(j+1, rng.Intn(500)+10, image.Rect(0, 0, 12, 12),
				color.RGBA{uint8(rng.Intn(256)), uint8(rng.Intn(256)), uint8(rng.Intn(256)), 255})
		}
		gen[f] = regs
	}
	gen["img07.png"] = []RegionHash{} // 空区域：应跳过
	skipErr := errors.New("boom")     // 处理失败：img13.png 应跳过

	proc := func(f string) (string, []RegionHash, error) {
		if f == "img13.png" {
			return "", nil, skipErr
		}
		regs, ok := gen[f]
		if !ok {
			return "", nil, errors.New("missing")
		}
		return f, regs, nil
	}

	serial := New()
	gotSerial := serial.BuildParallel(files, 1, proc, nil)

	parallel := New()
	progress := make([]BuildProgress, 0, len(files))
	gotParallel := parallel.BuildParallel(files, 0, proc, func(p BuildProgress) {
		progress = append(progress, p)
	})

	if gotSerial != gotParallel {
		t.Fatalf("入索引文件数不一致: serial=%d parallel=%d", gotSerial, gotParallel)
	}
	if gotParallel != len(files)-2 {
		t.Fatalf("跳过逻辑错误: 期望 %d 个文件入索引，得到 %d", len(files)-2, gotParallel)
	}
	if len(progress) != len(files) {
		t.Fatalf("进度回调次数错误: %d", len(progress))
	}
	for i, p := range progress {
		if p.Done != i+1 {
			t.Fatalf("进度未按输入顺序回调: idx=%d Done=%d", i, p.Done)
		}
		if files[i] == "img13.png" {
			if p.Err == nil {
				t.Fatalf("img13.png 应报错跳过: %+v", p)
			}
			continue
		}
		if files[i] == "img07.png" {
			if p.Err != nil || p.ID != "" || p.Regions != 0 {
				t.Fatalf("img07.png 应空区域跳过: %+v", p)
			}
			continue
		}
		if p.Err != nil {
			t.Fatalf("idx=%d 不应报错: %v", i, p.Err)
		}
		if p.ID != files[i] || p.Regions != len(gen[files[i]]) {
			t.Fatalf("idx=%d 进度信息不符: id=%s regions=%d", i, p.ID, p.Regions)
		}
	}

	if len(serial.Entries) != len(parallel.Entries) || len(serial.Images) != len(parallel.Images) {
		t.Fatalf("索引规模不一致: %d/%d vs %d/%d",
			len(serial.Entries), len(serial.Images), len(parallel.Entries), len(parallel.Images))
	}
	for i := range serial.Entries {
		if serial.Entries[i] != parallel.Entries[i] {
			t.Fatalf("第 %d 条区域记录不一致", i)
		}
	}
	for s := range serial.Segments {
		if !reflect.DeepEqual(serial.Segments[s], parallel.Segments[s]) {
			t.Fatalf("Segments[%d] 不一致", s)
		}
		if !reflect.DeepEqual(serial.Shapes[s], parallel.Shapes[s]) {
			t.Fatalf("Shapes[%d] 不一致", s)
		}
	}

	// gob 对 map 的编码顺序不确定，因此不比较文件字节，而是 Load 回来后比较结构，
	// 确认并行构建的索引文件可完整还原（内容与串行一致）。
	sp := filepath.Join(t.TempDir(), "idx.bin")
	if err := parallel.Save(sp); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(sp)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(loaded.Entries, parallel.Entries) {
		t.Fatal("Load 后区域条目不一致")
	}
	for s := range loaded.Segments {
		if !reflect.DeepEqual(loaded.Segments[s], parallel.Segments[s]) ||
			!reflect.DeepEqual(loaded.Shapes[s], parallel.Shapes[s]) {
			t.Fatalf("Load 后分段表[%d]不一致", s)
		}
	}
}

// TestBuildParallelEmpty 验证空输入与单文件输入不 panic 且行为正确。
func TestBuildParallelEmpty(t *testing.T) {
	if n := New().BuildParallel(nil, 0, func(f string) (string, []RegionHash, error) { return f, nil, nil }, nil); n != 0 {
		t.Fatalf("空输入应返回 0，得到 %d", n)
	}
	ix := New()
	if n := ix.BuildParallel([]string{"a.png"}, 0, func(f string) (string, []RegionHash, error) {
		return f, []RegionHash{{RegionID: 1, Hash: 1, Area: 1}}, nil
	}, nil); n != 1 {
		t.Fatalf("单文件应返回 1，得到 %d", n)
	}
	if ix.Len() != 1 || !ix.Images["a.png"] {
		t.Fatal("单文件入索引失败")
	}
}

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}
