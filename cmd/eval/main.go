// Command eval evaluates the SCZL image-retrieval algorithm on either:
//
//   - mode=manifest: the generated test_set with manifest.json (rotation-heavy)
//   - mode=target: the legacy test_pngs_target/ directory (filename encodes the
//     target as TEST<N>_FROM_<source>.png), used for regression checks.
//
// 用法:
//
//	# 新测试集（gentest 生成，含旋转）
//	go run ./cmd/eval -mode manifest -lib test_pngs -query test_set
//
//	# 旧测试集（回归用）
//	go run ./cmd/eval -mode target -lib test_pngs -query test_pngs_target
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"

	"go-image-search/internal/imageproc"
	"go-image-search/internal/sczl"
)

var (
	mode     = flag.String("mode", "manifest", "评估模式: manifest 或 target")
	libDir   = flag.String("lib", "test_pngs", "索引图库目录")
	queryDir = flag.String("query", "test_set", "查询目录 (manifest模式=测试集根, target模式=查询图目录)")
	topK     = flag.Int("top", 10, "检索返回 top-N，判断 recall@1/3/5 时用")
	adaptive = flag.Bool("adaptive", true, "启用自适应权重 (true/false)")
)

type manifestEntry struct {
	Image    string  `json:"image"`
	Src      string  `json:"src"`
	Rotation float64 `json:"rotation"`
}

type missRow struct {
	query string
	want  string
	got   []string // top-5
	rank  int      // want 实际排名 (-1 = 未命中)
	rot   float64  // manifest 模式下的旋转角度
}

type spriteStat struct {
	name  string
	ok1   int
	count int
}

func main() {
	flag.Parse()
	if *mode != "manifest" && *mode != "target" {
		fmt.Fprintln(os.Stderr, "-mode 必须是 manifest 或 target")
		os.Exit(2)
	}

	// 1. 构建索引库
	files, err := imageproc.LoadSupported(*libDir)
	if err != nil {
		fatal("扫描索引库失败: %v", err)
	}
	if len(files) == 0 {
		fatal("索引库 %s 中无支持的图片", *libDir)
	}
	ix := sczl.New()
	skipped := 0
	for _, f := range files {
		img, err := imageproc.Load(f)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[跳过] 无法解码 %s: %v\n", f, err)
			skipped++
			continue
		}
		d := sczl.Extract(img)
		if !d.Valid {
			fmt.Fprintf(os.Stderr, "[跳过] 无效描述子: %s\n", f)
			skipped++
			continue
		}
		ix.AddImage(f, d)
	}
	fmt.Printf("index: %d reference sprites (skipped %d)\n", ix.Len(), skipped)
	if ix.Len() == 0 {
		fatal("索引为空")
	}

	opts := sczl.DefaultOptions()
	opts.Adaptive = *adaptive
	opts.TopK = *topK

	switch *mode {
	case "manifest":
		runManifest(ix, opts)
	case "target":
		runTarget(ix, opts)
	}
}

// queryResult 单个查询的并行结果。
type queryResult struct {
	query    string
	want     string
	rot      float64
	gotRank  int
	topNames []string
	valid    bool
}

// runQueriesParallel 并行执行 Extract + Query，结果存预分配数组无竞争。
func runQueriesParallel(ix *sczl.Index, opts sczl.Options, items []queryResult) {
	n := len(items)
	workers := runtime.NumCPU()
	if workers > n {
		workers = n
	}
	if workers < 1 {
		workers = 1
	}
	var wg sync.WaitGroup
	chunk := (n + workers - 1) / workers
	for w := 0; w < workers; w++ {
		s := w * chunk
		e := s + chunk
		if e > n {
			e = n
		}
		if s >= e {
			continue
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				qpath := filepath.Join(*queryDir, items[i].query)
				img, err := imageproc.Load(qpath)
				if err != nil {
					continue
				}
				qd := sczl.Extract(img)
				if !qd.Valid {
					continue
				}
				matches := ix.Query(qd, opts)
				items[i].gotRank = findRank(matches, items[i].want)
				items[i].topNames = topNames(matches, 5)
				items[i].valid = true
			}
		}(s, e)
	}
	wg.Wait()
}

// -------------------- manifest 模式 --------------------

func runManifest(ix *sczl.Index, opts sczl.Options) {
	data, err := os.ReadFile(filepath.Join(*queryDir, "manifest.json"))
	if err != nil {
		fatal("读取 manifest.json 失败: %v", err)
	}
	var entries []manifestEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		fatal("解析 manifest.json 失败: %v", err)
	}
	if len(entries) == 0 {
		fatal("manifest 为空")
	}

	items := make([]queryResult, len(entries))
	for i, e := range entries {
		items[i] = queryResult{query: e.Image, want: e.Src, rot: e.Rotation}
	}
	runQueriesParallel(ix, opts, items)

	hits1, hits3, hits5 := 0, 0, 0
	total := 0
	var misses []missRow
	stMap := map[string]*spriteStat{}
	for _, r := range items {
		if !r.valid {
			continue
		}
		total++
		if r.gotRank == 0 {
			hits1++
			hits3++
			hits5++
		} else if r.gotRank > 0 && r.gotRank < 3 {
			hits3++
			hits5++
		} else if r.gotRank > 0 && r.gotRank < 5 {
			hits5++
		}
		if r.gotRank < 0 || r.gotRank >= 1 {
			misses = append(misses, missRow{
				query: r.query, want: r.want, got: r.topNames, rank: r.gotRank, rot: r.rot,
			})
		}
		if st, ok := stMap[r.want]; ok {
			st.count++
			if r.gotRank == 0 {
				st.ok1++
			}
		} else {
			stMap[r.want] = &spriteStat{name: r.want, count: 1, ok1: boolToInt(r.gotRank == 0)}
		}
	}
	printStats(hits1, hits3, hits5, total, misses, stMap, true)
}

// -------------------- target 模式（回归） --------------------

func runTarget(ix *sczl.Index, opts sczl.Options) {
	qfiles, err := filepath.Glob(filepath.Join(*queryDir, "*.png"))
	if err != nil || len(qfiles) == 0 {
		fatal("查询目录 %s 中无 png", *queryDir)
	}
	sort.Strings(qfiles)

	re := regexp.MustCompile(`^TEST\d+_FROM_(.+?)\.png$`)
	var items []queryResult
	for _, qf := range qfiles {
		base := filepath.Base(qf)
		m := re.FindStringSubmatch(base)
		if m == nil {
			fmt.Fprintf(os.Stderr, "[跳过] 无法从文件名解析期望目标: %s\n", base)
			continue
		}
		want := normalizeLibName(m[1]) + ".png"
		items = append(items, queryResult{query: base, want: want})
	}
	if len(items) == 0 {
		fatal("未解析到任何测试用例")
	}
	runQueriesParallel(ix, opts, items)

	hits1, hits3, hits5 := 0, 0, 0
	total := 0
	var misses []missRow
	stMap := map[string]*spriteStat{}
	for _, r := range items {
		if !r.valid {
			misses = append(misses, missRow{query: r.query, want: r.want, got: nil, rank: -1})
			continue
		}
		total++
		if r.gotRank == 0 {
			hits1++
			hits3++
			hits5++
		} else if r.gotRank > 0 && r.gotRank < 3 {
			hits3++
			hits5++
		} else if r.gotRank > 0 && r.gotRank < 5 {
			hits5++
		}
		if r.gotRank < 0 || r.gotRank >= 1 {
			misses = append(misses, missRow{
				query: r.query, want: r.want, got: r.topNames, rank: r.gotRank,
			})
		}
		if st, ok := stMap[r.want]; ok {
			st.count++
			if r.gotRank == 0 {
				st.ok1++
			}
		} else {
			stMap[r.want] = &spriteStat{name: r.want, count: 1, ok1: boolToInt(r.gotRank == 0)}
		}
	}
	printStats(hits1, hits3, hits5, total, misses, stMap, false)
}

// -------------------- 通用辅助 --------------------

// findRank 返回 want 出现在 matches 的第几名 (0 基), -1 = 未命中.
// ImageID 是完整路径；匹配只用文件名.
func findRank(matches []sczl.Match, wantBase string) int {
	for i, m := range matches {
		if filepath.Base(m.ImageID) == wantBase {
			return i
		}
	}
	return -1
}

func topNames(matches []sczl.Match, n int) []string {
	if n > len(matches) {
		n = len(matches)
	}
	out := make([]string, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, filepath.Base(matches[i].ImageID))
	}
	return out
}

func normalizeLibName(name string) string {
	return strings.TrimLeft(name, "._")
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func printStats(hits1, hits3, hits5, total int, misses []missRow, stMap map[string]*spriteStat, withRot bool) {
	fmt.Printf("recall@1: %d/%d = %.1f%%\n", hits1, total, 100*float64(hits1)/float64(total))
	fmt.Printf("recall@3: %d/%d = %.1f%%\n", hits3, total, 100*float64(hits3)/float64(total))
	fmt.Printf("recall@5: %d/%d = %.1f%%\n", hits5, total, 100*float64(hits5)/float64(total))

	if len(misses) > 0 {
		fmt.Printf("\n-- misses (%d) --\n", len(misses))
		for _, m := range misses {
			tag := ""
			if withRot {
				tag = fmt.Sprintf(" rot=%6.1f°", m.rot)
			}
			rankStr := ""
			if m.rank >= 0 {
				rankStr = fmt.Sprintf(" (want 排 #%d)", m.rank+1)
			}
			fmt.Printf("%-18s%s want=%-60s got=%v%s\n",
				m.query, tag, m.want, m.got, rankStr)
		}
	}

	var stats []*spriteStat
	for _, st := range stMap {
		stats = append(stats, st)
	}
	sort.Slice(stats, func(i, j int) bool { return stats[i].name < stats[j].name })
	fmt.Printf("\n-- per-sprite @1 --\n")
	for _, st := range stats {
		acc := 100 * float64(st.ok1) / float64(st.count)
		mark := " "
		if float64(st.ok1) < float64(st.count) {
			mark = "*"
		}
		fmt.Printf("%s %-72s %d/%d (%.0f%%)\n", mark, st.name, st.ok1, st.count, acc)
	}
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "error: "+format+"\n", args...)
	os.Exit(1)
}
