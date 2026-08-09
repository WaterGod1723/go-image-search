package main

import (
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type manifestEntry struct {
	Image     string    `json:"image"`
	Src       string    `json:"src"`
	Crop      [4]int    `json:"crop"`
	Canvas    [2]int    `json:"canvas"`
	BGHex     string    `json:"bg_hex"`
	Rotation  float64   `json:"rotation"`
	Scale     float64   `json:"scale"`
	Translate [2]int    `json:"translate"`
	Texts     []TextInfo `json:"texts"`
}

type TextInfo struct {
	Side  string `json:"side"`
	Text  string `json:"text"`
	Size  int    `json:"size"`
	X     int    `json:"x"`
	Y     int    `json:"y"`
	Color string `json:"color"`
}

func dumpMask(m []float64) {
	for y := 0; y < maskN; y += 2 {
		line := make([]byte, maskN)
		for x := 0; x < maskN; x++ {
			if m[y*maskN+x] > 0.5 {
				line[x] = '#'
			} else if m[y*maskN+x] > 0.1 {
				line[x] = '.'
			} else {
				line[x] = ' '
			}
		}
		fmt.Printf("   [%s]\n", string(line))
	}
}

func insertionSortInts(arr []int, less func(a, b int) bool) {
	for i := 1; i < len(arr); i++ {
		for j := i; j > 0 && less(arr[j], arr[j-1]); j-- {
			arr[j], arr[j-1] = arr[j-1], arr[j]
		}
	}
}

func main() {
	root := "."
	if len(os.Args) > 1 {
		root = os.Args[1]
	}
	if len(os.Args) > 2 {
		switch os.Args[2] {
		case "diag":
			runDiag(root)
			return
		case "palette":
			runPalette(root)
			return
		case "space":
			runSpaceDiag(root)
			return
		case "hist":
			runEvalHist(root)
			return
		case "inv":
			runInvTest(root)
			return
		case "server":
			runServer(root, os.Args[3:])
			return
		case "train":
			runTrain(root, os.Args[3:])
			return
		case "nn":
			runNNEval(root, os.Args[3:])
			return
		case "seg":
			runSegDump(root, os.Args[3:])
			return
		case "rr":
			runRankReport(root, os.Args[3:])
			return
		case "sczl":
			runSCZLEval(root, os.Args[3:])
			return
		}
	}
	if len(os.Args) > 2 && os.Args[2] == "adp" {
		runEvalMode(root, true)
		return
	}
	if len(os.Args) > 2 && os.Args[2] == "fixed" {
		runEvalMode(root, false)
		return
	}
	dumpDir := ""
	if len(os.Args) > 3 {
		dumpDir = os.Args[3]
	}
	if len(os.Args) > 2 && os.Args[2] == "d" {
		runEvalModeDump(root, dumpDir)
		return
	}
	if len(os.Args) > 2 && os.Args[2] == "fixed" {
		runEvalMode(root, false)
		return
	}
	runEvalMode(root, true)
}

func runEvalModeDump(root string, dumpDir string) {
	runEvalModeBody(root, false, dumpDir)
}

func runEvalMode(root string, adaptive bool) {
	runEvalModeBody(root, adaptive, "")
}

func runEvalModeBody(root string, adaptive bool, dumpDir string) {
	srcDir := filepath.Join(root, "test_pngs")
	sampleDir := filepath.Join(root, "test_set")

	// 1) build reference index from source icons
	files, err := listPNG(srcDir)
	if err != nil {
		fatal(err)
	}
	var refs []*Feat
	var refNames []string
	for _, fn := range files {
		img, err := loadPNG(filepath.Join(srcDir, fn))
		if err != nil {
			fatal(err)
		}
		px := refPixels(img)
		if len(px) == 0 {
			continue
		}
		refs = append(refs, buildFeat(px))
		refNames = append(refNames, fn)
	}
	fmt.Printf("index: %d reference sprites\n", len(refs))

	// 2) load ground truth
	data, err := os.ReadFile(filepath.Join(sampleDir, "manifest.json"))
	if err != nil {
		fatal(err)
	}
	var entries []manifestEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		fatal(err)
	}
	gt := map[string]string{}
	for _, e := range entries {
		gt[e.Image] = e.Src
	}

	// 3) evaluate
	hits1, hits5 := 0, 0
	type miss struct {
		img, want string
		got       []string
	}
	var misses []miss
	type stat struct {
		name  string
		ok1   int
		count int
	}
	stMap := map[string]*stat{}

	for i := 0; i < len(entries); i++ {
		e := entries[i]
		img, err := loadPNG(filepath.Join(sampleDir, e.Image))
		if err != nil {
			fatal(err)
		}
		px := extractQuery(img)
		if len(px) == 0 {
			fmt.Printf("WARN: empty segmentation for %s\n", e.Image)
			continue
		}
		q := buildFeat(px)
		var res []int
		if adaptive {
			res = compositeAdaptive(q, refs)
		} else {
			w := defaultWeights(q.Mono)
			res = composite(q, refs, w)
		}
		want := e.Src

if dumpDir != "" && strings.Contains(e.Image, "00004") {
			fmt.Printf(">> %s mono=%v want=%s\n", e.Image, e.Src, e.Src)
			for _, idx := range res[:3] {
				s := scores(q, refs[idx])
				fmt.Printf("   %-40s hist=%.2f rad=%.2f ang=%.2f shp=%.2f col=%.2f zern=%.2f dice=%.2f\n",
					refNames[idx], s[0], s[1], s[2], s[3], s[4], s[6], maskScore(q, refs[idx]))
			}
			wi := -1
			for i, n := range refNames {
				if n == want {
					wi = i
				}
			}
			s := scores(q, refs[wi])
			fmt.Printf("   want: %s hist=%.2f shp=%.2f zern=%.2f dice=%.2f\n", want, s[0], s[3], s[6], maskScore(q, refs[wi]))
		}

		rank1 := res[0]
		got1 := refNames[rank1]
		if got1 == want {
			hits1++
		}
		in5 := false
		var got5 []string
		for _, r := range res[:min(5, len(res))] {
			got5 = append(got5, refNames[r])
			if refNames[r] == want {
				in5 = true
			}
		}
		if in5 {
			hits5++
		}
		if got1 != want {
			misses = append(misses, miss{e.Image, want, got5})
		}
		if st, ok := stMap[want]; ok {
			st.count++
			if got1 == want {
				st.ok1++
			}
		} else {
			stMap[want] = &stat{name: want, count: 1, ok1: boolToInt(got1 == want)}
		}
	}

	total := len(entries)
	fmt.Printf("recall@1: %d/%d = %.1f%%\n", hits1, total, 100*float64(hits1)/float64(total))
	fmt.Printf("recall@5: %d/%d = %.1f%%\n", hits5, total, 100*float64(hits5)/float64(total))

	if len(misses) > 0 {
		fmt.Printf("\n-- misses (%d) --\n", len(misses))
		for _, m := range misses {
			fmt.Printf("%-18s want=%-60s got=%v\n", m.img, m.want, m.got)
		}
	}

	// per-sprite accuracy table
	var stats []*stat
	for _, st := range stMap {
		stats = append(stats, st)
	}
	sort.Slice(stats, func(i, j int) bool { return stats[i].name < stats[j].name })
	fmt.Printf("\n-- per-sprite @1 --\n")
	for _, st := range stats {
		acc := 100 * float64(st.ok1) / float64(st.count)
		flag := " "
		if float64(st.ok1) < float64(st.count) {
			flag = "*"
		}
		fmt.Printf("%s %-72s %d/%d (%.0f%%)\n", flag, st.name, st.ok1, st.count, acc)
	}
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func listPNG(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if strings.HasSuffix(strings.ToLower(e.Name()), ".png") {
			out = append(out, e.Name())
		}
	}
	sort.Strings(out)
	return out, nil
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// dumpSegmentation writes the original query and the extracted sprite (green
// overlay on a black background) for visual QA.
func dumpSegmentation(dir, name string, canvas *image.NRGBA, px []Px) {
	w, h := canvas.Bounds().Dx(), canvas.Bounds().Dy()
	left := image.NewNRGBA(image.Rect(0, 0, w, h))
	right := image.NewNRGBA(image.Rect(0, 0, w, h))
	move := make([]bool, w*h)
	for _, p := range px {
		if p.Y < h && p.X < w {
			move[p.Y*w+p.X] = true
		}
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			s := (y*w + x) * 4
			left.Pix[s], left.Pix[s+1], left.Pix[s+2], left.Pix[s+3] = canvas.Pix[s], canvas.Pix[s+1], canvas.Pix[s+2], 255
			if move[y*w+x] {
				right.Pix[s], right.Pix[s+1], right.Pix[s+2], right.Pix[s+3] = 0, 255, 0, 255
			} else {
				right.Pix[s], right.Pix[s+1], right.Pix[s+2], right.Pix[s+3] = 0, 0, 0, 255
			}
		}
	}
	combo := image.NewNRGBA(image.Rect(0, 0, w*2, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			s := (y*w + x) * 4
			o := (y*(w*2) + x) * 4
			combo.Pix[o], combo.Pix[o+1], combo.Pix[o+2], combo.Pix[o+3] = left.Pix[s], left.Pix[s+1], left.Pix[s+2], left.Pix[s+3]
			o2 := (y*(w*2) + w + x) * 4
			combo.Pix[o2], combo.Pix[o2+1], combo.Pix[o2+2], combo.Pix[o2+3] = right.Pix[s], right.Pix[s+1], right.Pix[s+2], right.Pix[s+3]
		}
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return
	}
	f, err := os.Create(filepath.Join(dir, name))
	if err != nil {
		return
	}
	defer f.Close()
	_ = png.Encode(f, combo)
}
