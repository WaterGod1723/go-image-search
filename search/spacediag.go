package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// runEvalHist evaluates how often the true match is top by histogram alone.
func runEvalHist(root string) {
	srcDir := filepath.Join(root, "test_pngs")
	sampleDir := filepath.Join(root, "test_set")
	refs, names := indexRefs(srcDir)
	entries := loadManifest(sampleDir)
	hits := 0
	for _, e := range entries {
		img, err := loadPNG(filepath.Join(sampleDir, e.Image))
		if err != nil {
			fatal(err)
		}
		px := extractQuery(img)
		if len(px) == 0 {
			continue
		}
		q := buildFeat(px)
		best, bestI := -1.0, -1
		hs := make([]float64, len(refs))
		for i, r := range refs {
			hs[i] = histSim(q.Hist, r.Hist)
			if hs[i] > best {
				best, bestI = hs[i], i
			}
		}
		if names[bestI] == e.Src {
			hits++
		} else {
			fmt.Printf("%s want=%s got=%s(h=%.3f) trueRank=%d\n", e.Image, e.Src, names[bestI], best, rankOf(hs, hs[idxOf(names, e.Src)]))
		}
	}
	fmt.Printf("hist-only recall@1: %d/%d = %.1f%%\n", hits, len(entries), 100*float64(hits)/float64(len(entries)))
}

func idxOf(names []string, s string) int {
	for i, n := range names {
		if n == s {
			return i
		}
	}
	return -1
}

func rankOf(v []float64, x float64) int {
	r := 1
	for _, z := range v {
		if z > x {
			r++
		}
	}
	return r
}

func indexRefs(srcDir string) ([]*Feat, []string) {
	var refs []*Feat
	var names []string
	files, err := listPNG(srcDir)
	if err != nil {
		fatal(err)
	}
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
		names = append(names, fn)
	}
	return refs, names
}

func loadManifest(dir string) []manifestEntry {
	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		fatal(err)
	}
	var es []manifestEntry
	if err := json.Unmarshal(data, &es); err != nil {
		fatal(err)
	}
	return es
}

// runSpaceDiag reports, per feature space, the rank of the true match for each
// query (rank 1 = that space alone would have retrieved it).
func runSpaceDiag(root string) {
	srcDir := filepath.Join(root, "test_pngs")
	sampleDir := filepath.Join(root, "test_set")

	files := mustList(srcDir)
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

	data, err := os.ReadFile(filepath.Join(sampleDir, "manifest.json"))
	if err != nil {
		fatal(err)
	}
	var entries []DiagEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		fatal(err)
	}

	names := []string{"hist", "radial", "angmag", "shape", "color", "zern", "geom"}
	rankHist := make([][]int, 7)
	total := 0
	for i := range rankHist {
		rankHist[i] = make([]int, len(refs)+1)
	}
	monoN, colorN := 0, 0
	colorHits := make([]int, 7)
	monoHits := make([]int, 7)

	for _, e := range entries {
		img, err := loadPNG(filepath.Join(sampleDir, e.Image))
		if err != nil {
			fatal(err)
		}
		px := extractQuery(img)
		if len(px) == 0 {
			fmt.Printf("skip(empty seg) %s\n", e.Image)
			continue
		}
		q := buildFeat(px)
		wantIdx := -1
		for i := range refNames {
			if refNames[i] == e.Src {
				wantIdx = i
				break
			}
		}
		if wantIdx < 0 {
			continue
		}

		sAll := make([][]float64, len(refs))
		for i := range refs {
			sAll[i] = scores(q, refs[i])
		}
		trueRow := sAll[wantIdx]
		for sp := 0; sp < 7; sp++ {
			r := 1
			for i := 0; i < len(refs); i++ {
				if i != wantIdx && sAll[i][sp] > trueRow[sp] {
					r++
				}
			}
			rankHist[sp][r]++
			if r == 1 {
				if q.Mono {
					monoHits[sp]++
				} else {
					colorHits[sp]++
				}
			}
		}
		if q.Mono {
			monoN++
		} else {
			colorN++
		}
		total++
	}

	fmt.Printf("queries evaluated: %d (mono=%d color=%d) refs=%d\n", total, monoN, colorN, len(refs))
	for sp := 0; sp < 7; sp++ {
		var r1, r3 int
		for r := 1; r < len(rankHist[sp]); r++ {
			if r == 1 {
				r1 += rankHist[sp][r]
			}
			if r <= 3 {
				r3 += rankHist[sp][r]
			}
		}
		fmt.Printf("space %-7s rank1=%4d/%-4d (%.0f%%)  rank<=3=%4d  [mono@1=%d color@1=%d]\n",
			names[sp], r1, total, 100*float64(r1)/float64(total), r3, monoHits[sp], colorHits[sp])
	}
}

var _diagConst = 0
