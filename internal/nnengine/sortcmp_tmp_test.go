package nnengine

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"go-image-search/internal/imageproc"
	"go-image-search/internal/nnengine/sczl"
)

type mfEntry struct {
	Image string `json:"image"`
	Src   string `json:"src"`
}

func TestCompareSortingTmp(t *testing.T) {
	root := "../../"
	e := New(root + "weights_split.gob")
	if e.nn == nil {
		t.Fatal("weights not loaded")
	}
	cache := filepath.Join(os.TempDir(), "dev-best-v2-index.bin")
	if err := e.BuildIndex(root+"test_pngs", cache); err != nil {
		t.Fatalf("BuildIndex: %v", err)
	}
	raw, _ := os.ReadFile(root + "test_set/manifest.json")
	var mf []mfEntry
	json.Unmarshal(raw, &mf)

	refs := e.refs
	names := e.names

	count := func(rankFn func(q *Feat, qd sczl.Descriptor) []int) (int, int, int) {
		at1, at3, at5 := 0, 0, 0
		for _, e0 := range mf {
			img, err := imageproc.Load(filepath.Join(root, "test_set", e0.Image))
			if err != nil {
				continue
			}
			px := extractQuery(toNRGBA(img))
			if len(px) == 0 {
				continue
			}
			q := buildFeat(px)
			qd := sczl.Extract(toNRGBA(img))
			ranked := rankFn(q, qd)
			want := filepath.Base(e0.Src)
			for i, ridx := range ranked {
				if filepath.Base(names[ridx]) == want {
					if i == 0 {
						at1++
					}
					if i < 3 {
						at3++
					}
					if i < 5 {
						at5++
					}
					break
				}
			}
		}
		return at1, at3, at5
	}

	// 1) 纯 hist 排序
	rankByHist := func(q *Feat, qd sczl.Descriptor) []int {
		order := make([]int, len(refs))
		for i := range order {
			order[i] = i
		}
		insertionSort(order, func(a, b int) bool { return histSim(q.Hist, refs[a].Hist) > histSim(q.Hist, refs[b].Hist) })
		return order
	}

	h1, h3, h5 := count(rankByHist)
	var out string
	out += fmt.Sprintf("纯hist排序          recall@1=%d/%d recall@3=%d/%d recall@5=%d/%d\n", h1, len(mf), h3, len(mf), h5, len(mf))

	// e.Search 走 rankNN（当前纯 score）
	at1e, at3e, at5e := 0, 0, 0
	for _, e0 := range mf {
		img, _ := imageproc.Load(filepath.Join(root, "test_set", e0.Image))
		hits, err := e.Search(img, 5)
		if err != nil {
			continue
		}
		for i, h := range hits {
			if filepath.Base(h.Name) == filepath.Base(e0.Src) {
				if i == 0 {
					at1e++
				}
				if i < 3 {
					at3e++
				}
				if i < 5 {
					at5e++
				}
				break
			}
		}
	}
	out += fmt.Sprintf("rankNN纯score(现)    recall@1=%d/%d recall@3=%d/%d recall@5=%d/%d\n", at1e, len(mf), at3e, len(mf), at5e, len(mf))
	os.WriteFile("/tmp/sortcompare.txt", []byte(out), 0o644)
}