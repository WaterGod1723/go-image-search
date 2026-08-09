package main

import (
	"fmt"
	"path/filepath"

	"image-search-test/search/sczl"
)

// runSCZLEval evaluates the sczl (color-agnostic occupancy + Fourier + SC +
// HOG) retrieval algorithm on this project's test_set, split by whether the
// query sprite is monochrome. It is the candidate "dedicated grayscale
// branch": it ignores color entirely and discriminates by internal structure,
// so it should do well exactly where the histogram ties (gray icons).
func runSCZLEval(root string, args []string) {
	srcDir := filepath.Join(root, "test_pngs")
	files, err := listPNG(srcDir)
	if err != nil {
		fatal(err)
	}
	ix := sczl.New()
	for _, fn := range files {
		img, err := loadPNG(filepath.Join(srcDir, fn))
		if err != nil {
			fatal(err)
		}
		d := sczl.Extract(img)
		if !d.Valid {
			continue
		}
		ix.AddImage(fn, d)
	}
	fmt.Printf("sczl index: %d refs\n", ix.Len())

	entries := loadManifest(filepath.Join(root, "test_set"))
	opts := sczl.DefaultOptions()
	opts.TopK = 10

	hit1, hit5, total := 0, 0, 0
	monoN, colorN := 0, 0
	mono1, mono5, color1, color5 := 0, 0, 0, 0
	type miss struct{ img, want string; got []string }
	var misses []miss
	for _, e := range entries {
		img, err := loadPNG(filepath.Join(root, "test_set", e.Image))
		if err != nil {
			fatal(err)
		}
		px := extractQuery(img)
		if len(px) == 0 {
			continue
		}
		q := buildFeat(px)
		qd := sczl.Extract(img)
		if !qd.Valid {
			continue
		}
		total++
		if q.Mono {
			monoN++
		} else {
			colorN++
		}

		matches := ix.Query(qd, opts)
		var got []string
		in5 := false
		for i, m := range matches {
			if i == 0 {
				if m.ImageID == e.Src {
					hit1++
					if q.Mono {
						mono1++
					} else {
						color1++
					}
				}
			}
			if i < 5 {
				got = append(got, m.ImageID)
				if m.ImageID == e.Src {
					in5 = true
				}
			}
		}
		if in5 {
			hit5++
			if q.Mono {
				mono5++
			} else {
				color5++
			}
		}
		if len(matches) == 0 || matches[0].ImageID != e.Src {
			misses = append(misses, miss{e.Image, e.Src, got})
		}
	}
	fmt.Printf("sczl recall@1 = %d/%d (%.1f%%)   recall@5 = %d/%d (%.1f%%)\n",
		hit1, total, 100*float64(hit1)/float64(total), hit5, total, 100*float64(hit5)/float64(total))
	fmt.Printf("  mono : %d/%d (%.1f%%)  @5 %d/%d\n", mono1, monoN, pct(mono1, monoN), mono5, monoN)
	fmt.Printf("  color: %d/%d (%.1f%%)  @5 %d/%d\n", color1, colorN, pct(color1, colorN), color5, colorN)
	if len(misses) > 0 {
		fmt.Printf("\n-- sczl misses (%d) --\n", len(misses))
		for _, m := range misses {
			fmt.Printf("%-18s want=%-60s got=%v\n", m.img, m.want, m.got)
		}
	}
}

func pct(a, b int) float64 {
	if b <= 0 {
		return 0
	}
	return 100 * float64(a) / float64(b)
}
