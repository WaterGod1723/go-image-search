package main

import (
	"fmt"
	"path/filepath"
	"sort"
)

// runPalette dumps a per-reference summary (size, mono flag, top colors).
func runPalette(root string) {
	srcDir := filepath.Join(root, "test_pngs")
	files := mustList(srcDir)

	for _, fn := range files {
		img, err := loadPNG(filepath.Join(srcDir, fn))
		if err != nil {
			fatal(err)
		}
		px := refPixels(img)
		if len(px) == 0 {
			continue
		}
		f := buildFeat(px)

		// color saturation share from hist (sum over S bins >= 2)
		sat := 0.0
		for i, v := range f.Hist {
			if (i/vBins)%sBins >= 2 {
				sat += v
			}
		}
		flag := "GRAY"
		if sat > 0.30 {
			flag = "COLOR"
		}

		// top colors by 8-bit bucket frequency
		buckets := map[int][4]int{} // key -> r,g,b,count
		for _, p := range px {
			key := (int(p.R>>3) << 10) | (int(p.G>>3) << 5) | int(p.B>>3)
			c := buckets[key]
			c[0], c[1], c[2] = int(p.R&0xF8), int(p.G&0xF8), int(p.B&0xF8)
			c[3]++
			buckets[key] = c
		}
		type cc struct {
			rgb [3]int
			n   int
		}
		top := make([]cc, 0, len(buckets))
		for _, c := range buckets {
			top = append(top, cc{rgb: [3]int{c[0], c[1], c[2]}, n: c[3]})
		}
		sort.Slice(top, func(i, j int) bool { return top[i].n > top[j].n })

		fmt.Printf("%-5s %-64s n=%d tops=", flag, fn, f.N)
		for i := 0; i < len(top) && i < 3; i++ {
			fmt.Printf(" #%02x%02x%02x(%d%%)", top[i].rgb[0], top[i].rgb[1], top[i].rgb[2], 100*top[i].n/len(px))
		}
		fmt.Println()
	}
}

func mustList(dir string) []string {
	files, err := listPNG(dir)
	if err != nil {
		fatal(err)
	}
	return files
}

var _ = filepath.Join