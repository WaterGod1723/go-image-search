package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

const _diagConst2 = 0

// maxi returns the larger of two ints.
func maxi(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// DiagEntry mirrors the generator's Sample for segmentation diagnosis.
type DiagEntry struct {
	Image     string  `json:"image"`
	Src       string  `json:"src"`
	Crop      [4]int  `json:"crop"`
	Canvas    [2]int  `json:"canvas"`
	Scale     float64 `json:"scale"`
	Translate [2]int  `json:"translate"`
}

// runDiag compares extracted query segmentation against ground-truth sprite
// placement from manifest.json (center + expected pixel count).
func runDiag(root string) {
	srcDir := filepath.Join(root, "test_pngs")
	sampleDir := filepath.Join(root, "test_set")
	data, err := os.ReadFile(filepath.Join(sampleDir, "manifest.json"))
	if err != nil {
		fatal(err)
	}
	var entries []DiagEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		fatal(err)
	}

	bad, good := 0, 0
	for _, e := range entries {
		img, err := loadPNG(filepath.Join(sampleDir, e.Image))
		if err != nil {
			fatal(err)
		}
		src, err := loadPNG(filepath.Join(srcDir, e.Src))
		if err != nil {
			fatal(err)
		}
		w, h := img.Bounds().Dx(), img.Bounds().Dy()

		px := extractQuery(img)
		if len(px) == 0 {
			fmt.Printf("%s %-24s X no extract\n", e.Image, e.Src)
			bad++
			continue
		}

		// expected sprite center (rotation is about sprite center).
		ex := float64(w)/2 + float64(e.Translate[0])
		ey := float64(h)/2 + float64(e.Translate[1])
		var cx, cy float64
		for _, p := range px {
			cx += float64(p.X)
			cy += float64(p.Y)
		}
		cx /= float64(len(px))
		cy /= float64(len(px))
		cd := math.Hypot(cx-ex, cy-ey)

		sw, sh := src.Bounds().Dx(), src.Bounds().Dy()
		refN := 0
		for y := 0; y < sh; y++ {
			for x := 0; x < sw; x++ {
				_, _, _, a := src.At(x, y).RGBA()
				if a > 4*0x101 {
					refN++
				}
			}
		}
		fill := 0.0
		if sw > 0 && sh > 0 {
			fill = float64(refN) / float64(sw*sh)
		}
		expArea := float64(sw*sh) * e.Scale * e.Scale * fill
		ratio := float64(len(px)) / math.Max(1, expArea)

		minX, minY, maxX, maxY := 1<<30, 1<<30, -1, -1
		for _, p := range px {
			if p.X < minX {
				minX = p.X
			}
			if p.Y < minY {
				minY = p.Y
			}
			if p.X > maxX {
				maxX = p.X
			}
			if p.Y > maxY {
				maxY = p.Y
			}
		}

		flag := "OK"
		if cd > float64(maxi(16, w/8)) || ratio < 0.3 || ratio > 2.2 ||
			(maxX-minX) > w || (maxY-minY) > h {
			flag = "FAIL"
			bad++
		} else {
			good++
		}
		fmt.Printf("%s %-18s %-28s dist=%-6.1f ratio=%.2f bbox=[%d,%d,%d,%d] n=%d exp=%d\n",
			flag, e.Image, e.Src, cd, ratio, minX, minY, maxX-minX+1, maxY-minY+1, len(px), int(expArea))
	}
	fmt.Printf("\nsegs OK=%d FAIL=%d\n", good, bad)
}
