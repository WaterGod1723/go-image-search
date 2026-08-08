package main

import (
	"fmt"
	"path/filepath"
)

// runInvTest evaluates descriptor invariance under 90-degree rotation.
// Rotation-invariant descriptors (zern, radial) should stay ~1.0.
func runInvTest(root string) {
	srcDir := filepath.Join(root, "test_pngs")
	files := mustList(srcDir)
	stats := map[string]invStat{}

	for _, fn := range files {
		img, err := loadPNG(filepath.Join(srcDir, fn))
		if err != nil {
			fatal(err)
		}
		px := refPixels(img)
		if len(px) == 0 {
			continue
		}
		f0 := buildFeat(px)
		rpx := umr(px)
		f90 := buildFeat(rpx)
		stats[fn] = invStat{
			z:    cosSim(f0.Zern, f90.Zern),
			rad:  cosSim(f0.Radial, f90.Radial),
			hist: histSim(f0.Hist, f90.Hist),
		}
	}

	keys := make([]string, 0, len(stats))
	for k := range stats {
		keys = append(keys, k)
	}
	insertStrings(keys, func(a, b string) bool {
		return stats[a].z < stats[b].z
	})
	for _, k := range keys {
		fmt.Printf("%-56s zern=%.3f rad=%.3f hist=%.3f\n", k, stats[k].z, stats[k].rad, stats[k].hist)
	}
}

type invStat struct {
	z, rad, hist float64
}

func umr(px []Px) []Px {
	var sx, sy float64
	for _, p := range px {
		sx += float64(p.X)
		sy += float64(p.Y)
	}
	n := float64(len(px))
	cx, cy := sx/n, sy/n
	out := make([]Px, len(px))
	for i, p := range px {
		dx, dy := float64(p.X)-cx, float64(p.Y)-cy
		out[i] = Px{X: int(cx - dy), Y: int(cy + dx), R: p.R, G: p.G, B: p.B, A: p.A}
	}
	return out
}

func insertStrings(s []string, less func(a, b string) bool) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && less(s[j], s[j-1]); j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}