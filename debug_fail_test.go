package main

import (
	"fmt"
	"path/filepath"
	"testing"

	"go-image-search/internal/imageproc"
	"go-image-search/internal/index"
	"go-image-search/internal/phash"
	"go-image-search/internal/segment"
)

func TestDebugFail(t *testing.T) {
	cfg := segment.DefaultConfig()
	mc := segment.DefaultMergeConfig()
	g := segment.DefaultGravityConfig()
	cases := []struct{ q, lib string }{
		{"test_pngs_target/TEST1_FROM_dashboard_76dp.png", "test_pngs/dashboard_76dp.png"},
		{"test_pngs_target/TEST2_FROM_dashboard_76dp.png", "test_pngs/dashboard_76dp.png"},
		{"test_pngs_target/TEST11_FROM_fushixiebao.png", "test_pngs/fushixiebao.png"},
	}
	for _, c := range cases {
		qlib, _ := loadTestImage(c.lib)
		qimg, _ := loadTestImage(c.q)
		fmt.Printf("\n########## %s vs %s ##########\n", filepath.Base(c.q), filepath.Base(c.lib))
		libHs := [][]index.RegionHash(nil)
		qHs := [][]index.RegionHash(nil)
		for vi, v := range imageproc.QueryVariants(qlib) {
			h, _ := hashImage(v, cfg, mc, g)
			libHs = append(libHs, h)
			fmt.Printf("LIB v%d: %d regions\n", vi, len(h))
			for i, r := range h {
				fmt.Printf("  #%d hash=%016x shape=%016x area=%d bbox=%v fill=%.2f asp=%.2f global=%.1f color=%v\n",
					i, r.Hash, r.Shape, r.Area, r.BBox, r.Fill, r.Aspect, r.Global, r.Color)
			}
		}
		for vi, v := range imageproc.QueryVariants(qimg) {
			h, _ := hashImage(v, cfg, mc, g)
			qHs = append(qHs, h)
			fmt.Printf("Q   v%d: %d regions\n", vi, len(h))
			for i, r := range h {
				fmt.Printf("  #%d hash=%016x shape=%016x area=%d bbox=%v fill=%.2f asp=%.2f global=%.1f color=%v\n",
					i, r.Hash, r.Shape, r.Area, r.BBox, r.Fill, r.Aspect, r.Global, r.Color)
			}
		}
		fmt.Println("--- min dist (query region vs lib region) ---")
		for qi, qSet := range qHs {
			for li, lSet := range libHs {
				bestHash, bestShape := 999, 999
				for _, a := range qSet {
					for _, b := range lSet {
						dh := phash.Hamming(a.Hash, b.Hash)
						ds := phash.Hamming(a.Shape, b.Shape)
						if dh < bestHash { bestHash = dh }
						if ds < bestShape { bestShape = ds }
					}
				}
				fmt.Printf("  qv%d-lv%d: minHash=%d minShape=%d\n", qi, li, bestHash, bestShape)
			}
		}
	}
}
