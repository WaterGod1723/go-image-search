// Command anal dumps statistical characteristics of the source icons to guide
// feature selection for the reverse-image-search algorithm.
package main

import (
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
)

type counter map[uint32]int

type stat struct {
	name    string
	w, h    int
	colors  int
	opaque  int
	transp  int
	top     [6][3]int
	typeTag string
}

func main() {
	dir := "test_pngs"
	entries, err := os.ReadDir(dir)
	if err != nil {
		panic(err)
	}

	var stats []stat
	for _, e := range entries {
		if filepath.Ext(e.Name()) != ".png" {
			continue
		}
		if !e.IsDir() {
			f, err := os.Open(filepath.Join(dir, e.Name()))
			if err != nil {
				panic(err)
			}
			img, err := png.Decode(f)
			f.Close()
			if err != nil {
				fmt.Printf("%-50s decode_err: %v\n", e.Name(), err)
				continue
			}
			stats = append(stats, classify(e.Name(), img))
		}
	}

	var byCol = make([]stat, len(stats))
	copy(byCol, stats)
	for i := 0; i < len(byCol); i++ {
		fmt.Printf("%-48s %4dx%-4d colors=%-3d opaque=%-6d transparent=%-6d type=%-8s\n",
			byCol[i].name, byCol[i].w, byCol[i].h, byCol[i].colors, byCol[i].opaque, byCol[i].transp, byCol[i].typeTag)
	}

	nSingle := 0
	for _, s := range stats {
		if s.colors == 1 {
			nSingle++
		}
	}
	fmt.Printf("\n# icons with exactly 1 opaque quantized color: %d / %d (%.1f%%)\n",
		nSingle, len(stats), 100*float64(nSingle)/float64(len(stats)))
}

// classify buckets opaque pixels into quantized 4-bit-per-channel colors.
func classify(name string, img image.Image) stat {
	b := img.Bounds()
	cc := make(counter)
	opaque, transp := 0, 0
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, bl, a := img.At(x, y).RGBA()
			if a < 4*0x101 {
				transp++
				continue
			}
			opaque++
			rq := (r >> 8) >> 4
			gq := (g >> 8) >> 4
			bq := (bl >> 8) >> 4
			key := uint32(rq)<<8 | uint32(gq)<<4 | uint32(bq)
			cc[key]++
		}
	}
	type kv struct {
		k int
		v int
	}
	var sorted []kv
	for k, v := range cc {
		sorted = append(sorted, kv{int(k), v})
	}
	for i := 1; i < len(sorted); i++ {
		for j := i; j > 0 && sorted[j].v > sorted[j-1].v; j-- {
			sorted[j], sorted[j-1] = sorted[j-1], sorted[j]
		}
	}

	tag := ""
	if len(sorted) == 1 {
		tag = "MONO"
	} else if len(sorted) >= 2 && float64(sorted[0].v) > 0.7*float64(opaque) {
		tag = "MONO_DOM"
	} else if len(sorted) >= 3 {
		tag = "MULTI"
	}

	var top [6][3]int
	for i := 0; i < len(sorted) && i < 6; i++ {
		k := sorted[i].k
		top[i] = [3]int{(k >> 8) << 4, (k >> 4 & 0xF) << 4, (k & 0xF) << 4}
	}
	return stat{
		name:    filepath.Base(name),
		w:       b.Dx(),
		h:       b.Dy(),
		colors:  len(sorted),
		opaque:  opaque,
		transp:  transp,
		top:     top,
		typeTag: tag,
	}
}