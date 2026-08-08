// Command stat dumps per-test-sample: what percent of the canvas is background,
// what the icon colors look like, and whether border text is separable from the
// icon purely by color. Used to validate the segmentation strategy.
package main

import (
	"fmt"
	"image/png"
	"os"
	"path/filepath"
)

func main() {
	// analyze color separation: bg histogram mode vs sprite colors
	files, _ := os.ReadDir("test_set")
	for _, e := range files {
		if filepath.Ext(e.Name()) != ".png" {
			continue
		}
		f, _ := os.Open(filepath.Join("test_set", e.Name()))
		img, err := png.Decode(f)
		f.Close()
		if err != nil {
			continue
		}
		b := img.Bounds()
		w, h := b.Dx(), b.Dy()

		// sample a set of colors with counts via small 6-bit quantization
		type qkey struct{ r, g, bl uint8 }
		hist := make(map[qkey]int)
		for y := b.Min.Y; y < b.Max.Y; y += 2 {
			for x := b.Min.X; x < b.Max.X; x += 2 {
				r, g, bl, _ := img.At(x, y).RGBA()
				k := qkey{uint8(r >> 10), uint8(g >> 10), uint8(bl >> 10)}
				hist[k]++
			}
		}
		// find top-3
		type kv struct {
			k qkey
			v int
		}
		var arr []kv
		for k, v := range hist {
			arr = append(arr, kv{k, v})
		}
		for i := 0; i < len(arr); i++ {
			for j := i + 1; j < len(arr); j++ {
				if arr[j].v > arr[i].v {
					arr[i], arr[j] = arr[j], arr[i]
				}
			}
		}
		bg := arr[0].k
		// % pixels within small distance of bg
		close := 0
		tot := 0
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				r, g, bl, _ := img.At(x, y).RGBA()
				dr := int((uint8(r>>8)>>2)<<2) - int(bg.r<<2)
				dg := int((uint8(g>>8)>>2)<<2) - int(bg.g<<2)
				db := int((uint8(bl>>8)>>2)<<2) - int(bg.bl<<2)
				tot++
				if dr*dr+dg*dg+db*db < 20 {
					close++
				}
			}
		}
		pct := float64(close) / float64(tot) * 100
		fmt.Printf("%s %4dx%-4d bg=[%02x%02x%02x] bgPct=%.1f%%\n",
			e.Name(), w, h, bg.r<<2, bg.g<<2, bg.bl<<2, pct)
	}
}
