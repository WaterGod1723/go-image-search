// Command inspect reports basic info (size, RGBA at corners/center) for each source PNG.
package main

import (
	"fmt"
	"image/png"
	"os"
	"path/filepath"
)

func main() {
	dir := "test_pngs"
	entries, err := os.ReadDir(dir)
	if err != nil {
		panic(err)
	}
	for _, e := range entries {
		if filepath.Ext(e.Name()) != ".png" {
			continue
		}
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
		b := img.Bounds()
		w, h := b.Dx(), b.Dy()
		_, _, _, a := img.At(b.Min.X, b.Min.Y).RGBA()
		_, _, _, a2 := img.At((b.Min.X+b.Max.X)/2, (b.Min.Y+b.Max.Y)/2).RGBA()
		fmt.Printf("%-50s %4dx%-4d cornerAlpha=%d centerAlpha=%d\n", e.Name(), w, h, a>>8, a2>>8)
	}
}