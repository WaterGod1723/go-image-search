package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
)

// TTF/TTC candidate system font paths, tried in order.
var fontCandidates = []string{
	`C:\Windows\Fonts\msyh.ttc`,
	`C:\Windows\Fonts\simhei.ttf`,
	`C:\Windows\Fonts\simsun.ttc`,
	`C:\Windows\Fonts\arial.ttf`,
	"/System/Library/Fonts/PingFang.ttc",
	"/System/Library/Fonts/STHeiti Medium.ttc",
	"/System/Library/Fonts/Supplemental/Songti.ttc",
	"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
}

func listPNGs(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() || !strings.EqualFold(filepath.Ext(e.Name()), ".png") {
			continue
		}
		out = append(out, e.Name())
	}
	return out, nil
}

func loadPNG(path string) (*image.NRGBA, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		return nil, err
	}
	if n, ok := img.(*image.NRGBA); ok {
		return n, nil
	}
	// convert any other type to NRGBA (straight alpha)
	b := img.Bounds()
	nrgba := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			nrgba.Set(x, y, color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)))
		}
	}
	return nrgba, nil
}

func savePNG(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

// trimAlpha returns the bbox of pixels with alpha > 4/255, or nil if transparent.
func trimAlpha(img *image.NRGBA) *[4]int {
	b := img.Bounds()
	minX, minY := b.Max.X, b.Max.Y
	maxX, maxY := b.Min.X-1, b.Min.Y-1
	any := false
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			_, _, _, a := img.At(x, y).RGBA()
			if a > 4*0x101 {
				any = true
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	if !any {
		return nil
	}
	return &[4]int{minX, minY, maxX - minX + 1, maxY - minY + 1}
}

// cutBox extracts the [x, y, w, h] region of img into a new NRGBA.
func cutBox(img *image.NRGBA, box [4]int) *image.NRGBA {
	x, y, w, h := box[0], box[1], box[2], box[3]
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	for j := 0; j < h; j++ {
		for i := 0; i < w; i++ {
			out.Set(i, j, img.At(x+i, y+j))
		}
	}
	return out
}

// flatFill sets every pixel of img to c.
func flatFill(img *image.NRGBA, c color.NRGBA) {
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i] = c.R
		img.Pix[i+1] = c.G
		img.Pix[i+2] = c.B
		img.Pix[i+3] = 255
	}
}

func randColor(rng *rand.Rand, lo, hi float64) color.NRGBA {
	ch := func() uint8 {
		return uint8(lo*255 + rng.Float64()*(hi-lo)*255)
	}
	return color.NRGBA{ch(), ch(), ch(), 255}
}

func hexColor(c color.NRGBA) string {
	return fmt.Sprintf("#%02x%02x%02x", c.R, c.G, c.B)
}