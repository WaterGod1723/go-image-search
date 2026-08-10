package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
)

var fontCandidates = []string{
	`C:\Windows\Fonts\msyh.ttc`,
	`C:\Windows\Fonts\simhei.ttf`,
	`C:\Windows\Fonts\simsun.ttc`,
	`C:\Windows\Fonts\arial.ttf`,
	"/System/Library/Fonts/PingFang.ttc",
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
	b := img.Bounds()
	nrgba := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			nrgba.Set(x, y, color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)))
		}
	}
	return nrgba, nil
}

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
	return fmt.Sprintf("%02x%02x%02x", c.R, c.G, c.B)
}

func affinePaint(dst *image.NRGBA, sprite *image.NRGBA, s, deg, ox, oy float64) {
	sb := sprite.Bounds()
	sw, sh := sb.Dx(), sb.Dy()
	rad := deg * math.Pi / 180
	cos, sin := math.Cos(rad), math.Sin(rad)
	dw, dh := dst.Bounds().Dx(), dst.Bounds().Dy()
	for y := 0; y < dh; y++ {
		for x := 0; x < dw; x++ {
			dx, dy := float64(x)-ox, float64(y)-oy
			sx := (dx*cos + dy*sin) / s
			sy := (-dx*sin + dy*cos) / s
			sx += float64(sw) / 2
			sy += float64(sh) / 2
			if sx < -1 || sy < -1 || sx > float64(sw) || sy > float64(sh) {
				continue
			}
			r, g, b, a := bilinearSample(sprite, sx, sy)
			if a <= 0 {
				continue
			}
			oi := dst.PixOffset(x, y)
			af := a / 255
			bf := 1 - af
			dst.Pix[oi] = uint8(float64(r)*af + float64(dst.Pix[oi])*bf)
			dst.Pix[oi+1] = uint8(float64(g)*af + float64(dst.Pix[oi+1])*bf)
			dst.Pix[oi+2] = uint8(float64(b)*af + float64(dst.Pix[oi+2])*bf)
		}
	}
}

func bilinearSample(img *image.NRGBA, x, y float64) (r, g, b, a float64) {
	dd := img.Bounds()
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	if x0 < dd.Min.X {
		x0 = dd.Min.X
	}
	if y0 < dd.Min.Y {
		y0 = dd.Min.Y
	}
	if x0 > dd.Max.X-1 {
		x0 = dd.Max.X - 1
	}
	if y0 > dd.Max.Y-1 {
		y0 = dd.Max.Y - 1
	}
	x1, y1 := x0+1, y0+1
	if x1 > dd.Max.X-1 {
		x1 = dd.Max.X - 1
	}
	if y1 > dd.Max.Y-1 {
		y1 = dd.Max.Y - 1
	}
	fx, fy := x-float64(x0), y-float64(y0)
	o00 := img.PixOffset(x0, y0)
	o10 := img.PixOffset(x1, y0)
	o01 := img.PixOffset(x0, y1)
	o11 := img.PixOffset(x1, y1)
	var out [4]float64
	for c := 0; c < 4; c++ {
		v00 := float64(img.Pix[o00+c])
		v10 := float64(img.Pix[o10+c])
		v01 := float64(img.Pix[o01+c])
		v11 := float64(img.Pix[o11+c])
		top := v00*(1-fx) + v10*fx
		bot := v01*(1-fx) + v11*fx
		out[c] = top*(1-fy) + bot*fy
	}
	return out[0], out[1], out[2], out[3]
}
