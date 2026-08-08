package main

import (
	"image"
	"math"
	"math/rand"
)

// renderSample builds one canvas ("the query image") from the trimmed sprite.
func renderSample(rng *rand.Rand, lib *fontLib, sprite *image.NRGBA, crop [4]int) (*image.NRGBA, Sample) {
	var spec Sample
	spec.Crop = crop

	sw, sh := sprite.Bounds().Dx(), sprite.Bounds().Dy()
	cmin := int(*minS + rng.Float64()*(*maxS-*minS))
	w := cmin + rng.Intn(maxi(24, cmin/3))
	h := cmin + rng.Intn(maxi(24, cmin/3))
	if rng.Intn(3) == 0 {
		h = w
	}

	canvas := image.NewNRGBA(image.Rect(0, 0, w, h))
	bg := randColor(rng, 0.10, 0.90)
	flatFill(canvas, bg)

	// sprite -> canvas affine mapping
	frac := *gl + rng.Float64()*(*gh-*gl)
	if rng.Intn(10) == 0 { // occasionally spill over the canvas edge
		frac = 1.0 + rng.Float64()*0.5
	}
	scale := frac * float64(mini(w, h)) / float64(maxi(sw, sh))

	rot := 0.0
	if rng.Float64() < 0.9 {
		rot = rng.Float64()*360 - 180
	}
	ox := float64(w)/2 + (rng.Float64()-0.5)*float64(w)*0.9
	oy := float64(h)/2 + (rng.Float64()-0.5)*float64(h)*0.9

	affinePaint(canvas, sprite, scale, rot, ox, oy)

	spec.Texts = drawEdgeText(rng, lib, canvas, *allSd)

	spec.Canvas = [2]int{w, h}
	spec.BGHex = hexColor(bg)
	spec.Rotation = rot
	spec.Scale = scale
	spec.Translate = [2]int{int(ox - float64(w)/2), int(oy - float64(h)/2)}
	return canvas, spec
}

// affinePaint paints sprite over dst: the sprite is scaled by s, rotated by
// deg (radians), and its center placed at (ox, oy). Bilinear inverse mapping.
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

// bilinearSample returns the straight (non-premultiplied) RGBA at (x, y).
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

func maxi(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func mini(a, b int) int {
	if a < b {
		return a
	}
	return b
}