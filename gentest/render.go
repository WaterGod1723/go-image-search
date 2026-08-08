package main

import (
	"image"
	"math"
	"math/rand"
)

// renderSample builds one canvas ("the query image") from the trimmed sprite.
// The scaled+rotated sprite is guaranteed to fit entirely inside the canvas
// (never clipped), and is kept large enough that thin icon strokes survive.
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

	rot := 0.0
	if rng.Float64() < 0.9 {
		rot = rng.Float64()*360 - 180
	}
	rad := rot * math.Pi / 180
	cos, sin := math.Cos(rad), math.Sin(rad)

	// Rotated bounding box of the sprite at unit scale, so we can guarantee fit.
	rotW := float64(sw)*math.Abs(cos) + float64(sh)*math.Abs(sin)
	rotH := float64(sw)*math.Abs(sin) + float64(sh)*math.Abs(cos)

	// Desired size: sprite's max side = frac * canvas min side.
	frac := *gl + rng.Float64()*(*gh-*gl)
	scale := frac * float64(mini(w, h)) / float64(maxi(sw, sh))

	// Keep the sprite large enough that thin strokes (1-3px at source) survive
	// bilinear rendering and the retrieval threshold: floor the rendered max
	// side at ~45% of the canvas min side.
	minScale := 0.45 * float64(mini(w, h)) / float64(maxi(sw, sh))

	// Never let the rotated sprite poke outside the canvas (keep a 2px margin).
	maxFit := math.Min(float64(w-2)/rotW, float64(h-2)/rotH)

	// clamp: lower bound minScale, upper bound maxFit (minScale cannot exceed
	// maxFit because sprite max side <= canvas min side).
	if scale < minScale {
		scale = minScale
	}
	if scale > maxFit {
		scale = maxFit
	}

	// Random translation, clamped so the whole rotated sprite stays on-canvas.
	cw, ch := rotW*scale, rotH*scale
	mx := float64(w) - cw
	my := float64(h) - ch
	if mx < 0 {
		mx = 0
	}
	if my < 0 {
		my = 0
	}
	ox := cw/2 + mx/2 + (rng.Float64()-0.5)*mx
	oy := ch/2 + my/2 + (rng.Float64()-0.5)*my

	affinePaint(canvas, sprite, scale, rot, ox, oy)

	// screen-space bbox of the drawn sprite, used to keep edge text clear of it
	spriteRect := spriteScreenRect(sw, sh, scale, rot, ox, oy)
	spec.Texts = drawEdgeText(rng, lib, canvas, *allSd, spriteRect)

	spec.Canvas = [2]int{w, h}
	spec.BGHex = hexColor(bg)
	spec.Rotation = rot
	spec.Scale = scale
	spec.Translate = [2]int{int(ox - float64(w)/2), int(oy - float64(h)/2)}
	return canvas, spec
}

// spriteScreenRect returns the axis-aligned bounding box on the canvas that the
// scaled+rotated sprite occupies (in [x0,y0,x1,y1]).
func spriteScreenRect(sw, sh int, scale, deg, ox, oy float64) [4]float64 {
	rad := deg * math.Pi / 180
	cs, sn := math.Cos(rad), math.Sin(rad)
	x0, y0 := 1e18, 1e18
	x1, y1 := -1e18, -1e18
	for _, c := range [][2]float64{{-1, -1}, {1, -1}, {-1, 1}, {1, 1}} {
		cx, cy := c[0]*float64(sw)/2, c[1]*float64(sh)/2
		sx := ox + (cx*cs+cy*sn)*scale
		sy := oy + (-cx*sn+cy*cs)*scale
		if sx < x0 {
			x0 = sx
		}
		if sx > x1 {
			x1 = sx
		}
		if sy < y0 {
			y0 = sy
		}
		if sy > y1 {
			y1 = sy
		}
	}
	return [4]float64{x0, y0, x1, y1}
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