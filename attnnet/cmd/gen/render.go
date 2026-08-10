package main

import (
	"image"
	"math"
	"math/rand"

	"attnnet"
)

// renderOne builds one full-resolution canvas from the trimmed sprite (same
// recipe as the parent repo's gentest tool: random background, rotation, scale,
// translation, edge text), and returns it with its transform spec.
func renderOne(rng *rand.Rand, lib *fontLib, sprite *image.NRGBA, crop [4]int) (*image.NRGBA, Sample) {
	var spec Sample
	spec.Crop = crop

	sw, sh := sprite.Bounds().Dx(), sprite.Bounds().Dy()
	cmin := 320 + rng.Intn(240)
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

	rotW := float64(sw)*math.Abs(cos) + float64(sh)*math.Abs(sin)
	rotH := float64(sw)*math.Abs(sin) + float64(sh)*math.Abs(cos)

	frac := 0.35 + rng.Float64()*0.5
	scale := frac * float64(mini(w, h)) / float64(maxi(sw, sh))
	minScale := 0.45 * float64(mini(w, h)) / float64(maxi(sw, sh))
	maxFit := math.Min(float64(w-2)/rotW, float64(h-2)/rotH)
	if scale < minScale {
		scale = minScale
	}
	if scale > maxFit {
		scale = maxFit
	}

	cw, ch := rotW*scale, rotH*scale
	mx := math.Max(float64(w)-cw, 0)
	my := math.Max(float64(h)-ch, 0)
	ox := cw/2 + mx/2 + (rng.Float64()-0.5)*mx
	oy := ch/2 + my/2 + (rng.Float64()-0.5)*my

	affinePaint(canvas, sprite, scale, rot, ox, oy)

	spriteRect := spriteScreenRect(sw, sh, scale, rot, ox, oy)
	spec.Texts = drawEdgeText(rng, lib, canvas, false, spriteRect)

	spec.Canvas = [2]int{w, h}
	spec.BGHex = hexColor(bg)
	spec.Rotation = rot
	spec.Scale = scale
	spec.Translate = [2]int{int(ox - float64(w)/2), int(oy - float64(h)/2)}
	return canvas, spec
}

// spriteScreenRect returns the axis-aligned bbox [x0,y0,x1,y1] of the scaled
// and rotated sprite on the canvas.
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

// buildMask maps the sprite bbox (canvas coords) onto the small patch grid:
// a patch is "icon" when its center falls inside the scaled bbox.
func buildMask(cfg attnnet.Config, canvas [2]int, rect [4]float64) []byte {
	gs := float64(cfg.ImageSize)
	sx := gs / float64(canvas[0])
	sy := gs / float64(canvas[1])
	x0, y0 := rect[0]*sx, rect[1]*sy
	x1, y1 := rect[2]*sx, rect[3]*sy
	grid := cfg.ImageSize / cfg.PatchSize
	mask := make([]byte, grid*grid)
	for py := 0; py < grid; py++ {
		for px := 0; px < grid; px++ {
			cx := (float64(px)+0.5)*float64(cfg.PatchSize)
			cy := (float64(py)+0.5)*float64(cfg.PatchSize)
			if cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1 {
				mask[py*grid+px] = 1
			}
		}
	}
	return mask
}

// downscale renders canvas into a cfg.ImageSize x cfg.ImageSize RGB block.
func downscale(src *image.NRGBA, size int) []byte {
	sw, sh := src.Bounds().Dx(), src.Bounds().Dy()
	out := make([]byte, size*size*3)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			fx := (float64(x) + 0.5) * float64(sw) / float64(size)
			fy := (float64(y) + 0.5) * float64(sh) / float64(size)
			r, g, b, _ := bilinearSample(src, fx-0.5, fy-0.5)
			o := (y*size + x) * 3
			out[o] = uint8(r)
			out[o+1] = uint8(g)
			out[o+2] = uint8(b)
		}
	}
	return out
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
