package main

import (
	"fmt"
	"image"
	"image/color"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
)

// runAugRefs synthesizes decorated reference icons: each source sprite is
// re-rendered on a transparent canvas and embellished with random colored
// shapes and random text, producing perSrc distinct NEW reference icons. This
// enlarges (and de-duplicates) the reference pool so the synthetic training
// set covers far more icon variants than the raw sources alone.
func runAugRefs(rng *rand.Rand, lib *fontLib, srcDir, outDir string, perSrc int) {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		log.Fatal(err)
	}
	srcs, err := listPNGs(srcDir)
	if err != nil {
		log.Fatal(err)
	}
	written := 0
	for _, fn := range srcs {
		img, err := loadPNG(filepath.Join(srcDir, fn))
		if err != nil {
			log.Fatal(err)
		}
		box := trimAlpha(img)
		if box == nil {
			continue
		}
		sprite := cutBox(img, *box)
		base := strings.TrimSuffix(fn, filepath.Ext(fn))
		for v := 0; v < perSrc; v++ {
			canvas := buildAugRef(rng, lib, sprite)
			out := filepath.Join(outDir, fmt.Sprintf("aug_%s_v%02d.png", base, v))
			if err := savePNG(out, canvas); err != nil {
				log.Fatal(err)
			}
			written++
		}
	}
	fmt.Printf("augmented refs: %d written to %q\n", written, outDir)
}

// buildAugRef renders the sprite centered on a transparent square canvas and
// adds 1-3 random colored shapes plus 0-2 random text labels.
func buildAugRef(rng *rand.Rand, lib *fontLib, sprite *image.NRGBA) *image.NRGBA {
	sw, sh := sprite.Bounds().Dx(), sprite.Bounds().Dy()
	side := maxi(sw, sh)
	side = int(float64(side) * (1.35 + rng.Float64()*0.25))
	if side < 96 {
		side = 96
	}
	canvas := image.NewNRGBA(image.Rect(0, 0, side, side))
	ox := (side - sw) / 2
	oy := (side - sh) / 2
	compositeInto(canvas, sprite, ox, oy)

	nShapes := 1 + rng.Intn(3) // 1..3
	for i := 0; i < nShapes; i++ {
		drawRandomShape(rng, canvas)
	}
	nTexts := rng.Intn(3) // 0..2
	for i := 0; i < nTexts; i++ {
		drawRandomText(rng, lib, canvas, side)
	}
	return canvas
}

func compositeInto(dst *image.NRGBA, src *image.NRGBA, ox, oy int) {
	sb := src.Bounds()
	db := dst.Bounds()
	for y := 0; y < sb.Dy(); y++ {
		for x := 0; x < sb.Dx(); x++ {
			px := src.NRGBAAt(x, y)
			if px.A == 0 {
				continue
			}
			dx, dy := ox+x, oy+y
			if dx < 0 || dy < 0 || dx >= db.Dx() || dy >= db.Dy() {
				continue
			}
			dst.SetNRGBA(dx, dy, px)
		}
	}
}

// drawRandomShape paints a filled circle, rect or triangle with a random
// saturated color at a random spot.
func drawRandomShape(rng *rand.Rand, canvas *image.NRGBA) {
	w, h := canvas.Bounds().Dx(), canvas.Bounds().Dy()
	c := randomInk(rng)
	maxR := float64(maxi(w, h))
	r := maxR * (0.08 + rng.Float64()*0.16)
	cx := r/2 + rng.Float64()*(float64(w)-r)
	cy := r/2 + rng.Float64()*(float64(h)-r)

	switch rng.Intn(3) {
	case 0: // circle
		for yy := 0; yy < h; yy++ {
			for xx := 0; xx < w; xx++ {
				dx, dy := float64(xx)-cx, float64(yy)-cy
				if dx*dx+dy*dy <= r*r {
					overPaint(canvas, xx, yy, c)
				}
			}
		}
	case 1: // rect
		x0, y0 := int(cx-r), int(cy-r*0.7)
		x1, y1 := int(cx+r), int(cy+r*0.7)
		for yy := y0; yy < y1; yy++ {
			for xx := x0; xx < x1; xx++ {
				overPaint(canvas, xx, yy, c)
			}
		}
	default: // triangle (flat-bottom isoceles, pointing up)
		x0, y0 := cx, cy-r
		x1, y1 := cx-r, cy+r
		x2, y2 := cx+r, cy+r
		for yy := 0; yy < h; yy++ {
			for xx := 0; xx < w; xx++ {
				if inTriangle(float64(xx), float64(yy), x0, y0, x1, y1, x2, y2) {
					overPaint(canvas, xx, yy, c)
				}
			}
		}
	}
}

func inTriangle(px, py, x0, y0, x1, y1, x2, y2 float64) bool {
	d := (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2)
	if d == 0 {
		return false
	}
	a := ((y1-y2)*(px-x2) + (x2-x1)*(py-y2)) / d
	b := ((y2-y0)*(px-x2) + (x0-x2)*(py-y2)) / d
	c := 1 - a - b
	return a >= 0 && b >= 0 && c >= 0
}

// drawRandomText draws a random word with random color somewhere on the canvas.
func drawRandomText(rng *rand.Rand, lib *fontLib, canvas *image.NRGBA, side int) {
	size := int(float64(side) * (0.07 + rng.Float64()*0.11))
	if size < 14 {
		size = 14
	}
	f, err := lib.faceAt(float64(size))
	if err != nil {
		return
	}
	c := randomInk(rng)
	c.A = 255
	word := fontWord(rng)
	w, h := canvas.Bounds().Dx(), canvas.Bounds().Dy()
	x := rng.Intn(maxi(1, w-size*len(word)))
	y := size + rng.Intn(maxi(1, h-size*2))
	textDrawH(canvas, f, c, word, x, y)
}

// randomInk is a random saturated color, mostly opaque.
func randomInk(rng *rand.Rand) color.NRGBA {
	hi, lo := rng.Float64()*0.6, rng.Float64()*0.4
	ch := func() uint8 { return uint8((lo + rng.Float64()*(hi-lo)) * 255) }
	a := 200 + uint8(rng.Intn(56))
	return color.NRGBA{ch(), ch(), ch(), a}
}

// overPaint composites straight-alpha c over dst(x,y).
func overPaint(dst *image.NRGBA, x, y int, c color.NRGBA) {
	w, h := dst.Bounds().Dx(), dst.Bounds().Dy()
	if x < 0 || y < 0 || x >= w || y >= h {
		return
	}
	o := dst.PixOffset(x, y)
	sa := float64(c.A) / 255
	da := float64(dst.Pix[o+3]) / 255
	oa := sa + da*(1-sa)
	if oa <= 1e-6 {
		return
	}
	for i := 0; i < 3; i++ {
		v := (float64(dst.Pix[o+i])*da*(1-sa) + float64(pixAt(c, i))*sa) / oa
		dst.Pix[o+i] = clampU8(v)
	}
	dst.Pix[o+3] = clampU8(oa * 255)
}

func pixAt(c color.NRGBA, i int) uint8 {
	switch i {
	case 0:
		return c.R
	case 1:
		return c.G
	default:
		return c.B
	}
}

func clampU8(v float64) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v + 0.5)
}