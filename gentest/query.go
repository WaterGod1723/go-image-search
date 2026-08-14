package main

import (
	"image"
	"image/color"
	"math"
	"math/rand"

	"golang.org/x/image/font"
)

// renderQuery renders a ref sprite on a canvas with random rotation/position
// (ref fully visible, never clipped) and non-overlapping interference: random
// colored lines, curves, and text drawn ONLY outside the ref's screen bounding
// box (+ margin). The ref's max dimension is guaranteed ≥ 30% of the canvas's
// min dimension so the icon is always clearly the primary subject.
func renderQuery(rng *rand.Rand, lib *fontLib, sprite *image.NRGBA, crop [4]int) (*image.NRGBA, Sample) {
	var spec Sample
	spec.Crop = crop

	sw, sh := sprite.Bounds().Dx(), sprite.Bounds().Dy()
	spriteMax := float64(maxi(sw, sh))

	// Canvas: 200–400 px, can be non-square, 1/3 chance square.
	cmin := 200 + rng.Intn(200)
	w := cmin + rng.Intn(cmin/3)
	h := cmin + rng.Intn(cmin/3)
	if rng.Intn(3) == 0 {
		h = w
	}

	canvas := image.NewNRGBA(image.Rect(0, 0, w, h))
	bg := randColor(rng, 0.10, 0.90)
	flatFill(canvas, bg)

	// Ref: random rotation, scale chosen so ref max dimension ≥ 30% of canvas
	// min dimension, but still fits entirely on canvas.
	rot := rng.Float64() * 360
	rad := rot * math.Pi / 180
	cos, sin := math.Abs(math.Cos(rad)), math.Abs(math.Sin(rad))
	rotW := float64(sw)*cos + float64(sh)*sin
	rotH := float64(sw)*sin + float64(sh)*cos
	rotMax := math.Max(rotW, rotH)

	const minRefRatio = 0.30
	minScale := minRefRatio * float64(mini(w, h)) / spriteMax
	// upper bound: ref must fit on canvas with 4px margin
	margin := 4.0
	maxFit := math.Min(float64(w)-2*margin, float64(h)-2*margin) / rotMax
	if maxFit < minScale {
		maxFit = minScale // edge case: very small canvas
	}
	// random scale in [minScale, maxFit], biased towards lower end
	scale := minScale + rng.Float64()*(maxFit-minScale)
	if scale < minScale {
		scale = minScale
	}

	scaledW := rotW * scale
	scaledH := rotH * scale
	slackX := float64(w) - scaledW - 2*margin
	slackY := float64(h) - scaledH - 2*margin
	if slackX < 0 {
		slackX = 0
	}
	if slackY < 0 {
		slackY = 0
	}
	ox := scaledW/2 + margin + rng.Float64()*slackX
	oy := scaledH/2 + margin + rng.Float64()*slackY

	affinePaint(canvas, sprite, scale, rot, ox, oy)

	// Exclusion zone: ref screen bounding box + margin (interference must not
	// enter this rectangle so the ref stays fully visible).
	spriteRect := spriteScreenRect(sw, sh, scale, rot, ox, oy)
	exMargin := 6
	zone := [4]int{
		int(spriteRect[0]) - exMargin,
		int(spriteRect[1]) - exMargin,
		int(spriteRect[2]) + exMargin,
		int(spriteRect[3]) + exMargin,
	}

	// Interference: lines, curves, text — all skip the exclusion zone.
	nLines := 3 + rng.Intn(6) // 3–8
	for i := 0; i < nLines; i++ {
		drawInterferenceLine(rng, canvas, zone, w, h)
	}
	nCurves := 1 + rng.Intn(3) // 1–3
	for i := 0; i < nCurves; i++ {
		drawInterferenceCurve(rng, canvas, zone, w, h)
	}
	nTexts := rng.Intn(4) // 0–3
	var texts []TextOut
	for i := 0; i < nTexts; i++ {
		to := drawInterferenceText(rng, lib, canvas, zone, w, h)
		if to.Text != "" {
			texts = append(texts, to)
		}
	}

	spec.Canvas = [2]int{w, h}
	spec.BGHex = hexColor(bg)
	spec.Rotation = rot
	spec.Scale = scale
	spec.Translate = [2]int{int(ox - float64(w)/2), int(oy - float64(h)/2)}
	spec.Texts = texts
	return canvas, spec
}

// ---- interference helpers ----

func inZone(x, y int, zone [4]int) bool {
	return x >= zone[0] && x <= zone[2] && y >= zone[1] && y <= zone[3]
}

func absi(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func rectOverlap(a, b [4]int) bool {
	return a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]
}

// paintDot paints a filled disk of the given radius at (x,y).
func paintDot(canvas *image.NRGBA, x, y, radius int, c color.NRGBA) {
	if radius <= 1 {
		overPaint(canvas, x, y, c)
		return
	}
	for dy := -radius + 1; dy < radius; dy++ {
		for dx := -radius + 1; dx < radius; dx++ {
			if dx*dx+dy*dy <= radius*radius {
				overPaint(canvas, x+dx, y+dy, c)
			}
		}
	}
}

// drawInterferenceLine draws a random straight line (Bresenham) with random
// color and width, skipping pixels inside the exclusion zone.
func drawInterferenceLine(rng *rand.Rand, canvas *image.NRGBA, zone [4]int, w, h int) {
	x0, y0 := rng.Intn(w), rng.Intn(h)
	x1, y1 := rng.Intn(w), rng.Intn(h)
	c := randomInk(rng)
	c.A = 255
	width := 1 + rng.Intn(3)

	dx := absi(x1 - x0)
	dy := absi(y1 - y0)
	sx := 1
	if x0 > x1 {
		sx = -1
	}
	sy := 1
	if y0 > y1 {
		sy = -1
	}
	err := dx - dy
	for {
		if !inZone(x0, y0, zone) {
			paintDot(canvas, x0, y0, width, c)
		}
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 > -dy {
			err -= dy
			x0 += sx
		}
		if e2 < dx {
			err += dx
			y0 += sy
		}
	}
}

// drawInterferenceCurve draws a random cubic Bezier curve with random color,
// skipping pixels inside the exclusion zone.
func drawInterferenceCurve(rng *rand.Rand, canvas *image.NRGBA, zone [4]int, w, h int) {
	p0x, p0y := float64(rng.Intn(w)), float64(rng.Intn(h))
	p1x, p1y := float64(rng.Intn(w)), float64(rng.Intn(h))
	p2x, p2y := float64(rng.Intn(w)), float64(rng.Intn(h))
	p3x, p3y := float64(rng.Intn(w)), float64(rng.Intn(h))
	c := randomInk(rng)
	c.A = 255
	width := 1 + rng.Intn(2)

	steps := 200
	for i := 0; i <= steps; i++ {
		t := float64(i) / float64(steps)
		mt := 1 - t
		x := mt*mt*mt*p0x + 3*mt*mt*t*p1x + 3*mt*t*t*p2x + t*t*t*p3x
		y := mt*mt*mt*p0y + 3*mt*mt*t*p1y + 3*mt*t*t*p2y + t*t*t*p3y
		ix, iy := int(x), int(y)
		if !inZone(ix, iy, zone) {
			paintDot(canvas, ix, iy, width, c)
		}
	}
}

// drawInterferenceText draws a random word at a position outside the exclusion
// zone. Returns the TextOut metadata (empty Text if no valid position found).
func drawInterferenceText(rng *rand.Rand, lib *fontLib, canvas *image.NRGBA, zone [4]int, w, h int) TextOut {
	size := 12 + rng.Intn(16)
	f, err := lib.faceAt(float64(size))
	if err != nil {
		return TextOut{}
	}
	c := randomInk(rng)
	c.A = 255
	word := fontWord(rng)
	adv := int(font.MeasureString(f, word) >> 6)

	for attempt := 0; attempt < 10; attempt++ {
		x := rng.Intn(maxi(1, w-adv))
		y := size + rng.Intn(maxi(1, h-size*2))
		// text bbox: [x, y-size, x+adv, y+size]
		if !rectOverlap([4]int{x, y - size, x + adv, y + size}, zone) {
			textDrawH(canvas, f, c, word, x, y)
			return TextOut{
				Side:  "interference",
				Text:  word,
				Size:  size,
				X:     x,
				Y:     y,
				Color: hexColorFmt(c),
			}
		}
	}
	return TextOut{}
}
