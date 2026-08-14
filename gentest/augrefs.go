package main

import (
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
)

// AugConfig holds all augmentation parameters for decorated reference icons.
// Every field is configurable via flags so experiments can be reproduced.
type AugConfig struct {
	ScaleMin      float64 // minimum sprite scale relative to canvas
	ScaleMax      float64 // maximum sprite scale relative to canvas
	ScaleBeta     float64 // >1 biases towards mid-size (triangular peak)
	RotationProb  float64 // probability of applying rotation
	RotationMax   float64 // max rotation in degrees (0-360)
	FlipProb      float64 // probability of horizontal flip
	PositionRange float64 // ±fraction of canvas for sprite center offset
	OcclusionProb float64 // probability of applying partial occlusion
	OcclusionMin  float64 // min occlusion area fraction of sprite screen rect
	OcclusionMax  float64 // max occlusion area fraction of sprite screen rect
	EasyRatio     float64 // fraction of samples at easy difficulty
	MediumRatio   float64 // fraction of samples at medium difficulty
	HardRatio     float64 // fraction of samples at hard difficulty
}

// DefaultAugConfig returns sensible defaults that avoid all-position-centered /
// fixed-size shortcuts while keeping enough easy samples for stable training.
func DefaultAugConfig() AugConfig {
	return AugConfig{
		ScaleMin:      0.25,
		ScaleMax:      1.0,
		ScaleBeta:     2.0,
		RotationProb:  0.80,
		RotationMax:   360.0,
		FlipProb:      0.15,
		PositionRange: 0.30,
		OcclusionProb: 0.30,
		OcclusionMin:  0.10,
		OcclusionMax:  0.30,
		EasyRatio:     0.30,
		MediumRatio:   0.50,
		HardRatio:     0.20,
	}
}

// augDifficulty controls the number of distractor shapes/texts.
type augDifficulty int

const (
	diffEasy augDifficulty = iota
	diffMedium
	diffHard
)

func (c AugConfig) pickDifficulty(rng *rand.Rand) augDifficulty {
	r := rng.Float64()
	if r < c.EasyRatio {
		return diffEasy
	}
	if r < c.EasyRatio+c.MediumRatio {
		return diffMedium
	}
	return diffHard
}

func (d augDifficulty) shapeRange() (int, int) {
	switch d {
	case diffEasy:
		return 1, 2
	case diffMedium:
		return 2, 4
	default:
		return 3, 6
	}
}

func (d augDifficulty) textRange() (int, int) {
	switch d {
	case diffEasy:
		return 0, 1
	case diffMedium:
		return 0, 2
	default:
		return 1, 3
	}
}

// AugRefEntry records the provenance of one augmented reference file.
type AugRefEntry struct {
	File     string  `json:"file"`
	Source   string  `json:"source"`    // original source filename
	SourceID string  `json:"source_id"` // normalized identifier for split isolation
	Variant  int     `json:"variant"`
	Difficulty string `json:"difficulty"`
	Scale    float64 `json:"scale"`
	Rotation float64 `json:"rotation"`
	Flip     bool    `json:"flip"`
	Occlusion float64 `json:"occlusion"` // 0 = none, else fraction
}

// randScaleTri produces a scale in [lo, hi] with a triangular peak at the
// midpoint (averaging two uniforms), so mid-size sprites appear more often than
// extreme tiny or extreme huge renders.
func randScaleTri(rng *rand.Rand, lo, hi float64) float64 {
	u := (rng.Float64() + rng.Float64()) / 2
	return lo + (hi-lo)*u
}

// flipH returns a horizontally-flipped copy of sprite.
func flipH(sprite *image.NRGBA) *image.NRGBA {
	sb := sprite.Bounds()
	w, h := sb.Dx(), sb.Dy()
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			out.SetNRGBA(x, y, sprite.NRGBAAt(w-1-x, y))
		}
	}
	return out
}

// affinePaintAlpha draws sprite onto a (possibly transparent) canvas with
// scale s, rotation deg (around sprite center), and center at (ox, oy).
// Unlike affinePaint in render.go this properly composites alpha so the result
// remains a transparent-background decorated ref.
func affinePaintAlpha(dst *image.NRGBA, sprite *image.NRGBA, s, deg, ox, oy float64) {
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
			da := float64(dst.Pix[oi+3]) / 255
			oa := af + da*(1-af)
			if oa <= 1e-6 {
				continue
			}
			dst.Pix[oi] = clampU8((float64(r)*af + float64(dst.Pix[oi])*da*(1-af)) / oa)
			dst.Pix[oi+1] = clampU8((float64(g)*af + float64(dst.Pix[oi+1])*da*(1-af)) / oa)
			dst.Pix[oi+2] = clampU8((float64(b)*af + float64(dst.Pix[oi+2])*da*(1-af)) / oa)
			dst.Pix[oi+3] = clampU8(oa * 255)
		}
	}
}

// sourceIDOf normalizes a source filename into a stable identifier for
// split isolation. For augmented refs the pattern is aug_<base>_vNN.png;
// the base (with .png) is the sourceID. For plain refs, the filename itself.
func sourceIDOf(fn string) string {
	base := strings.TrimSuffix(fn, filepath.Ext(fn))
	if strings.HasPrefix(base, "aug_") {
		rest := strings.TrimPrefix(base, "aug_")
		if idx := strings.LastIndex(rest, "_v"); idx > 0 {
			return rest[:idx] + filepath.Ext(fn)
		}
	}
	return fn
}

// runAugRefs synthesizes decorated reference icons: each source sprite is
// re-rendered on a transparent canvas with random geometry (position, scale,
// rotation, flip, occlusion) and embellished with random colored shapes and
// text at a difficulty level chosen from the config ratios. A manifest
// (augrefs_manifest.json) records the source and augmentation params for each
// file so training can split by source and avoid data leakage.
func runAugRefs(rng *rand.Rand, lib *fontLib, srcDir, outDir string, perSrc int, cfg AugConfig) {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		log.Fatal(err)
	}
	srcs, err := listPNGs(srcDir)
	if err != nil {
		log.Fatal(err)
	}
	var manifest []AugRefEntry
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
		sid := sourceIDOf(fn)
		for v := 0; v < perSrc; v++ {
			entry, canvas := buildAugRef(rng, lib, sprite, fn, sid, v, cfg)
			out := filepath.Join(outDir, entry.File)
			if err := savePNG(out, canvas); err != nil {
				log.Fatal(err)
			}
			manifest = append(manifest, entry)
			written++
		}
	}
	mf := filepath.Join(outDir, "augrefs_manifest.json")
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(mf, data, 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("augmented refs: %d written to %q (manifest: %s)\n", written, outDir, mf)
}

// buildAugRef renders the sprite on a transparent canvas with random geometry
// and difficulty-controlled decorations, returning both the manifest entry and
// the rendered image.
func buildAugRef(rng *rand.Rand, lib *fontLib, sprite *image.NRGBA, srcFile, sourceID string, variant int, cfg AugConfig) (AugRefEntry, *image.NRGBA) {
	sw, sh := sprite.Bounds().Dx(), sprite.Bounds().Dy()

	// --- pick difficulty ---
	diff := cfg.pickDifficulty(rng)

	// --- random scale (triangular, mid-size preferred) ---
	scale := randScaleTri(rng, cfg.ScaleMin, cfg.ScaleMax)
	if scale < 0.05 {
		scale = 0.05
	}

	// --- random rotation ---
	rot := 0.0
	if rng.Float64() < cfg.RotationProb {
		rot = rng.Float64() * cfg.RotationMax
	}
	rad := rot * math.Pi / 180
	cos, sin := math.Abs(math.Cos(rad)), math.Abs(math.Sin(rad))

	// rotated bounding box at unit scale
	rotW := float64(sw)*cos + float64(sh)*sin
	rotH := float64(sw)*sin + float64(sh)*cos
	scaledW := rotW * scale
	scaledH := rotH * scale

	// --- canvas size: proportional to scaled sprite + margin for decorations ---
	margin := 1.25 + rng.Float64()*0.35
	side := int(math.Max(scaledW, scaledH) * margin)
	if side < 96 {
		side = 96
	}
	if side > 1024 {
		side = 1024
	}

	// --- random flip ---
	flipped := false
	workSprite := sprite
	if rng.Float64() < cfg.FlipProb {
		workSprite = flipH(sprite)
		flipped = true
	}

	canvas := image.NewNRGBA(image.Rect(0, 0, side, side))

	// --- random position: center ± positionRange fraction of slack ---
	slackX := float64(side) - scaledW
	slackY := float64(side) - scaledH
	ox := scaledW/2 + slackX/2 + (rng.Float64()-0.5)*slackX*cfg.PositionRange*2
	oy := scaledH/2 + slackY/2 + (rng.Float64()-0.5)*slackY*cfg.PositionRange*2
	if ox < scaledW/2 {
		ox = scaledW / 2
	}
	if oy < scaledH/2 {
		oy = scaledH / 2
	}

	affinePaintAlpha(canvas, workSprite, scale, rot, ox, oy)

	// --- decorations (shapes + texts) by difficulty ---
	nShapeLo, nShapeHi := diff.shapeRange()
	nShapes := nShapeLo
	if nShapeHi > nShapeLo {
		nShapes += rng.Intn(nShapeHi - nShapeLo + 1)
	}
	for i := 0; i < nShapes; i++ {
		drawRandomShape(rng, canvas)
	}

	nTextLo, nTextHi := diff.textRange()
	nTexts := nTextLo
	if nTextHi > nTextLo {
		nTexts += rng.Intn(nTextHi - nTextLo + 1)
	}
	for i := 0; i < nTexts; i++ {
		drawRandomText(rng, lib, canvas, side)
	}

	// --- random occlusion: partially cover the sprite screen rect ---
	occFrac := 0.0
	if rng.Float64() < cfg.OcclusionProb {
		occFrac = cfg.OcclusionMin + rng.Float64()*(cfg.OcclusionMax-cfg.OcclusionMin)
		spriteRect := spriteScreenRect(sw, sh, scale, rot, ox, oy)
		drawOcclusion(rng, canvas, spriteRect, occFrac)
	}

	diffStr := "easy"
	switch diff {
	case diffMedium:
		diffStr = "medium"
	case diffHard:
		diffStr = "hard"
	}

	base := strings.TrimSuffix(srcFile, filepath.Ext(srcFile))
	entry := AugRefEntry{
		File:       fmt.Sprintf("aug_%s_v%02d.png", base, variant),
		Source:     srcFile,
		SourceID:   sourceID,
		Variant:    variant,
		Difficulty: diffStr,
		Scale:      scale,
		Rotation:   rot,
		Flip:       flipped,
		Occlusion:  occFrac,
	}
	return entry, canvas
}

// drawOcclusion paints 1-2 semi-opaque shapes over a fraction of the sprite's
// screen rect so the target icon is partially (never fully) obscured.
func drawOcclusion(rng *rand.Rand, canvas *image.NRGBA, spriteRect [4]float64, frac float64) {
	x0, y0, x1, y1 := spriteRect[0], spriteRect[1], spriteRect[2], spriteRect[3]
	rw := x1 - x0
	rh := y1 - y0
	if rw <= 2 || rh <= 2 {
		return
	}
	area := rw * rh * frac
	if area < 4 {
		return
	}
	nOcc := 1 + rng.Intn(2) // 1-2 occluders
	for i := 0; i < nOcc; i++ {
		ow := math.Sqrt(area / float64(nOcc) * (0.5 + rng.Float64()))
		oh := ow * (0.5 + rng.Float64())
		cx := x0 + rng.Float64()*rw
		cy := y0 + rng.Float64()*rh
		c := randomInk(rng)
		c.A = uint8(150 + rng.Intn(71)) // 150-220 alpha
		switch rng.Intn(2) {
		case 0: // filled rect
			rx0, ry0 := int(cx-ow/2), int(cy-oh/2)
			rx1, ry1 := int(cx+ow/2), int(cy+oh/2)
			for yy := ry0; yy < ry1; yy++ {
				for xx := rx0; xx < rx1; xx++ {
					overPaint(canvas, xx, yy, c)
				}
			}
		default: // filled circle
			r := ow / 2
			for yy := int(cy - r); yy < int(cy+r); yy++ {
				for xx := int(cx - r); xx < int(cx+r); xx++ {
					dx, dy := float64(xx)-cx, float64(yy)-cy
					if dx*dx+dy*dy <= r*r {
						overPaint(canvas, xx, yy, c)
					}
				}
			}
		}
	}
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
