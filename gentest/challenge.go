package main

import (
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"math"
	"math/rand"
	"os"
	"path/filepath"
)

// ChallengeSample is the manifest entry for one challenge query.
type ChallengeSample struct {
	Image    string `json:"image"`
	Src      string `json:"src"`       // primary target source filename
	SourceID string `json:"source_id"`
	Type     string `json:"type"`     // position|scale|rotation|occlusion|composition|hard-negative
	Param    string `json:"param"`    // human-readable variant label
	TrueRef  string `json:"true_ref"` // target ref filename in the gallery
}

// runChallenge generates a controlled challenge test set that probes the
// model's invariance to specific transforms. Each challenge query is a
// rendering of a source icon with a FIXED transform (position, scale, rotation,
// occlusion) or a composition (multiple icons). A manifest records the
// challenge type and target ref so the search tool can evaluate per-type
// recall. The challenge set does NOT participate in training.
func runChallenge(rng *rand.Rand, lib *fontLib, srcDir, outDir string, perType int) {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	srcs, err := listPNGs(srcDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	// preload all source sprites
	type srcIcon struct {
		name   string
		sprite *image.NRGBA
		box    [4]int
	}
	icons := make([]srcIcon, 0, len(srcs))
	for _, fn := range srcs {
		img, err := loadPNG(filepath.Join(srcDir, fn))
		if err != nil {
			continue
		}
		box := trimAlpha(img)
		if box == nil {
			continue
		}
		icons = append(icons, srcIcon{name: fn, sprite: cutBox(img, *box), box: *box})
	}
	if len(icons) == 0 {
		fmt.Fprintf(os.Stderr, "error: no usable source icons\n")
		os.Exit(1)
	}

	var manifest []ChallengeSample
	idx := 0

	addEntry := func(src, sid, ctype, param, trueRef string, canvas *image.NRGBA) {
		name := fmt.Sprintf("chall_%05d.png", idx)
		idx++
		out := filepath.Join(outDir, name)
		if err := savePNG(out, canvas); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		manifest = append(manifest, ChallengeSample{
			Image: name, Src: src, SourceID: sid,
			Type: ctype, Param: param, TrueRef: trueRef,
		})
	}

	canvasSide := 480 // fixed canvas for all challenge queries

	// --- position invariance ---
	positions := []struct {
		label string
		fx, fy float64
	}{
		{"left", 0.25, 0.50},
		{"center", 0.50, 0.50},
		{"right", 0.75, 0.50},
		{"top", 0.50, 0.25},
		{"bottom", 0.50, 0.75},
	}
	for _, ic := range icons {
		for _, pos := range positions {
			canvas := image.NewNRGBA(image.Rect(0, 0, canvasSide, canvasSide))
			bg := randColor(rng, 0.15, 0.85)
			flatFill(canvas, bg)
			scale := 0.55 * float64(canvasSide) / float64(maxi(ic.sprite.Bounds().Dx(), ic.sprite.Bounds().Dy()))
			ox := pos.fx * float64(canvasSide)
			oy := pos.fy * float64(canvasSide)
			affinePaint(canvas, ic.sprite, scale, 0, ox, oy)
			addEntry(ic.name, sourceIDOf(ic.name), "position", pos.label, ic.name, canvas)
		}
	}

	// --- scale invariance ---
	scales := []struct {
		label string
		frac  float64
	}{
		{"0.25", 0.25},
		{"0.50", 0.50},
		{"0.75", 0.75},
		{"1.00", 1.00},
	}
	for _, ic := range icons {
		for _, sc := range scales {
			canvas := image.NewNRGBA(image.Rect(0, 0, canvasSide, canvasSide))
			bg := randColor(rng, 0.15, 0.85)
			flatFill(canvas, bg)
			scale := sc.frac * float64(canvasSide) / float64(maxi(ic.sprite.Bounds().Dx(), ic.sprite.Bounds().Dy()))
			// clamp so the scaled sprite fits
			spriteMax := float64(maxi(ic.sprite.Bounds().Dx(), ic.sprite.Bounds().Dy()))
			if spriteMax*scale > float64(canvasSide-4) {
				scale = float64(canvasSide-4) / spriteMax
			}
			ox := float64(canvasSide) / 2
			oy := float64(canvasSide) / 2
			affinePaint(canvas, ic.sprite, scale, 0, ox, oy)
			addEntry(ic.name, sourceIDOf(ic.name), "scale", sc.label, ic.name, canvas)
		}
	}

	// --- rotation invariance ---
	rotations := []struct {
		label string
		deg   float64
	}{
		{"0", 0},
		{"45", 45},
		{"90", 90},
		{"135", 135},
		{"180", 180},
		{"270", 270},
	}
	for _, ic := range icons {
		for _, rot := range rotations {
			canvas := image.NewNRGBA(image.Rect(0, 0, canvasSide, canvasSide))
			bg := randColor(rng, 0.15, 0.85)
			flatFill(canvas, bg)
			rad := rot.deg * math.Pi / 180
			sw, sh := float64(ic.sprite.Bounds().Dx()), float64(ic.sprite.Bounds().Dy())
			rotW := sw*math.Abs(math.Cos(rad)) + sh*math.Abs(math.Sin(rad))
			rotH := sw*math.Abs(math.Sin(rad)) + sh*math.Abs(math.Cos(rad))
			spriteMax := math.Max(rotW, rotH)
			scale := 0.55 * float64(canvasSide) / spriteMax
			ox := float64(canvasSide) / 2
			oy := float64(canvasSide) / 2
			affinePaint(canvas, ic.sprite, scale, rot.deg, ox, oy)
			addEntry(ic.name, sourceIDOf(ic.name), "rotation", rot.label, ic.name, canvas)
		}
	}

	// --- occlusion ---
	occlusions := []struct {
		label string
		frac  float64
	}{
		{"0.10", 0.10},
		{"0.20", 0.20},
		{"0.30", 0.30},
	}
	for _, ic := range icons {
		for _, occ := range occlusions {
			canvas := image.NewNRGBA(image.Rect(0, 0, canvasSide, canvasSide))
			bg := randColor(rng, 0.15, 0.85)
			flatFill(canvas, bg)
			scale := 0.55 * float64(canvasSide) / float64(maxi(ic.sprite.Bounds().Dx(), ic.sprite.Bounds().Dy()))
			ox := float64(canvasSide) / 2
			oy := float64(canvasSide) / 2
			affinePaint(canvas, ic.sprite, scale, 0, ox, oy)
			rect := spriteScreenRect(ic.sprite.Bounds().Dx(), ic.sprite.Bounds().Dy(), scale, 0, ox, oy)
			// deterministic occlusion at the given fraction
			drawFixedOcclusion(rng, canvas, rect, occ.frac)
			addEntry(ic.name, sourceIDOf(ic.name), "occlusion", occ.label, ic.name, canvas)
		}
	}

	// --- composition interference (A + B, A+B+C, ...) ---
	for _, ic := range icons {
		for nExtra := 1; nExtra <= 4; nExtra++ {
			canvas := image.NewNRGBA(image.Rect(0, 0, canvasSide, canvasSide))
			bg := randColor(rng, 0.15, 0.85)
			flatFill(canvas, bg)

			// primary icon: large, centered-ish
			scale := 0.50 * float64(canvasSide) / float64(maxi(ic.sprite.Bounds().Dx(), ic.sprite.Bounds().Dy()))
			ox := float64(canvasSide) * 0.40
			oy := float64(canvasSide) * 0.50
			affinePaint(canvas, ic.sprite, scale, 0, ox, oy)

			// extra icons: smaller, scattered
			for k := 0; k < nExtra; k++ {
				j := rng.Intn(len(icons))
				ex := icons[j]
				escale := 0.20 * float64(canvasSide) / float64(maxi(ex.sprite.Bounds().Dx(), ex.sprite.Bounds().Dy()))
				eox := float64(canvasSide) * (0.20 + 0.60*rng.Float64())
				eoy := float64(canvasSide) * (0.20 + 0.60*rng.Float64())
				affinePaint(canvas, ex.sprite, escale, 0, eox, eoy)
			}
			param := fmt.Sprintf("A+%d", nExtra)
			addEntry(ic.name, sourceIDOf(ic.name), "composition", param, ic.name, canvas)
		}
	}

	// --- hard-negative pairs ---
	// Render icon A alongside a visually similar but different icon B.
	for i, ic := range icons {
		// pick a different icon at random (the search side will filter by split)
		j := (i + 1 + rng.Intn(len(icons)-1)) % len(icons)
		if j == i {
			j = (i + 1) % len(icons)
		}
		other := icons[j]
		canvas := image.NewNRGBA(image.Rect(0, 0, canvasSide, canvasSide))
		bg := randColor(rng, 0.15, 0.85)
		flatFill(canvas, bg)

		// A on the left, similar B on the right
		scaleA := 0.40 * float64(canvasSide) / float64(maxi(ic.sprite.Bounds().Dx(), ic.sprite.Bounds().Dy()))
		affinePaint(canvas, ic.sprite, scaleA, 0, float64(canvasSide)*0.30, float64(canvasSide)*0.50)
		scaleB := 0.40 * float64(canvasSide) / float64(maxi(other.sprite.Bounds().Dx(), other.sprite.Bounds().Dy()))
		affinePaint(canvas, other.sprite, scaleB, 0, float64(canvasSide)*0.70, float64(canvasSide)*0.50)

		addEntry(ic.name, sourceIDOf(ic.name), "hard-negative", "pair", ic.name, canvas)
	}

	mf := filepath.Join(outDir, "challenge_manifest.json")
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	if err := os.WriteFile(mf, data, 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("challenge set: %d queries written to %q (manifest: %s)\n", len(manifest), outDir, mf)
}

// drawFixedOcclusion draws a single semi-opaque rectangle covering the given
// fraction of the sprite's screen rect.
func drawFixedOcclusion(rng *rand.Rand, canvas *image.NRGBA, spriteRect [4]float64, frac float64) {
	x0, y0, x1, y1 := spriteRect[0], spriteRect[1], spriteRect[2], spriteRect[3]
	rw, rh := x1-x0, y1-y0
	if rw <= 2 || rh <= 2 {
		return
	}
	area := rw * rh * frac
	if area < 4 {
		return
	}
	ow := math.Sqrt(area)
	oh := ow * (0.5 + rng.Float64())
	cx := x0 + rw*0.5
	cy := y0 + rh*0.5
	c := color.NRGBA{
		R: uint8(rng.Intn(256)),
		G: uint8(rng.Intn(256)),
		B: uint8(rng.Intn(256)),
		A: 180,
	}
	rx0, ry0 := int(cx-ow/2), int(cy-oh/2)
	rx1, ry1 := int(cx+ow/2), int(cy+oh/2)
	for yy := ry0; yy < ry1; yy++ {
		for xx := rx0; xx < rx1; xx++ {
			overPaint(canvas, xx, yy, c)
		}
	}
}
