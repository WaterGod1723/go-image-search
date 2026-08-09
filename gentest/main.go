// Command gentest generates a randomized test set from the source icons in
// test_pngs, so an image-retrieval algorithm can be evaluated with known
// ground truth.
//
// Each generated sample:
//  1. picks a random source icon and trims its transparent border;
//  2. applies a random combination of transforms (each optional):
//       - background color change,
//       - rotation (around the sprite center),
//       - scale + translation on the canvas,
//       - random text close to the borders (font height <= 0.3*cmin);
//  3. is written to the output dir; manifest.json records, per sample, the
//     source file, the crop, and every transform used.
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"
)

var (
	srcDir = flag.String("src", "test_pngs", "directory of source icons")
	outDir = flag.String("out", "test_set", "output directory for samples")
	count  = flag.Int("n", 120, "number of samples to generate")
	seed   = flag.Int64("seed", 0, "random seed (0 = time based)")
	minS   = flag.Float64("min-canvas", 320, "canvas side lower bound in px")
	maxS   = flag.Float64("max-canvas", 560, "canvas side upper bound in px")
	gl     = flag.Float64("sprite-lo", 0.35, "sprite side fraction of canvas side (min)")
	gh     = flag.Float64("sprite-hi", 0.85, "sprite side fraction of canvas side (max)")
	allSd  = flag.Bool("all-sides", false, "put text on every side (default: random subset)")
	noRot  = flag.Bool("no-rot", false, "disable rotation interference (rot always 0)")
)

// Sample is the ground-truth record for a generated image.
type Sample struct {
	Image     string    `json:"image"`
	Src       string    `json:"src"`
	Crop      [4]int    `json:"crop"` // [x, y, w, h] trimmed sprite box in source
	Canvas    [2]int    `json:"canvas"`
	BGHex     string    `json:"bg_hex"`
	Rotation  float64   `json:"rotation"`
	Scale     float64   `json:"scale"`
	Translate [2]int    `json:"translate"`
	Texts     []TextOut `json:"texts"`
}

var collected []Sample

func main() {
	flag.Parse()

	rngSrc := time.Now().UnixNano()
	if *seed != 0 {
		rngSrc = *seed
	}
	rng := rand.New(rand.NewSource(rngSrc))
	fmt.Printf("generating %d samples from %q, seed=%d\n", *count, *srcDir, rngSrc)

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		log.Fatal(err)
	}

	srcs, err := listPNGs(*srcDir)
	if err != nil {
		log.Fatal(err)
	}
	if len(srcs) == 0 {
		log.Fatalf("no png files found in %s", *srcDir)
	}

	lib, err := loadFont()
	if err != nil {
		log.Fatal(err)
	}

	for i := 0; i < *count; i++ {
		src := srcs[rng.Intn(len(srcs))]
		orig, err := loadPNG(filepath.Join(*srcDir, src))
		if err != nil {
			log.Fatal(err)
		}
		box := trimAlpha(orig)
		if box == nil {
			continue // fully transparent sprite
		}
		sprite := cutBox(orig, *box)
		canvas, spec := renderSample(rng, lib, sprite, *box)
		spec.Src = src

		name := fmt.Sprintf("sample_%05d.png", i)
		if err := savePNG(filepath.Join(*outDir, name), canvas); err != nil {
			log.Fatal(err)
		}
		spec.Image = name
		collected = append(collected, spec)
	}

	if err := writeManifest(); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("done: %d samples written to %q (manifest.json)\n", len(collected), *outDir)
}