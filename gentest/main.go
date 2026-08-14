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
	"runtime"
	"sync"
	"time"
)

var (
	srcDir     = flag.String("src", "test_pngs", "directory of source icons")
	outDir     = flag.String("out", "test_set", "output directory for samples")
	count      = flag.Int("n", 120, "number of samples to generate")
	seed       = flag.Int64("seed", 0, "random seed (0 = time based)")
	minS       = flag.Float64("min-canvas", 320, "canvas side lower bound in px")
	maxS       = flag.Float64("max-canvas", 560, "canvas side upper bound in px")
	gl         = flag.Float64("sprite-lo", 0.35, "sprite side fraction of canvas side (min)")
	gh         = flag.Float64("sprite-hi", 0.85, "sprite side fraction of canvas side (max)")
	allSd      = flag.Bool("all-sides", false, "put text on every side (default: random subset)")
	noRot      = flag.Bool("norot", false, "disable sprite rotation (debug: isolate rotation interference)")
	augRefsDir = flag.String("augrefs", "", "if set, synthesize decorated reference icons into this dir instead of samples")
	augPer     = flag.Int("aug-per-src", 3, "decorated reference variants per source icon")
	augSeed    = flag.Int64("augseed", 0, "random seed for reference augmentation (0 = time based)")

	// Augmentation config flags (override DefaultAugConfig).
	augScaleMin   = flag.Float64("aug-scale-min", 0.25, "aug ref: min sprite scale")
	augScaleMax   = flag.Float64("aug-scale-max", 1.0, "aug ref: max sprite scale")
	augRotProb    = flag.Float64("aug-rot-prob", 0.80, "aug ref: rotation probability")
	augRotMax     = flag.Float64("aug-rot-max", 360.0, "aug ref: max rotation degrees")
	augFlipProb   = flag.Float64("aug-flip-prob", 0.15, "aug ref: horizontal flip probability")
	augPosRange   = flag.Float64("aug-pos-range", 0.30, "aug ref: position offset range (fraction of slack)")
	augOccProb    = flag.Float64("aug-occ-prob", 0.30, "aug ref: occlusion probability")
	augOccMin     = flag.Float64("aug-occ-min", 0.10, "aug ref: min occlusion fraction")
	augOccMax     = flag.Float64("aug-occ-max", 0.30, "aug ref: max occlusion fraction")
	augEasyRatio  = flag.Float64("aug-easy-ratio", 0.30, "aug ref: easy difficulty fraction")
	augMedRatio   = flag.Float64("aug-med-ratio", 0.50, "aug ref: medium difficulty fraction")
	augHardRatio  = flag.Float64("aug-hard-ratio", 0.20, "aug ref: hard difficulty fraction")

	challengeDir = flag.String("challenge", "", "if set, generate a challenge test set into this dir")
)

// augCfgFromFlags builds an AugConfig from flag values.
func augCfgFromFlags() AugConfig {
	return AugConfig{
		ScaleMin:      *augScaleMin,
		ScaleMax:      *augScaleMax,
		ScaleBeta:     2.0,
		RotationProb:  *augRotProb,
		RotationMax:   *augRotMax,
		FlipProb:      *augFlipProb,
		PositionRange: *augPosRange,
		OcclusionProb: *augOccProb,
		OcclusionMin:  *augOccMin,
		OcclusionMax:  *augOccMax,
		EasyRatio:     *augEasyRatio,
		MediumRatio:   *augMedRatio,
		HardRatio:     *augHardRatio,
	}
}

// Sample is the ground-truth record for a generated image.
type Sample struct {
	Image     string    `json:"image"`
	Src       string    `json:"src"`
	SourceID  string    `json:"source_id,omitempty"` // normalized source for split isolation
	Crop      [4]int    `json:"crop"` // [x, y, w, h] trimmed sprite box in source
	Canvas    [2]int    `json:"canvas"`
	BGHex     string    `json:"bg_hex"`
	Rotation  float64   `json:"rotation"`
	Scale     float64   `json:"scale"`
	Translate [2]int    `json:"translate"`
	Texts     []TextOut `json:"texts"`
}

var collected []Sample

// parFor runs fn over [0,n) across runtime.GOMAXPROCS workers.
func parFor(n int, fn func(i int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		for i := 0; i < n; i++ {
			fn(i)
		}
		return
	}
	var wg sync.WaitGroup
	next := make(chan int)
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for i := range next {
				fn(i)
			}
		}()
	}
	for i := 0; i < n; i++ {
		next <- i
	}
	close(next)
	wg.Wait()
}

func main() {
	flag.Parse()

	rngSrc := time.Now().UnixNano()
	if *seed != 0 {
		rngSrc = *seed
	}
	rng := rand.New(rand.NewSource(rngSrc))
	if *challengeDir != "" {
		lib, err := loadFont()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("generating challenge set from %q -> %q\n", *srcDir, *challengeDir)
		runChallenge(rng, lib, *srcDir, *challengeDir, 0)
		return
	}
	if *augRefsDir != "" {
		lib, err := loadFont()
		if err != nil {
			log.Fatal(err)
		}
		arng := rng
		if *augSeed != 0 {
			arng = rand.New(rand.NewSource(*augSeed))
		}
		cfg := augCfgFromFlags()
		fmt.Printf("augmenting refs from %q -> %q perSrc=%d  scale=[%.2f,%.2f] rotProb=%.2f rotMax=%.0f flipProb=%.2f occProb=%.2f  easy=%.0f%% med=%.0f%% hard=%.0f%%\n",
			*srcDir, *augRefsDir, *augPer,
			cfg.ScaleMin, cfg.ScaleMax, cfg.RotationProb, cfg.RotationMax,
			cfg.FlipProb, cfg.OcclusionProb,
			100*cfg.EasyRatio, 100*cfg.MediumRatio, 100*cfg.HardRatio)
		runAugRefs(arng, lib, *srcDir, *augRefsDir, *augPer, cfg)
		return
	}
	fmt.Printf("generating %d samples from %q, seed=%d (parallel, %d workers)\n", *count, *srcDir, rngSrc, runtime.GOMAXPROCS(0))

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

	// Pre-assign source choices and per-sample RNG seeds in the main goroutine
	// so the output is deterministic and workers are stateless.
	srcChoices := make([]string, *count)
	for i := 0; i < *count; i++ {
		srcChoices[i] = srcs[rng.Intn(len(srcs))]
	}

	results := make([]Sample, *count)
	parFor(*count, func(i int) {
		src := srcChoices[i]
		// per-sample RNG: deterministic, independent of worker scheduling
		srng := rand.New(rand.NewSource(rngSrc*7919 + int64(i)))
		orig, err := loadPNG(filepath.Join(*srcDir, src))
		if err != nil {
			return
		}
		box := trimAlpha(orig)
		if box == nil {
			return // fully transparent sprite
		}
		sprite := cutBox(orig, *box)
		canvas, spec := renderQuery(srng, lib, sprite, *box)
		spec.Src = src
		spec.SourceID = sourceIDOf(src)

		name := fmt.Sprintf("sample_%05d.png", i)
		if err := savePNG(filepath.Join(*outDir, name), canvas); err != nil {
			log.Fatal(err)
		}
		spec.Image = name
		results[i] = spec
	})

	// compact: remove empty entries (transparent sources that were skipped)
	for _, s := range results {
		if s.Image != "" {
			collected = append(collected, s)
		}
	}

	if err := writeManifest(); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("done: %d samples written to %q (manifest.json)\n", len(collected), *outDir)
}