// Command gen builds the attention-net test set: synthetic canvases, each a
// random source icon pasted on a solid background with optional edge text, and
// per-patch ground-truth masks covering only the icon.
//
// Usage: go run ./cmd/gen [-src ../test_pngs] [-n 1200] [-seed 0] [-out data]
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"attnnet"
)

var (
	srcDir = flag.String("src", "../test_pngs", "directory of source icons")
	count  = flag.Int("n", 1200, "number of samples to generate")
	seed   = flag.Int64("seed", 0, "random seed (0 = time based)")
	outDir = flag.String("out", "data", "output directory")
)

func main() {
	flag.Parse()

	rngSrc := time.Now().UnixNano()
	if *seed != 0 {
		rngSrc = *seed
	}
	rng := rand.New(rand.NewSource(rngSrc))
	cfg := attnnet.Default()
	fmt.Printf("generating %d samples from %q (seed=%d, %dx%d -> %d patches)\n",
		*count, *srcDir, rngSrc, cfg.ImageSize, cfg.ImageSize, cfg.NTok())

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

	set := attnnet.NewSet(cfg, *count)
	var manifest []Sample

	done := 0
	for i := 0; i < *count; i++ {
		src := srcs[rng.Intn(len(srcs))]
		orig, err := loadPNG(filepath.Join(*srcDir, src))
		if err != nil {
			log.Fatal(err)
		}
		box := trimAlpha(orig)
		if box == nil {
			continue
		}
		sprite := cutBox(orig, *box)
		canvas, spec := renderOne(rng, lib, sprite, *box)
		img := downscale(canvas, cfg.ImageSize)
		rect := spriteScreenRect(sprite.Bounds().Dx(), sprite.Bounds().Dy(),
			spec.Scale, spec.Rotation, float64(spec.Translate[0])+float64(spec.Canvas[0])/2,
			float64(spec.Translate[1])+float64(spec.Canvas[1])/2)
		mask := buildMask(cfg, spec.Canvas, rect)
		copy(set.Img(i), img)
		copy(set.Mask(i), mask)
		spec.Image = fmt.Sprintf("sample_%05d", i)
		spec.Src = src
		manifest = append(manifest, spec)
		done++
	}

	set.N = done
	path := filepath.Join(*outDir, "testset.bin")
	if err := set.Save(path); err != nil {
		log.Fatal(err)
	}
	writeManifest(filepath.Join(*outDir, "manifest.json"), manifest)
	fmt.Printf("done: %d samples -> %s (+ manifest.json)\n", done, path)
}

