package main

import (
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"strconv"
)

// Px is one pixel of a sprite (in sprite-local coordinates during feature
// extraction, or in image coordinates during segmentation).
type Px struct {
	X, Y    int
	R, G, B uint8
	A       float64 // coverage weight (alpha, or fg-distance for queries)
}

func loadPNG(path string) (*image.NRGBA, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		return nil, err
	}
	b := img.Bounds()
	nrgba := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			nrgba.Set(x, y, color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)))
		}
	}
	return nrgba, nil
}

// refPixels extracts the opaque pixels (alpha > 4/255) of a source icon.
func refPixels(img *image.NRGBA) []Px {
	b := img.Bounds()
	var px []Px
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			i := img.PixOffset(x, y)
			if img.Pix[i+3] > 4 {
				px = append(px, Px{X: x - b.Min.X, Y: y - b.Min.Y, R: img.Pix[i], G: img.Pix[i+1], B: img.Pix[i+2], A: float64(img.Pix[i+3]) / 255})
			}
		}
	}
	if len(px) == 0 {
		return nil
	}
	return px
}

// ---------- segmentation ----------

func rgbDist(a, b Px) float64 {
	dr := float64(a.R) - float64(b.R)
	dg := float64(a.G) - float64(b.G)
	db := float64(a.B) - float64(b.B)
	return math.Sqrt(dr*dr+dg*dg+db*db) / 255
}

// estimateBG finds the dominant flat background color (mode of binned colors).
// adaptiveFGMask thresholds the query against its estimated background using an
// adaptive cutoff derived from the pixel-to-background distance distribution,
// instead of a fixed 0.16. It returns the mask and the foreground pixel count.
func adaptiveFGMask(img *image.NRGBA, bg Px) ([]bool, int) {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	mask := make([]bool, w*h)
	// Histogram of distance-to-bg over all pixels (256 bins, distances in
	// [0,1] mapped to [0,255]).
	const bins = 256
	var dist [bins]int
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := img.PixOffset(x, y)
			p := Px{R: img.Pix[i], G: img.Pix[i+1], B: img.Pix[i+2]}
			d := rgbDist(p, bg)
			bi := int(d * bins)
			if bi >= bins {
				bi = bins - 1
			}
			dist[bi]++
		}
	}

	th := adaptiveDistThreshold(dist)
	fg := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := img.PixOffset(x, y)
			p := Px{R: img.Pix[i], G: img.Pix[i+1], B: img.Pix[i+2]}
			if rgbDist(p, bg) > float64(th)/256 {
				mask[y*w+x] = true
				fg++
			}
		}
	}
	return mask, fg
}

// adaptiveDistThreshold picks a cutoff index in the distance histogram. The
// background is a dense low-distance peak; the foreground is a second mass at
// higher distance. We walk upward from the background peak and pick the first
// bin that is a local valley (a gap), falling back to a floor of ~0.12 so we
// never lose thin AA strokes. dist is in units of bins (0..255 == distance
// 0..1).
func adaptiveDistThreshold(dist [256]int) int {
	// find the background peak (most pixels) in the low half.
	bgPeak := 0
	for i := 1; i < 256; i++ {
		if dist[i] > dist[bgPeak] {
			bgPeak = i
		}
	}
	// walk up from bgPeak; the threshold is the first significant valley.
	for i := bgPeak + 1; i < 255; i++ {
		if dist[i] < dist[i-1] && dist[i] < dist[i+1] && dist[i] < dist[bgPeak]/8 {
			return i
		}
	}
	// no clear valley: fall back to a fixed floor that keeps thin strokes.
	floor := 30 // ~0.12 * 256
	if low := bgPeak + 12; low > floor {
		floor = low
	}
	return floor
}

func estimateBG(img *image.NRGBA) Px {
	// bin: 16x16x16 = 4096 bins
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	const dim = 16
	var hist [dim * dim * dim]int
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := img.PixOffset(x, y)
			bi := int(img.Pix[i]>>4)*dim*dim + int(img.Pix[i+1]>>4)*dim + int(img.Pix[i+2]>>4)
			hist[bi]++
		}
	}
	bestbi, best := 0, -1
	for i, c := range hist {
		if c > best {
			best, bestbi = c, i
		}
	}
	r := (bestbi/(dim*dim))*256/dim + 8
	g := ((bestbi/dim)%dim)*256/dim + 8
	bl := (bestbi%dim)*256/dim + 8
	return Px{R: uint8(r), G: uint8(g), B: uint8(bl)}
}

// extractQuery isolates the sprite from a query canvas: bg detection, distance
// mask, morphology (dilate+erode) to heal gaps, then keeps the largest blob.
func extractQuery(img *image.NRGBA) []Px {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	bg := estimateBG(img)

	// fg mask with threshold relative to bg flatness. The threshold is
	// adaptive: we histogram each pixel's distance to the estimated background
	// and pick a value in the gap between the background-noise cluster (tiny
	// distances) and the foreground cluster. A fixed threshold fails when the
	// sprite is a dark color on a dark background (e.g. #434343 icon on
	// #393560 bg) where the naive 0.16 Euclidean cutoff is below the true
	// icon-to-bg distance — it would discard most of the icon.
	mask, fgCount := adaptiveFGMask(img, bg)
	if fgCount == 0 {
		return nil
	}

	// morphological close: dilate D + erode D heals gaps in thin strokes while
	// keeping interiors. D=1 handles single-pixel AA gaps; a larger D (via
	// CLOSED) merges fragments of a thin rotated sprite at the cost of erasing
	// the thinnest interiors.
	closeD := 3
	if v := os.Getenv("CLOSED"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			closeD = n
		}
	}
	for pass := 0; pass < closeD; pass++ {
		mask = dilate(mask, w, h)
	}
	for pass := 0; pass < closeD; pass++ {
		mask = erode(mask, w, h)
	}

	// connected components: keep the largest fully-interior component plus any
	// other interior component at least blobMinFrac of its area (a rotated thin
	// sprite fragments into several pieces; edge text / noise blobs are far
	// smaller). Fall back to the largest component overall if nothing is
	// interior.
	comps := comps(mask, w, h)
	if len(comps) == 0 {
		return nil
	}
	blobMinFrac := 0.15
	if v := os.Getenv("BLOBMIN"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 && f <= 1 {
			blobMinFrac = f
		}
	}
	var interior [][]int
	for _, c := range comps {
		if !touchesBorder(c, w, h) {
			interior = append(interior, c)
		}
	}
	var best []int
	if len(interior) > 0 {
		maxArea := 0
		for _, c := range interior {
			if len(c) > maxArea {
				maxArea = len(c)
			}
		}
		for _, c := range interior {
			if len(c) >= int(float64(maxArea)*blobMinFrac) {
				best = append(best, c...)
			}
		}
	}
	if best == nil {
		// everything touches the border, or all interior blobs are tiny:
		// keep the largest blob overall
		best = comps[0]
		for _, c := range comps[1:] {
			if len(c) > len(best) {
				best = c
			}
		}
	}

	px := make([]Px, 0, len(best))
	uniformA := os.Getenv("UA") == "1"
	for _, idx := range best {
		x, y := idx%w, idx/w
		i := img.PixOffset(x, y)
		p := Px{X: x, Y: y, R: img.Pix[i], G: img.Pix[i+1], B: img.Pix[i+2]}
		if uniformA {
			p.A = 1
		} else {
			d := rgbDist(p, bg) / 0.2
			if d > 1 {
				d = 1
			}
			p.A = d
		}
		px = append(px, p)
	}
	return px
}

// touchesBorder reports whether any pixel of the component lies on the canvas
// boundary.
func touchesBorder(idx []int, w, h int) bool {
	for _, p := range idx {
		if p < w || p%w == 0 || p%w == w-1 || p >= (h-1)*w {
			return true
		}
	}
	return false
}

// label computes connected components; returns per-component pixel indices.
func label(mask []bool, w, h int) [][]int { return comps(mask, w, h) }

func dilate(mask []bool, w, h int) []bool {
	out := make([]bool, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if mask[y*w+x] {
				for dy := -1; dy <= 1; dy++ {
					for dx := -1; dx <= 1; dx++ {
						nx, ny := x+dx, y+dy
						if nx >= 0 && ny >= 0 && nx < w && ny < h {
							out[ny*w+nx] = true
						}
					}
				}
			}
		}
	}
	return out
}

func erode(mask []bool, w, h int) []bool {
	out := make([]bool, w*h)
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			all := true
			for dy := -1; dy <= 1 && all; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if !mask[(y+dy)*w+(x+dx)] {
						all = false
						break
					}
				}
			}
			out[y*w+x] = all
		}
	}
	return out
}

func maskOf(px []Px, w, h int) []bool {
	m := make([]bool, w*h)
	for _, p := range px {
		m[p.Y*w+p.X] = true
	}
	return m
}

// comps splits the mask into 8-connected components, returning only the
// component pixel index sets (each a slice of idx).
func comps(mask []bool, w, h int) [][]int {
	seen := make([]bool, w*h)
	var out [][]int
	for i := range mask {
		if !mask[i] || seen[i] {
			continue
		}
		cur := make([]int, 0, 64)
		stack := []int{i}
		seen[i] = true
		for len(stack) > 0 {
			p := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			cur = append(cur, p)
			x, y := p%w, p/w
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					nx, ny := x+dx, y+dy
					if nx < 0 || ny < 0 || nx >= w || ny >= h {
						continue
					}
					ni := ny*w + nx
					if mask[ni] && !seen[ni] {
						seen[ni] = true
						stack = append(stack, ni)
					}
				}
			}
		}
		out = append(out, cur)
	}
	return out
}
