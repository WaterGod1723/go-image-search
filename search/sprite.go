package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
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
	r := (bestbi / (dim * dim)) * 256 / dim + 8
	g := ((bestbi / dim) % dim) * 256 / dim + 8
	bl := (bestbi % dim) * 256 / dim + 8
	return Px{R: uint8(r), G: uint8(g), B: uint8(bl)}
}

// extractQuery isolates the sprite from a query canvas: bg detection, distance
// mask, morphology (dilate+erode) to heal gaps, then keeps the largest blob.
func extractQuery(img *image.NRGBA) []Px {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	bg := estimateBG(img)

	// fg mask with threshold relative to bg flatness
	mask := make([]bool, w*h)
	th := 0.16
	if v := os.Getenv("THRESH"); v != "" {
		fmt.Sscanf(v, "%f", &th)
	}
	fgCount := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := img.PixOffset(x, y)
			p := Px{R: img.Pix[i], G: img.Pix[i+1], B: img.Pix[i+2]}
			if rgbDist(p, bg) > th {
				mask[y*w+x] = true
				fgCount++
			}
		}
	}
	if fgCount == 0 {
		return nil
	}

	// light close: dilate 1 + erode 1 heals single-pixel AA gaps in strokes while
	// keeping thin interiors (a heavier 2+2 erases them entirely).
	for pass := 0; pass < 1; pass++ {
		mask = dilate(mask, w, h)
	}
	for pass := 0; pass < 1; pass++ {
		mask = erode(mask, w, h)
	}

	// connected components: keep ALL fully-interior components (a rotated thin
	// sprite fragments into several; only border-touching parts are text that
	// spills, which we discard). Fall back to the largest interior component if
	// nothing is interior.
	comps := comps(mask, w, h)
	if len(comps) == 0 {
		return nil
	}
	var best []int
	for _, c := range comps {
		if touchesBorder(c, w, h) {
			continue
		}
		best = append(best, c...)
	}
	if best == nil {
		// everything touches the border: keep the largest blob overall
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