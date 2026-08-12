package main

import (
	"fmt"
	"image"
	"math"
	"os"

	"attnnet"
)

// Attention-based query localization.
//
// The trained attnnet model (see attnnet/, a standalone module) is a small
// attention-only transformer that maps a 48x48 RGB canvas to an 8x8 attention
// map over the icon region. When the ATTN env var points at its weights, this
// localization is used in two places:
//
//  1. query extraction refinement: the extracted sprite pixels are re-weighted
//     by how strongly the network believes they belong to the icon region, so
//     background / text leakage the heuristics kept is suppressed;
//  2. an attention-region query expert: the pixels inside the attention box are
//     extracted and their image descriptor (buildFeat) is built; the per-ref
//     similarities of that descriptor are appended to the pair block, giving
//     the fusion network a second query whose features only describe the icon
//     region (instead of passing the raw attention values).
//
// The input grid and patch size must match attnnet.Default(): 48x48, 6x6 patch,
// 8x8 = 64 tokens.

const (
	attnSize  = 48
	attnPatch = 6
	attnGrid  = attnSize / attnPatch // 8
	// attnNStat features appended to the pair block when attention is enabled.
	attnNStat = 8
)

// attnWeighter wraps the loaded attention model. It is stateless between
// queries: attnFor returns a fresh per-query attention map.
type attnWeighter struct {
	m *attnnet.Model
}

// attnExtractOn reports whether the attention map should also refine the query
// sprite extraction. ATTN_NOEXTRACT=1 keeps the heuristic extraction unchanged
// and only feeds the attention-region features to the network (ablation).
func attnExtractOn() bool {
	return os.Getenv("ATTN_NOEXTRACT") != "1"
}

// loadAttnModel loads the attention weights selected by the ATTN env var.
// Returns (nil, nil) when ATTN is unset (attention disabled).
func loadAttnModel() (*attnWeighter, error) {
	p := os.Getenv("ATTN")
	if p == "" {
		return nil, nil
	}
	m, err := attnnet.Load(p)
	if err != nil {
		return nil, fmt.Errorf("attnnet.Load(%s): %w", p, err)
	}
	return &attnWeighter{m: m}, nil
}

// attnFor runs the attention model on the query canvas and returns the 64
// attention values (softmax over the 8x8 grid).
func (aw *attnWeighter) attnFor(img *image.NRGBA) []float64 {
	return aw.m.Predict(attnInput(img))
}

// attnBox returns the canvas-space bounding box of the attention region: the
// 8x8 cells with attention >= 0.5*max, upscaled to the canvas. The fallback
// (degenerate / flat attention) is the whole canvas.
func attnBox(attn []float64, w, h int) (x0, y0, x1, y1 int) {
	var mx float64
	for _, v := range attn {
		if v > mx {
			mx = v
		}
	}
	th := 0.5 * mx
	x0, y0, x1, y1 = w, h, -1, -1
	cellW := float64(attnPatch) / attnSize * float64(w)
	cellH := float64(attnPatch) / attnSize * float64(h)
	for py := 0; py < attnGrid; py++ {
		for px := 0; px < attnGrid; px++ {
			if attn[py*attnGrid+px] < th {
				continue
			}
			cx0 := int(float64(px) * cellW)
			cx1 := int(float64(px+1)*cellW) - 1
			cy0 := int(float64(py) * cellH)
			cy1 := int(float64(py+1)*cellH) - 1
			if cx0 < x0 {
				x0 = cx0
			}
			if cx1 > x1 {
				x1 = cx1
			}
			if cy0 < y0 {
				y0 = cy0
			}
			if cy1 > y1 {
				y1 = cy1
			}
		}
	}
	if x1 < x0 || y1 < y0 {
		return 0, 0, w - 1, h - 1
	}
	return x0, y0, x1, y1
}

// attnQueryPixels extracts the sprite pixels inside the attention box, reusing
// the fg/bg distance heuristic but only within the box so text and stray
// background outside the icon region cannot pollute the descriptor. Returns nil
// when the attention is too flat / the box too large to be a confident icon
// localization (the expert is then skipped).
func attnQueryPixels(img *image.NRGBA, attn []float64) []Px {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	var mx, sum float64
	for _, v := range attn {
		sum += v
		if v > mx {
			mx = v
		}
	}
	if sum <= 0 || mx/sum < 0.045 { // ~3x uniform: map not peaked
		return nil
	}
	x0, y0, x1, y1 := attnBox(attn, w, h)
	if x1-x0 <= 4 || y1-y0 <= 4 {
		return nil // box too small: degenerate attention, skip the expert
	}
	boxFrac := float64((x1 - x0 + 1) * (y1 - y0 + 1)) / float64(w*h)
	if boxFrac > 0.6 {
		return nil // box too large: not a confident icon region
	}
	bg := estimateBG(img)
	var px []Px
	for y := y0; y <= y1; y++ {
		for x := x0; x <= x1; x++ {
			i := img.PixOffset(x, y)
			p := Px{X: x, Y: y, R: img.Pix[i], G: img.Pix[i+1], B: img.Pix[i+2]}
			if rgbDist(p, bg) <= 0.16 {
				continue
			}
			d := rgbDist(p, bg) / 0.2
			if d > 1 {
				d = 1
			}
			p.A = d
			px = append(px, p)
		}
	}
	return px
}

// attnInput downsamples a query canvas to the 48x48 RGB block the model reads.
func attnInput(img *image.NRGBA) []byte {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	out := make([]byte, attnSize*attnSize*3)
	for y := 0; y < attnSize; y++ {
		fy := (float64(y) + 0.5) * float64(h) / attnSize
		for x := 0; x < attnSize; x++ {
			fx := (float64(x) + 0.5) * float64(w) / attnSize
			r, g, b := sampleNRGBA(img, fx-0.5, fy-0.5)
			o := (y*attnSize + x) * 3
			out[o] = r
			out[o+1] = g
			out[o+2] = b
		}
	}
	return out
}

// sampleNRGBA bilinear-samples the straight-RGB color at fractional (x, y).
func sampleNRGBA(img *image.NRGBA, x, y float64) (r, g, b uint8) {
	bounds := img.Bounds()
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	if x0 < bounds.Min.X {
		x0 = bounds.Min.X
	}
	if y0 < bounds.Min.Y {
		y0 = bounds.Min.Y
	}
	if x0 > bounds.Max.X-1 {
		x0 = bounds.Max.X - 1
	}
	if y0 > bounds.Max.Y-1 {
		y0 = bounds.Max.Y - 1
	}
	x1, y1 := x0+1, y0+1
	if x1 > bounds.Max.X-1 {
		x1 = bounds.Max.X - 1
	}
	if y1 > bounds.Max.Y-1 {
		y1 = bounds.Max.Y - 1
	}
	fx, fy := x-float64(x0), y-float64(y0)
	for c := 0; c < 3; c++ {
		o00 := img.PixOffset(x0, y0) + c
		o10 := img.PixOffset(x1, y0) + c
		o01 := img.PixOffset(x0, y1) + c
		o11 := img.PixOffset(x1, y1) + c
		v00 := float64(img.Pix[o00])
		v10 := float64(img.Pix[o10])
		v01 := float64(img.Pix[o01])
		v11 := float64(img.Pix[o11])
		top := v00*(1-fx) + v10*fx
		bot := v01*(1-fx) + v11*fx
		val := top*(1-fy) + bot*fy
		switch c {
		case 0:
			r = uint8(val)
		case 1:
			g = uint8(val)
		case 2:
			b = uint8(val)
		}
	}
	return r, g, b
}

// attnWeightAt maps a canvas pixel to its 8x8 attention cell and returns the
// attention value, floored so a slightly-off localization never zeroes an
// entire sprite.
func attnWeightAt(attn []float64, x, y, w, h int) float64 {
	if len(attn) != attnGrid*attnGrid {
		return 1
	}
	fx := float64(x) * attnSize / float64(w)
	fy := float64(y) * attnSize / float64(h)
	px := int(fx / attnPatch)
	py := int(fy / attnPatch)
	if px < 0 {
		px = 0
	}
	if px >= attnGrid {
		px = attnGrid - 1
	}
	if py < 0 {
		py = 0
	}
	if py >= attnGrid {
		py = attnGrid - 1
	}
	v := attn[py*attnGrid+px]
	if v < 0.05 {
		v = 0.05
	}
	return v
}
