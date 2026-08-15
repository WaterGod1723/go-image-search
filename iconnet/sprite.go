package iconnet

import (
	"image"
	_ "image/jpeg"
	_ "image/png"
	"image/draw"
	"math"
	"os"
)

// Sprite preprocessing, ported 1:1 from pynet/preprocess.py so inference
// reproduces the Python evaluation numbers exactly.
//
// A query is a rendered icon on a colored canvas. The sprite is the largest
// foreground blob after removing the (border-estimated) background, with
// morphological closing to heal thin strokes and touching-fragment merge.
// Refs are transparent-background source icons (trim alpha, letterbox).

// Sprite is a normalized input: an N x N RGBA image (channels 0..255).
type Sprite struct {
	N    int
	RGBA []uint8 // row-major N*N pixels, 4 bytes each (R,G,B,A)
}

// Pixel access helpers ------------------------------------------------

func (s *Sprite) at(x, y int) (r, g, b, a uint8) {
	i := (y*s.N + x) * 4
	return s.RGBA[i], s.RGBA[i+1], s.RGBA[i+2], s.RGBA[i+3]
}

// Tensor returns the NCHW float tensor in [0,1] (channels R,G,B,A).
func (s *Sprite) Tensor() []float64 {
	n := s.N
	out := make([]float64, 4*n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			i := (y*n + x) * 4
			out[0*n*n+y*n+x] = float64(s.RGBA[i]) / 255.0
			out[1*n*n+y*n+x] = float64(s.RGBA[i+1]) / 255.0
			out[2*n*n+y*n+x] = float64(s.RGBA[i+2]) / 255.0
			out[3*n*n+y*n+x] = float64(s.RGBA[i+3]) / 255.0
		}
	}
	return out
}

// loadImage decodes a PNG/JPEG file into an NRGBA image, preserving
// non-premultiplied alpha (matches how Python/PIL reads pixels).
func loadImage(path string) (*image.NRGBA, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}
	b := img.Bounds()
	out := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(out, out.Bounds(), img, b.Min, draw.Src)
	return out, nil
}

// estimateBG returns the median RGB of the 8px border ring (matches
// pynet.preprocess.estimate_bg).
func estimateBG(img *image.NRGBA) (r, g, b float64) {
	w, h := img.Rect.Dx(), img.Rect.Dy()
	const bw = 8
	ring := make([]float64, 0, 4*bw*(w+h))
	collect := func(x, y int) {
		i := img.PixOffset(x, y)
		ring = append(ring, float64(img.Pix[i]), float64(img.Pix[i+1]), float64(img.Pix[i+2]))
	}
	for y := 0; y < bw && y < h; y++ {
		for x := 0; x < w; x++ {
			collect(x, y)
		}
	}
	for y := h - bw; y < h; y++ {
		if y < 0 {
			continue
		}
		for x := 0; x < w; x++ {
			collect(x, y)
		}
	}
	for y := 0; y < h; y++ {
		for x := 0; x < bw && x < w; x++ {
			collect(x, y)
		}
		for x := w - bw; x < w; x++ {
			if x < 0 {
				continue
			}
			collect(x, y)
		}
	}
	if len(ring) == 0 {
		return 0, 0, 0
	}
	// median per channel (ring has 3 components per pixel)
	npx := len(ring) / 3
	ch := make([]float64, npx)
	for k := 0; k < 3; k++ {
		for i := 0; i < npx; i++ {
			ch[i] = ring[i*3+k]
		}
		quickselect(ch, 0, npx-1, npx/2)
		m := ch[npx/2]
		switch k {
		case 0:
			r = m
		case 1:
			g = m
		case 2:
			b = m
		}
	}
	return r, g, b
}

func quickselect(a []float64, lo, hi, k int) {
	if lo >= hi {
		return
	}
	pivot := a[hi]
	i := lo
	for j := lo; j < hi; j++ {
		if a[j] < pivot {
			a[i], a[j] = a[j], a[i]
			i++
		}
	}
	a[i], a[hi] = a[hi], a[i]
	if k == i {
		return
	}
	if k < i {
		quickselect(a, lo, i-1, k)
	} else {
		quickselect(a, i+1, hi, k)
	}
}

// adaptiveDistThreshold matches pynet.preprocess.adaptive_dist_threshold:
// background peak then first significant valley.
func adaptiveDistThreshold(dist [256]int) float64 {
	bgPeak := 0
	for i := 1; i < 256; i++ {
		if dist[i] > dist[bgPeak] {
			bgPeak = i
		}
	}
	for i := bgPeak + 1; i < 255; i++ {
		if dist[i] < dist[i-1] && dist[i] < dist[i+1] && dist[i] < dist[bgPeak]/8 {
			return float64(i) / 256
		}
	}
	floor := 30.0 / 256.0
	low := float64(bgPeak+12) / 256.0
	if low > floor {
		floor = low
	}
	return floor
}

// extractSprite returns the sprite (RGBA on transparent bg) and ok flag.
// Port of pynet.preprocess.extract_sprite (Go-style Go segmentation).
// Returns rgb (sw*sh*3), mask (sw*sh bools), sw, sh, ok.
func extractSprite(img *image.NRGBA, closeK int) (rgb []uint8, mask []bool, sw, sh int, ok bool) {
	w, h := img.Rect.Dx(), img.Rect.Dy()
	if w == 0 || h == 0 {
		return nil, nil, 0, 0, false
	}
	br, bg, bb := estimateBG(img)
	// distance-to-bg histogram (256 bins)
	var dist [256]int
	d2 := make([]float64, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := img.PixOffset(x, y)
			dr := float64(img.Pix[i]) - br
			dg := float64(img.Pix[i+1]) - bg
			db := float64(img.Pix[i+2]) - bb
			d := math.Sqrt(dr*dr+dg*dg+db*db) / 255.0
			d2[y*w+x] = d
			bi := int(d * 256)
			if bi >= 256 {
				bi = 255
			}
			dist[bi]++
		}
	}
	thr := adaptiveDistThreshold(dist)
	fg := make([]bool, w*h)
	fgCount := 0
	for i := range fg {
		if d2[i] > thr {
			fg[i] = true
			fgCount++
		}
	}
	if fgCount < 20 {
		return nil, nil, 0, 0, false
	}
	// morphological closing (dilate then erode), closeK passes each
	if closeK >= 1 {
		for k := 0; k < closeK; k++ {
			fg = dilate(fg, w, h)
		}
		for k := 0; k < closeK; k++ {
			fg = erode(fg, w, h)
		}
	}
	// connected components
	labels := labelCC(fg, w, h)
	if labels == nil {
		return nil, nil, 0, 0, false
	}
	n := len(labels)
	sizes := make([]int, n)
	for i := range labels {
		sizes[i] = len(labels[i])
	}
	main := 0
	for i := 1; i < n; i++ {
		if sizes[i] > sizes[main] {
			main = i
		}
	}
	if sizes[main] < 20 {
		return nil, nil, 0, 0, false
	}
	keep := make([]bool, w*h)
	for _, p := range labels[main] {
		keep[p] = true
	}
	// add touching components (bbox-intersection proxy); keep GROWS so the
	// check is transitive (matches pynet _touches(keep, comp))
	added := true
	for added {
		added = false
		for i := 0; i < n; i++ {
			if i == main {
				continue
			}
			// skip if already in keep
			already := true
			for _, p := range labels[i] {
				if !keep[p] {
					already = false
					break
				}
			}
			if already {
				continue
			}
			if bboxTouch(keep, labels[i], w) {
				for _, p := range labels[i] {
					keep[p] = true
				}
				added = true
			}
		}
	}
	// crop bbox
	minX, minY, maxX, maxY := w, h, -1, -1
	for p := range keep {
		if !keep[p] {
			continue
		}
		x, y := p%w, p/w
		if x < minX {
			minX = x
		}
		if x > maxX {
			maxX = x
		}
		if y < minY {
			minY = y
		}
		if y > maxY {
			maxY = y
		}
	}
	if maxX < minX || maxY < minY {
		return nil, nil, 0, 0, false
	}
	sw, sh = maxX-minX+1, maxY-minY+1
	rgb = make([]uint8, sw*sh*3)
	mask = make([]bool, sw*sh)
	for y := minY; y <= maxY; y++ {
		for x := minX; x <= maxX; x++ {
			src := img.PixOffset(x, y)
			dst := (y-minY)*sw + (x-minX)
			// copy ALL RGB in the bbox (Python copies the whole crop; only
			// alpha is masked), so bg-colored pixels inside the bbox also
			// contribute to the later bilinear resize.
			rgb[dst*3] = img.Pix[src]
			rgb[dst*3+1] = img.Pix[src+1]
			rgb[dst*3+2] = img.Pix[src+2]
			if keep[y*w+x] {
				mask[dst] = true
			}
		}
	}
	return rgb, mask, sw, sh, true
}

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

// labelCC finds 4-connected components (matches scipy.ndimage.label default);
// returns per-component pixel indices.
func labelCC(mask []bool, w, h int) [][]int {
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
			// 4-connectivity: up/down/left/right
			for _, nb := range [4][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}} {
				nx, ny := x+nb[0], y+nb[1]
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
		out = append(out, cur)
	}
	return out
}

// bboxTouch reports whether the pixel set b overlaps the bbox of the keep
// mask (a cheap proxy for scipy adjacency, matching pynet _touches).
func bboxTouch(keep []bool, b []int, w int) bool {
	if len(b) == 0 {
		return false
	}
	minB, maxB := w, -1
	minBy, maxBy := int(^uint(0)>>1), -1
	for _, p := range b {
		x, y := p%w, p/w
		if x < minB {
			minB = x
		}
		if x > maxB {
			maxB = x
		}
		if y < minBy {
			minBy = y
		}
		if y > maxBy {
			maxBy = y
		}
	}
	// find keep bbox
	minA, maxA := w, -1
	minAy, maxAy := int(^uint(0)>>1), -1
	for p, v := range keep {
		if !v {
			continue
		}
		x, y := p%w, p/w
		if x < minA {
			minA = x
		}
		if x > maxA {
			maxA = x
		}
		if y < minAy {
			minAy = y
		}
		if y > maxAy {
			maxAy = y
		}
	}
	return !(maxA < minB || maxB < minA || maxAy < minBy || maxBy < minAy)
}

// trimTransparent trims rows/cols whose mask alpha <= 10 (matches
// pynet.preprocess.trim_transparent). alpha is 0..255 uint8 per pixel.
// Returns cropped rgb + alpha.
func trimTransparent(rgb []uint8, alpha []uint8, w, h int) ([]uint8, []uint8, int, int) {
	minX, minY, maxX, maxY := w, h, -1, -1
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if alpha[y*w+x] > 10 {
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	if maxX < minX || maxY < minY {
		return rgb, alpha, w, h
	}
	nw, nh := maxX-minX+1, maxY-minY+1
	nrgb := make([]uint8, nw*nh*3)
	nalpha := make([]uint8, nw*nh)
	for y := minY; y <= maxY; y++ {
		for x := minX; x <= maxX; x++ {
			s := y*w + x
			d := (y-minY)*nw + (x-minX)
			nrgb[d*3] = rgb[s*3]
			nrgb[d*3+1] = rgb[s*3+1]
			nrgb[d*3+2] = rgb[s*3+2]
			nalpha[d] = alpha[s]
		}
	}
	return nrgb, nalpha, nw, nh
}

// toSquare letterboxes (keep aspect) into side x side, resizing with
// bilinear interpolation, then alpha-masks the RGB (matches
// pynet.preprocess.to_square). alpha is 0..255 per pixel.
func toSquare(rgb []uint8, alpha []uint8, w, h, side int) *Sprite {
	s := w
	if h > s {
		s = h
	}
	if s == 0 {
		return &Sprite{N: side, RGBA: make([]uint8, side*side*4)}
	}
	padR := make([]uint8, s*s*3)
	padM := make([]uint8, s*s) // alpha 0..255 like Python
	py0 := (s - h) / 2
	px0 := (s - w) / 2
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			d := (py0+y)*s + (px0 + x)
			// copy ALL RGB (Python pads the whole rgb, alpha separately)
			padR[d*3] = rgb[(y*w+x)*3]
			padR[d*3+1] = rgb[(y*w+x)*3+1]
			padR[d*3+2] = rgb[(y*w+x)*3+2]
			padM[d] = alpha[y*w+x]
		}
	}
	// bilinear resize both to side x side
	rr := resizeBilinear(padR, s, s, side, 3)
	rm := resizeBilinear(padM, s, s, side, 1)
	out := &Sprite{N: side, RGBA: make([]uint8, side*side*4)}
	for y := 0; y < side; y++ {
		for x := 0; x < side; x++ {
			i := y*side + x
			a := rm[i]
			out.RGBA[i*4] = uint8(float64(rr[i*3]) * float64(a) / 255.0)
			out.RGBA[i*4+1] = uint8(float64(rr[i*3+1]) * float64(a) / 255.0)
			out.RGBA[i*4+2] = uint8(float64(rr[i*3+2]) * float64(a) / 255.0)
			out.RGBA[i*4+3] = a
		}
	}
	return out
}

// resizeBilinear resizes a uint8 image (w x h, row-major, nbands) to
// side x side using Pillow's exact BILINEAR resample: a two-pass separable
// convolution with a triangle kernel whose support is scaled by
// max(in/out, 1), fixed-point rounded with PRECISION_BITS=22. Output is
// byte-identical to PIL.Image.resize(size, Image.BILINEAR).
func resizeBilinear(src []uint8, sw, sh, side, nbands int) []uint8 {
	out := make([]uint8, side*side*nbands)
	hCoeffs, hXmin, hCnt := resampleCoeffs(sw, side)
	vCoeffs, vXmin, vCnt := resampleCoeffs(sh, side)
	ksizeH := coeffStride(sw, side)
	ksizeV := coeffStride(sh, side)
	// horizontal pass -> temp (side x sh, nbands)
	temp := make([]uint8, side*sh*nbands)
	for y := 0; y < sh; y++ {
		srcRow := y * sw * nbands
		for ox := 0; ox < side; ox++ {
			base := (y*side + ox) * nbands
			cnt := hCnt[ox]
			xm := hXmin[ox]
			k := hCoeffs[ox*ksizeH:]
			for c := 0; c < nbands; c++ {
				ss := int64(1 << (precBits - 1))
				for x := 0; x < cnt; x++ {
					ss += int64(src[srcRow+(xm+x)*nbands+c]) * k[x]
				}
				temp[base+c] = clip8(ss)
			}
		}
	}
	// vertical pass -> out (side x side, nbands)
	for oy := 0; oy < side; oy++ {
		cnt := vCnt[oy]
		ym := vXmin[oy]
		k := vCoeffs[oy*ksizeV:]
		for ox := 0; ox < side; ox++ {
			for c := 0; c < nbands; c++ {
				ss := int64(1 << (precBits - 1))
				for y := 0; y < cnt; y++ {
					ss += int64(temp[((ym+y)*side+ox)*nbands+c]) * k[y]
				}
				out[(oy*side+ox)*nbands+c] = clip8(ss)
			}
		}
	}
	return out
}

func coeffStride(inSize, outSize int) int {
	scale := float64(inSize) / float64(outSize)
	fs := scale
	if fs < 1.0 {
		fs = 1.0
	}
	return int(math.Ceil(fs))*2 + 1
}

// Pillow resample constants.
const precBits = 22

// resampleCoeffs precomputes per-output-pixel kernel support (xmin + count)
// and normalized fixed-point coefficients, mirroring Pillow precompute_coeffs
// with the BILINEAR filter (support 1.0, filterscale = max(scale, 1)).
func resampleCoeffs(inSize, outSize int) (k []int64, xmin []int, cnt []int) {
	scale := float64(inSize) / float64(outSize)
	filterscale := scale
	if filterscale < 1.0 {
		filterscale = 1.0
	}
	support := filterscale // bilinear filter support = 1.0 * filterscale
	ksize := int(math.Ceil(support))*2 + 1
	xmin = make([]int, outSize)
	cnt = make([]int, outSize)
	k = make([]int64, outSize*ksize)
	ss := 1.0 / filterscale
	for xx := 0; xx < outSize; xx++ {
		center := (float64(xx) + 0.5) * scale
		mn := int(center - support + 0.5)
		if mn < 0 {
			mn = 0
		}
		mx := int(center + support + 0.5)
		if mx > inSize {
			mx = inSize
		}
		c := mx - mn
		cnt[xx] = c
		xmin[xx] = mn
		row := k[xx*ksize : xx*ksize+c]
		ww := 0.0
		w := make([]float64, c)
		for x := 0; x < c; x++ {
			v := bilinearFilter((float64(x+mn) - center + 0.5) * ss)
			w[x] = v
			ww += v
		}
		if ww != 0.0 {
			for x := 0; x < c; x++ {
				row[x] = int64(0.5 + w[x]/ww*float64(1<<precBits))
			}
		}
	}
	return k, xmin, cnt
}

func bilinearFilter(x float64) float64 {
	if x < 0 {
		x = -x
	}
	if x < 1.0 {
		return 1.0 - x
	}
	return 0.0
}

// clip8 rounds the fixed-point sum (>>precBits) and clips to [0,255].
func clip8(ss int64) uint8 {
	v := int(ss >> precBits)
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// QuerySprite runs the full query pipeline: load, segment, trim, letterbox.
func QuerySprite(path string, side int) (*Sprite, bool) {
	img, err := loadImage(path)
	if err != nil {
		return nil, false
	}
	rgb, mask, sw, sh, ok := extractSprite(img, 3)
	if !ok {
		return nil, false
	}
	// extractSprite mask is bool; convert to 0/255 alpha (Python uses
	// np.where(keep, 255, 0))
	alpha := make([]uint8, len(mask))
	for i, m := range mask {
		if m {
			alpha[i] = 255
		}
	}
	nrgb, nalpha, nw, nh := trimTransparent(rgb, alpha, sw, sh)
	return toSquare(nrgb, nalpha, nw, nh, side), true
}

// RefSprite runs the reference pipeline: load RGBA, trim transparent, letterbox.
func RefSprite(path string, side int) (*Sprite, bool) {
	img, err := loadImage(path)
	if err != nil {
		return nil, false
	}
	w, h := img.Rect.Dx(), img.Rect.Dy()
	rgb := make([]uint8, w*h*3)
	alpha := make([]uint8, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := img.PixOffset(x, y)
			rgb[(y*w+x)*3] = img.Pix[i]
			rgb[(y*w+x)*3+1] = img.Pix[i+1]
			rgb[(y*w+x)*3+2] = img.Pix[i+2]
			alpha[y*w+x] = img.Pix[i+3]
		}
	}
	nrgb, nalpha, nw, nh := trimTransparent(rgb, alpha, w, h)
	return toSquare(nrgb, nalpha, nw, nh, side), true
}
