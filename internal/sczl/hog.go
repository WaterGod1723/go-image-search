package sczl

import "math"

// hog.go Histogram of Oriented Gradients descriptor.
//
// HOG captures the distribution of edge gradient orientations in spatial cells.
// Unlike LBP (pixel-level binary patterns, fragile to rendering differences and
// meaningless for flat-color icons), HOG operates on gradient magnitudes — which
// are naturally strong at icon boundaries and weak in uniform areas. This makes
// it invariant to absolute fill color and background luminance: a red filled
// train and a black outline train both produce boundary edges at the same
// locations with the same orientations.
//
// Architecture (Dalal & Triggs, simplified):
//   - 64x64 grayscale patch
//   - Sobel 3x3 gradients
//   - 8x8 pixel cells, 9 unsigned orientation bins [0, pi)
//   - 2x2 cell blocks with 50% overlap, L2-Hys normalization
//   - Output: 7x7 blocks x 36 dims = 1764-dim, L2-normalized

const (
	hogCellSize = 8 // pixels per cell
	hogNBins    = 9 // unsigned orientation bins [0, pi)
)

// hogDescriptor computes the HOG descriptor from a normSize x normSize grayscale
// patch. Only foreground-masked pixels contribute (background cells are zero —
// uniform backgrounds produce no gradients).
func hogDescriptor(gray []float64, mask []bool) []float64 {
	n := normSize
	if len(gray) != n*n || len(mask) != n*n {
		return nil
	}

	// Sobel gradients.
	gx := make([]float64, n*n)
	gy := make([]float64, n*n)
	for y := 1; y < n-1; y++ {
		for x := 1; x < n-1; x++ {
			i := y*n + x
			if !mask[i] {
				continue
			}
			ul := (y-1)*n + (x - 1)
			uc := (y - 1) * n + x
			ur := (y-1)*n + (x + 1)
			ml := y*n + (x - 1)
			mr := y*n + (x + 1)
			dl := (y+1)*n + (x - 1)
			dc := (y + 1) * n + x
			dr := (y+1)*n + (x + 1)
			gx[i] = -gray[ul] + gray[ur] - 2*gray[ml] + 2*gray[mr] - gray[dl] + gray[dr]
			gy[i] = -gray[ul] - 2*gray[uc] - gray[ur] + gray[dl] + 2*gray[dc] + gray[dr]
		}
	}

	// Gradient magnitude and unsigned orientation [0, pi).
	mag := make([]float64, n*n)
	ang := make([]float64, n*n)
	for i := range gx {
		if gx[i] == 0 && gy[i] == 0 {
			continue
		}
		mag[i] = math.Sqrt(gx[i]*gx[i] + gy[i]*gy[i])
		a := math.Atan2(gy[i], gx[i])
		if a < 0 {
			a += math.Pi
		}
		ang[i] = a
	}

	// Cell histograms: 8x8 cells, each 8x8 pixels, 9 bins.
	nCells := n / hogCellSize // 8
	cells := make([][]float64, nCells*nCells)
	for cy := 0; cy < nCells; cy++ {
		for cx := 0; cx < nCells; cx++ {
			hist := make([]float64, hogNBins)
			for py := 0; py < hogCellSize; py++ {
				for px := 0; px < hogCellSize; px++ {
					y := cy*hogCellSize + py
					x := cx*hogCellSize + px
					idx := y*n + x
					if mag[idx] <= 0 {
						continue
					}
					// Linear interpolation between adjacent bins for smoother hist.
					binF := ang[idx] / math.Pi * float64(hogNBins)
					if binF >= float64(hogNBins) {
						binF = float64(hogNBins) - 1e-6
					}
					bLo := int(math.Floor(binF))
					if bLo < 0 {
						bLo = 0
					}
					if bLo >= hogNBins {
						bLo = hogNBins - 1
					}
					frac := binF - float64(bLo)
					bHi := bLo + 1
					if bHi >= hogNBins {
						bHi = 0
					}
					hist[bLo] += mag[idx] * (1 - frac)
					hist[bHi] += mag[idx] * frac
				}
			}
			cells[cy*nCells+cx] = hist
		}
	}

	// Block normalization: 2x2 cell blocks, stride 1 (50% overlap).
	blockSize := 2
	nBlocks := nCells - blockSize + 1 // 7
	blockDim := blockSize * blockSize * hogNBins // 36
	desc := make([]float64, 0, nBlocks*nBlocks*blockDim)
	for by := 0; by < nBlocks; by++ {
		for bx := 0; bx < nBlocks; bx++ {
			block := make([]float64, 0, blockDim)
			for cy := 0; cy < blockSize; cy++ {
				for cx := 0; cx < blockSize; cx++ {
					block = append(block, cells[(by+cy)*nCells+(bx+cx)]...)
				}
			}
			// L2-Hys: clip to 0.2, renormalize.
			norm := 0.0
			for _, v := range block {
				norm += v * v
			}
			if norm > 0 {
				norm = math.Sqrt(norm)
				for i := range block {
					block[i] /= norm
					if block[i] > 0.2 {
						block[i] = 0.2
					}
				}
				norm = 0.0
				for _, v := range block {
					norm += v * v
				}
				if norm > 0 {
					norm = math.Sqrt(norm)
					for i := range block {
						block[i] /= norm
					}
				}
			}
			desc = append(desc, block...)
		}
	}

	return l2normalize(desc)
}
