package main

import (
	"math"
	"os"
	"strconv"
)

// scMaxPts caps the number of shape-context sample points per side. The
// Hungarian match is O(n^3), so cutting 120->48 is ~15x cheaper with no
// measured recall loss (sczl also uses a 48-point cap). Override with SCPTS.
var scMaxPts = 48

func init() {
	if v := os.Getenv("SCPTS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			scMaxPts = n
		}
	}
}

// refSC lazily computes and caches a reference's sample points and
// shape-context histograms. The reference side is query-independent, so it is
// computed once per Feat instead of once per (query, ref) pair.
func (f *Feat) refSC() ([]scPoint, []float64) {
	f.scOnce.Do(func() {
		f.scPts = sampleSCPoints(f.Mask48)
		f.scHist, _ = buildSC(f.scPts)
	})
	return f.scPts, f.scHist
}

// scQueryHists precomputes the shape-context histograms of a query rotated
// over the whole sweep, once per query, so every shortlist candidate shares
// them instead of each rebuilding the rotated query clouds.
func scQueryHists(q *Feat) [][]float64 {
	qp := sampleSCPoints(q.Mask48)
	hists := make([][]float64, scRot)
	for step := 0; step < scRot; step++ {
		ang := float64(step) * scRotDeg * math.Pi / 180
		rqp := rotateSCPoints(qp, ang)
		qh, _ := buildSC(rqp)
		hists[step] = qh
	}
	return hists
}

// scSimFromHists is the shape-context similarity between precomputed query
// histograms and a reference, over the best (cheapest) rotation.
func scSimFromHists(qHists [][]float64, r *Feat) float64 {
	if len(qHists) == 0 {
		return 0
	}
	_, rh := r.refSC()
	if len(rh) == 0 {
		return 0
	}
	best := math.Inf(1)
	for _, qh := range qHists {
		cost := matchCost(qh, rh)
		if cost < best {
			best = cost
		}
	}
	sim := 1.0 - best
	if sim < 0 {
		sim = 0
	}
	if sim > 1 {
		sim = 1
	}
	return sim
}

// Shape-context point matching (Belongie et al.) on the 64x64 soft masks.
//
// Points are sampled as occupied-block centroids (up to 8x8 = 64 points), each
// described by a log-polar histogram of the other points' relative positions.
// Two shapes are matched by a Hungarian assignment minimizing chi-squared cost.
// Rotation is handled by sweeping the query point cloud over coarse angles and
// taking the best (cheapest) match — appropriate here because the query sprite
// is an arbitrarily-rotated render of the reference.

type scPoint struct {
	x, y float64
}

const (
	scR      = 6  // radial bins
	scA      = 12 // angular bins
	scBlock  = 8  // sampling block grid over maskN
	scRot    = 36 // rotation steps (0..350 in 10deg)
	scRotDeg = 10
)

// sampleSCPoints returns point samples from a soft mask. To keep the point
// cloud dense enough for thin-stroke glyphs we sample every occupied cell,
// then decimate to at most scMaxPts evenly (by grid order).
func sampleSCPoints(m []float64) []scPoint {
	var pts []scPoint
	for y := 0; y < maskN; y++ {
		for x := 0; x < maskN; x++ {
			v := m[y*maskN+x]
			if v > 0.05 {
				pts = append(pts, scPoint{x: float64(x), y: float64(y)})
			}
		}
	}
	if len(pts) > scMaxPts {
		step := float64(len(pts)) / float64(scMaxPts)
		out := make([]scPoint, 0, scMaxPts)
		for i := 0; i < scMaxPts; i++ {
			out = append(out, pts[int(float64(i)*step)])
		}
		return out
	}
	return pts
}

// buildSC computes shape-context histograms (log-polar) for each point.
// Each row is scR*scA bins, L1-normalized. rmax = median pairwise distance.
func buildSC(pts []scPoint) ([]float64, float64) {
	n := len(pts)
	if n == 0 {
		return nil, 0
	}
	// median pairwise distance for scale normalization
	ds := make([]float64, 0, n*n)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			d := math.Hypot(pts[i].x-pts[j].x, pts[i].y-pts[j].y)
			ds = append(ds, d)
		}
	}
	if len(ds) == 0 {
		return nil, 0
	}
	// quickselect-style median
	med := quickSel(ds)
	if med <= 1e-6 {
		med = 1
	}
	rmax := 2.5 * med
	rmin := med * 0.05

	hist := make([]float64, n*scR*scA)
	// log-polar bins: r in [rmin, rmax] log-spaced, angle in 0..360
	for i := 0; i < n; i++ {
		row := hist[i*scR*scA:]
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			d := math.Hypot(pts[i].x-pts[j].x, pts[i].y-pts[j].y)
			if d < rmin || d > rmax {
				continue
			}
			t := math.Atan2(pts[j].y-pts[i].y, pts[j].x-pts[i].x)
			if t < 0 {
				t += 2 * math.Pi
			}
			ai := int(t / (2 * math.Pi) * scA)
			if ai >= scA {
				ai = scA - 1
			}
			lr := math.Log(d / rmin)
			lrMax := math.Log(rmax / rmin)
			ri := int(lr / lrMax * scR)
			if ri >= scR {
				ri = scR - 1
			}
			row[ri*scA+ai]++
		}
		// L1 normalize
		var s float64
		for k := 0; k < scR*scA; k++ {
			s += row[k]
		}
		if s > 0 {
			for k := 0; k < scR*scA; k++ {
				row[k] /= s
			}
		}
	}
	return hist, med
}

func quickSel(a []float64) float64 {
	k := len(a) / 2
	lo, hi := 0, len(a)-1
	for lo < hi {
		p := a[hi]
		i := lo
		for j := lo; j < hi; j++ {
			if a[j] < p {
				a[i], a[j] = a[j], a[i]
				i++
			}
		}
		a[i], a[hi] = a[hi], a[i]
		if k < i {
			hi = i - 1
		} else if k > i {
			lo = i + 1
		} else {
			break
		}
	}
	return a[k]
}

// chiSq distance between two histogram rows.
func chiSq(a, b []float64) float64 {
	var d float64
	for k := range a {
		if a[k]+b[k] > 0 {
			d += (a[k] - b[k]) * (a[k] - b[k]) / (a[k] + b[k])
		}
	}
	return d / 2
}

// hungarianMin solves the min-cost rectangular assignment; returns total cost.
// The classic O(n^3) algorithm needs nRows <= nCols; transpose if not.
func hungarianMin(cost [][]float64) float64 {
	n := len(cost)
	m := len(cost[0])
	if n == 0 || m == 0 {
		return math.Inf(1)
	}
	if n > m {
		// transpose so rows <= cols
		ct := make([][]float64, m)
		for j := 0; j < m; j++ {
			ct[j] = make([]float64, n)
			for i := 0; i < n; i++ {
				ct[j][i] = cost[i][j]
			}
		}
		return hungarianMin(ct)
	}

	u := make([]float64, n+1)
	v := make([]float64, m+1)
	p := make([]int, m+1)
	way := make([]int, m+1)
	inf := math.Inf(1)
	for i := 1; i <= n; i++ {
		p[0] = i
		j0 := 0
		minv := make([]float64, m+1)
		used := make([]bool, m+1)
		for j := 0; j <= m; j++ {
			minv[j] = inf
		}
		for {
			used[j0] = true
			i0 := p[j0]
			delta := inf
			j1 := 0
			for j := 1; j <= m; j++ {
				if used[j] {
					continue
				}
				cur := cost[i0-1][j-1] - u[i0] - v[j]
				if cur < minv[j] {
					minv[j] = cur
					way[j] = j0
				}
				if minv[j] < delta {
					delta = minv[j]
					j1 = j
				}
			}
			for j := 0; j <= m; j++ {
				if used[j] {
					u[p[j]] += delta
					v[j] -= delta
				} else {
					minv[j] -= delta
				}
			}
			j0 = j1
			if p[j0] == 0 {
				break
			}
		}
		for {
			j1 := way[j0]
			p[j0] = p[j1]
			j0 = j1
			if j0 == 0 {
				break
			}
		}
	}
	var total float64
	for j := 1; j <= m; j++ {
		i := p[j]
		if i > 0 && i <= n {
			total += cost[i-1][j-1]
		}
	}
	return total
}

// shapeContextSim compares two masks via shape-context matching; rotation is
// handled by a coarse sweep of the query points. Returns a similarity in
// [0,1] (1 = identical).
func shapeContextSim(q, r *Feat) float64 {
	return scSimFromHists(scQueryHists(q), r)
}

func rotateSCPoints(pts []scPoint, ang float64) []scPoint {
	cs, sn := math.Cos(ang), math.Sin(ang)
	out := make([]scPoint, len(pts))
	for i, p := range pts {
		out[i] = scPoint{x: p.x*cs - p.y*sn, y: p.x*sn + p.y*cs}
	}
	return out
}

// matchCost returns the Hungarian cost between two shape-context hist sets,
// padded so both sets have equal size.
func matchCost(qh, rh []float64) float64 {
	n := len(qh) / (scR * scA)
	m := len(rh) / (scR * scA)
	if n == 0 || m == 0 {
		return math.Inf(1)
	}
	size := n
	if m > size {
		size = m
	}
	cost := make([][]float64, n)
	for i := 0; i < n; i++ {
		cost[i] = make([]float64, m)
		qi := qh[i*scR*scA : (i+1)*scR*scA]
		for j := 0; j < m; j++ {
			rj := rh[j*scR*scA : (j+1)*scR*scA]
			cost[i][j] = chiSq(qi, rj)
		}
	}
	return hungarianMin(cost) / float64(size)
}
