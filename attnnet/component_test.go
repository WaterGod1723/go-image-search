package attnnet

import (
	"math"
	"math/rand"
	"testing"
)

func relErr(a, b float64) float64 {
	if math.Abs(a) < 1e-9 && math.Abs(b) < 1e-9 {
		return 0
	}
	return math.Abs(a-b) / math.Max(1, math.Abs(a)+math.Abs(b))
}

func checkParamGrad(t *testing.T, name string, w *[]float64, gw *[]float64, total func() float64, step int) {
	t.Helper()
	eps := 1e-6
	for i := 0; i < len(*w); i += step {
		orig := (*w)[i]
		(*w)[i] = orig + eps
		fp := total()
		(*w)[i] = orig - eps
		fm := total()
		(*w)[i] = orig
		num := (fp - fm) / (2 * eps)
		ana := (*gw)[i]
		if relErr(num, ana) > 1e-3 {
			t.Fatalf("%s elem %d: analytic %.8f numeric %.8f", name, i, ana, num)
		}
	}
}

// TestComponentLinear checks linearBackward against finite differences.
func TestComponentLinear(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	in, out, n := 4, 3, 5
	w := randN(rng, out*in, 0.5)
	b := randN(rng, out, 0.5)
	x := make([][]float64, n)
	for i := range x {
		x[i] = randN(rng, in, 1)
	}
	fwd := func() [][]float64 {
		return matVecRows(w, b, x, in)
	}
	dy := make([][]float64, n)
	for i := range dy {
		dy[i] = randN(rng, out, 1)
	}
	dx, dW, dB := linearBackward(w, b, x, dy, in)
	total := func() float64 {
		y := fwd()
		var s float64
		for i := 0; i < n; i++ {
			for o := 0; o < out; o++ {
				s += y[i][o] * dy[i][o]
			}
		}
		return s
	}
	checkParamGrad(t, "W", &w, &dW, total, 1)
	checkParamGrad(t, "b", &b, &dB, total, 1)
	// check dx: perturb x[i][j], compare to dx[i][j]
	eps := 1e-6
	for i := 0; i < n; i++ {
		for j := 0; j < in; j++ {
			orig := x[i][j]
			x[i][j] = orig + eps
			fp := total()
			x[i][j] = orig - eps
			fm := total()
			x[i][j] = orig
			num := (fp - fm) / (2 * eps)
			if relErr(num, dx[i][j]) > 1e-3 {
				t.Fatalf("x[%d][%d]: analytic %.8f numeric %.8f", i, j, dx[i][j], num)
			}
		}
	}
}

// TestComponentLayerNorm checks layernorm and its backward.
func TestComponentLayerNorm(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	D, n := 5, 4
	gamma := randN(rng, D, 1)
	beta := randN(rng, D, 1)
	rows := make([][]float64, n)
	dy := make([][]float64, n)
	for i := range rows {
		rows[i] = randN(rng, D, 1)
		dy[i] = randN(rng, D, 1)
	}
	fwdRow := func(v []float64) []float64 { return layernorm(v, gamma, beta) }
	dG := make([]float64, D)
	dB := make([]float64, D)
	dx := make([][]float64, n)
	for i := 0; i < n; i++ {
		dx[i] = layernormBackward(rows[i], dy[i], gamma, beta, dG, dB)
	}
	total := func() float64 {
		var s float64
		for i := 0; i < n; i++ {
			y := fwdRow(rows[i])
			for d := 0; d < D; d++ {
				s += y[d] * dy[i][d]
			}
		}
		return s
	}
	checkParamGrad(t, "gamma", &gamma, &dG, total, 1)
	checkParamGrad(t, "beta", &beta, &dB, total, 1)
	eps := 1e-6
	for i := 0; i < n; i++ {
		for j := 0; j < D; j++ {
			orig := rows[i][j]
			rows[i][j] = orig + eps
			fp := total()
			rows[i][j] = orig - eps
			fm := total()
			rows[i][j] = orig
			num := (fp - fm) / (2 * eps)
			if relErr(num, dx[i][j]) > 1e-3 {
				t.Fatalf("x[%d][%d]: analytic %.8f numeric %.8f", i, j, dx[i][j], num)
			}
		}
	}
}

// TestComponentAttention checks attentionBackward.
func TestComponentAttention(t *testing.T) {
	for _, mode := range []Mode{ModeIcon, ModeGlobal} {
		rng := rand.New(rand.NewSource(3))
		cfg := smallCfg(mode)
		n, D := cfg.nTok(), cfg.D
		q := randMat(rng, n, D)
		k := randMat(rng, n, D)
		v := randMat(rng, n, D)
		g := randN(rng, n, 1)
		if mode == ModeGlobal {
			g = make([]float64, n) // no gate in global mode
		}
		dOut := randMat(rng, n, D)

		out, attn := selfAttn(q, k, v, g, cfg, mode)
		total := func(qq, kk, vv [][]float64, gg []float64) float64 {
			o, _ := selfAttn(qq, kk, vv, gg, cfg, mode)
			var s float64
			for i := 0; i < n; i++ {
				for d := 0; d < D; d++ {
					s += o[i][d] * dOut[i][d]
				}
			}
			return s
		}
		dq, dk, dv, dg := attentionBackward(dOut, q, k, v, g, attn, cfg)
		_ = out
		eps := 1e-6
		check := func(name string, get func(i, d int) float64, numOf func(i, d int) float64) {
			for i := 0; i < n; i++ {
				for d := 0; d < D; d++ {
					num := numOf(i, d)
					ana := get(i, d)
					if relErr(num, ana) > 1e-3 {
						t.Fatalf("%s %v [%d][%d]: analytic %.8f numeric %.8f", name, mode, i, d, ana, num)
					}
				}
			}
		}
		check("dq", func(i, d int) float64 { return dq[i][d] }, func(i, d int) float64 {
			orig := q[i][d]
			q[i][d] = orig + eps
			fp := total(q, k, v, g)
			q[i][d] = orig - eps
			fm := total(q, k, v, g)
			q[i][d] = orig
			return (fp - fm) / (2 * eps)
		})
		check("dk", func(i, d int) float64 { return dk[i][d] }, func(i, d int) float64 {
			orig := k[i][d]
			k[i][d] = orig + eps
			fp := total(q, k, v, g)
			k[i][d] = orig - eps
			fm := total(q, k, v, g)
			k[i][d] = orig
			return (fp - fm) / (2 * eps)
		})
		check("dv", func(i, d int) float64 { return dv[i][d] }, func(i, d int) float64 {
			orig := v[i][d]
			v[i][d] = orig + eps
			fp := total(q, k, v, g)
			v[i][d] = orig - eps
			fm := total(q, k, v, g)
			v[i][d] = orig
			return (fp - fm) / (2 * eps)
		})
		if mode == ModeIcon {
			for i := 0; i < n; i++ {
				orig := g[i]
				g[i] = orig + eps
				fp := total(q, k, v, g)
				g[i] = orig - eps
				fm := total(q, k, v, g)
				g[i] = orig
				num := (fp - fm) / (2 * eps)
				if relErr(num, dg[i]) > 1e-3 {
					t.Fatalf("g[%d]: analytic %.8f numeric %.8f", i, dg[i], num)
				}
			}
		}
	}
}

func randMat(rng *rand.Rand, n, d int) [][]float64 {
	out := make([][]float64, n)
	for i := range out {
		out[i] = randN(rng, d, 1)
	}
	return out
}
