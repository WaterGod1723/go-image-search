package attnnet

import (
	"math"
	"math/rand"
	"testing"
)

func smallCfg(mode Mode) Config {
	return Config{
		ImageSize:  12,
		PatchSize:  4, // 3x3 = 9 tokens
		D:          8,
		Heads:      2,
		Depth:      1,
		MLPHidden:  16,
		Mode:       mode,
		GateLambda: 1.0,
	}
}

func randomSample(cfg Config, rng *rand.Rand) (x []uint8, mask []uint8) {
	nt := cfg.nTok()
	pd := cfg.patchDim()
	x = make([]uint8, nt*pd)
	for i := range x {
		x[i] = uint8(rng.Intn(256))
	}
	mask = make([]uint8, nt)
	for i := range mask {
		if rng.Float64() < 0.4 {
			mask[i] = 1
		}
	}
	return x, mask
}

// TestBackwardGradientCheck compares analytic gradients with centered finite
// differences of the combined loss (CE + gate BCE), for both attention modes.
func TestBackwardGradientCheck(t *testing.T) {
	for _, mode := range []Mode{ModeGlobal, ModeIcon} {
		cfg := smallCfg(mode)
		m := New(cfg, 42)
		rng := rand.New(rand.NewSource(7))
		x, mask := randomSample(cfg, rng)

		a, ce, gl := m.forward(x, mask)
		g := newGrads(m)
		ace, agl := m.backward(a, x, mask, g)
		if math.Abs(ace-ce) > 1e-9 || math.Abs(agl-gl) > 1e-9 {
			t.Fatalf("mode %d: loss mismatch fwd vs bwd (%.6f vs %.6f / %.6f vs %.6f)", mode, ce, ace, gl, agl)
		}

		total := func(mm *Model) float64 {
			_, c, glv := mm.forward(x, mask)
			return c + glv
		}
		eps := 1e-6
		params := m.params()
		grads := g.grads()
		for pi := range params {
			w := *params[pi]
			gw := *grads[pi]
			for i := 0; i < len(w); i += 13 {
				orig := w[i]
				w[i] = orig + eps
				fp := total(m)
				w[i] = orig - eps
				fm := total(m)
				w[i] = orig
				num := (fp - fm) / (2 * eps)
				ana := gw[i]
				if math.Abs(num-ana) > 1e-4 {
					t.Fatalf("mode %d param %d elem %d: analytic %.6f numeric %.6f", mode, pi, i, ana, num)
				}
			}
		}
	}
}

// TestTrainStepSmoke checks the trainer plumbing runs and gradients are finite.
func TestTrainStepSmoke(t *testing.T) {
	cfg := smallCfg(ModeIcon)
	tr := NewTrainer(New(cfg, 1))
	rng := rand.New(rand.NewSource(2))
	x, mask := randomSample(cfg, rng)
	ce, gl := tr.Step(x, mask)
	if math.IsNaN(ce) || math.IsNaN(gl) || math.IsInf(ce, 0) || math.IsInf(gl, 0) {
		t.Fatalf("non-finite loss: ce=%f gl=%f", ce, gl)
	}
	tr.Apply(1, 0.01)
	tr.Reset()
}

// TestSaveLoadRoundTrip checks gob save/load preserves predictions.
func TestSaveLoadRoundTrip(t *testing.T) {
	cfg := smallCfg(ModeIcon)
	m := New(cfg, 5)
	rng := rand.New(rand.NewSource(3))
	x, _ := randomSample(cfg, rng)
	before := m.Predict(x)
	path := t.TempDir() + "/m.gob"
	if err := m.Save(path); err != nil {
		t.Fatal(err)
	}
	m2, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	after := m2.Predict(x)
	for i := range before {
		if math.Abs(before[i]-after[i]) > 1e-12 {
			t.Fatalf("prediction mismatch at %d: %f vs %f", i, before[i], after[i])
		}
	}
}
