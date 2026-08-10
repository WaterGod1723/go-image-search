// Command train trains the small attention network on the generated test set,
// for both attention modes:
//
//	--mode global : vanilla self-attention over the whole canvas
//	--mode icon   : attention gated so it can only look at the icon region
//
// Metrics (on the validation split): attention mass inside the icon box
// (-> 1 means the network only attends to the icon) and top-k IoU of the
// predicted attention map vs the ground-truth mask.
//
// Usage: go run ./cmd/train [-set data/testset.bin] [-epochs 60] [-lr 0.003]
//        [-batch 32] [-seed 0]
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"attnnet"
)

var (
	setPath = flag.String("set", "data/testset.bin", "test set binary")
	epochs  = flag.Int("epochs", 60, "training epochs")
	lr      = flag.Float64("lr", 0.003, "learning rate")
	batch   = flag.Int("batch", 32, "mini-batch size")
	seed    = flag.Int64("seed", 0, "random seed (0 = time based)")
	outDir  = flag.String("out", "data", "weights output directory")
	modeSel = flag.String("mode", "both", "attention mode: global, icon or both")
)

func main() {
	flag.Parse()
	cfg, set, err := attnnet.LoadSet(*setPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load set:", err)
		os.Exit(1)
	}
	if *seed == 0 {
		*seed = time.Now().UnixNano()
	}
	fmt.Printf("set: %d samples, %d tokens/sample (grid %dx%d)\n", set.N, cfg.NTok(),
		cfg.ImageSize/cfg.PatchSize, cfg.ImageSize/cfg.PatchSize)

	train, val := attnnet.Split(set, 0.85, *seed)
	fmt.Printf("train %d  val %d\n", len(train), len(val))
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	var modes []attnnet.Mode
	switch *modeSel {
	case "global":
		modes = []attnnet.Mode{attnnet.ModeGlobal}
	case "icon":
		modes = []attnnet.Mode{attnnet.ModeIcon}
	case "both":
		modes = []attnnet.Mode{attnnet.ModeGlobal, attnnet.ModeIcon}
	default:
		fmt.Fprintln(os.Stderr, "unknown mode:", *modeSel)
		os.Exit(2)
	}

	for _, mode := range modes {
		name := "global"
		if mode == attnnet.ModeIcon {
			name = "icon"
		}
		fmt.Printf("\n===== training mode=%s (seed %d) =====\n", name, *seed)
		mcfg := cfg
		mcfg.Mode = mode
		tr := attnnet.NewTrainer(attnnet.New(mcfg, *seed))
		for ep := 1; ep <= *epochs; ep++ {
			t0 := time.Now()
			perm := shuffle(len(train), *seed+int64(ep))
			var ceSum, glSum float64
			var inBatch int
			tr.Reset()
			for bi, ti := range perm {
				idx := train[ti]
				ce, gl := tr.Step(set.Img(idx), set.Mask(idx))
				ceSum += ce
				glSum += gl
				inBatch++
				if inBatch == *batch || bi == len(perm)-1 {
					tr.Apply(1/float64(inBatch), *lr)
					tr.Reset()
					inBatch = 0
				}
			}
			valCE, mass, iou := evalVal(tr, set, val)
			dt := time.Since(t0).Round(time.Millisecond)
			fmt.Printf("ep %3d  loss %.4f  gate %.4f | valCE %.4f  massInIcon %.3f  topK-IoU %.3f  %v\n",
				ep, ceSum/float64(len(train)), glSum/float64(len(train)), valCE, mass, iou, dt)
		}
		wp := filepath.Join(*outDir, "weights_"+name+".gob")
		if err := tr.Model().Save(wp); err != nil {
			fmt.Fprintln(os.Stderr, "save:", err)
			os.Exit(1)
		}
		fmt.Printf("saved %s\n", wp)
	}

	fmt.Println("\n===== comparison (validation, final weights) =====")
	for _, mode := range []attnnet.Mode{attnnet.ModeGlobal, attnnet.ModeIcon} {
		name := "global"
		if mode == attnnet.ModeIcon {
			name = "icon"
		}
		m, err := attnnet.Load(filepath.Join(*outDir, "weights_"+name+".gob"))
		if err != nil {
			fmt.Fprintln(os.Stderr, "load:", err)
			continue
		}
		tr := attnnet.NewTrainer(m)
		_, mass, iou := evalVal(tr, set, val)
		fmt.Printf("%-6s: attention mass inside icon = %.3f   top-k IoU = %.3f\n", name, mass, iou)
	}
}

// evalVal reports mean CE, mean attention mass inside the icon mask and mean
// top-k IoU over the validation samples.
func evalVal(tr *attnnet.Trainer, set *attnnet.Set, val []int) (ce, mass, iou float64) {
	var n float64
	for _, i := range val {
		p, lce, _ := tr.Eval(set.Img(i), set.Mask(i))
		mask := set.Mask(i)
		var k float64
		var msum float64
		for j := range mask {
			if mask[j] != 0 {
				k++
				msum += p[j]
			}
		}
		inter := topKInter(p, mask)
		ce += lce
		mass += msum
		if un := 2*k - inter; un > 0 {
			iou += inter / un
		}
		n++
	}
	return ce / n, mass / n, iou / n
}

// topKInter returns |top-k predicted patches in mask| with k = |mask|.
func topKInter(p []float64, mask []byte) float64 {
	type idxv struct {
		i int
		v float64
	}
	arr := make([]idxv, len(p))
	var k float64
	for j := 0; j < len(mask); j++ {
		if mask[j] != 0 {
			k++
		}
		arr[j] = idxv{j, p[j]}
	}
	for i := 1; i < len(arr); i++ {
		for j := i; j > 0 && arr[j].v > arr[j-1].v; j-- {
			arr[j], arr[j-1] = arr[j-1], arr[j]
		}
	}
	var inter float64
	for i := 0; i < int(k) && i < len(arr); i++ {
		if mask[arr[i].i] != 0 {
			inter++
		}
	}
	return inter
}

// shuffle returns [0..n) in a seeded pseudo-random order.
func shuffle(n int, seed int64) []int {
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i
	}
	r := seed
	for i := n - 1; i > 0; i-- {
		j := int(r >> 33 % int64(i+1))
		if j < 0 {
			j = -j
		}
		r = r*6364136223846793005 + 1442695040888963407
		perm[i], perm[j] = perm[j], perm[i]
	}
	return perm
}

