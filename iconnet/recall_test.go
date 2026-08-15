package iconnet

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// TestRecallIndomain reproduces the Python in-domain recall with Go's own
// preprocessing + forward + retrieval. Uses pynet/data/test_indomain.
func TestRecallIndomain(t *testing.T) {
	root := filepath.Join("..", "pynet", "data", "test_indomain")
	manifest := filepath.Join(root, "manifest.json")
	if _, err := os.Stat(manifest); err != nil {
		t.Skip("no test_indomain data")
	}
	net, err := LoadFile(filepath.Join("weights.bin"))
	if err != nil {
		t.Fatal(err)
	}
	refDir := filepath.Join("..", "test_pngs")
	refNames, err := listPNGs(refDir)
	if err != nil {
		t.Fatal(err)
	}
	idx := NewIndex(net, refNames, func(k string) (*Sprite, bool) {
		return RefSprite(filepath.Join(refDir, k), 64)
	})
	fmt.Printf("indexed %d refs\n", idx.Len())

	type entry struct {
		Image string `json:"image"`
		Src   string `json:"src"`
	}
	raw, _ := os.ReadFile(manifest)
	var entries []entry
	if err := json.Unmarshal(raw, &entries); err != nil {
		t.Fatal(err)
	}
	hits, miss, total := 0, 0, 0
	goMiss := []string{}
	for _, e := range entries {
		sp, ok := QuerySprite(filepath.Join(root, e.Image), 64)
		if !ok {
			miss++
			continue
		}
		q := idx.EmbedSprite(sp)
		keys, _ := idx.Query(q, 1)
		total++
		if len(keys) > 0 && keys[0] == e.Src {
			hits++
		} else {
			goMiss = append(goMiss, e.Image)
		}
	}
	fmt.Printf("GO misses (%d): %v\n", len(goMiss), goMiss)
	recall := float64(hits) / float64(total)
	fmt.Printf("GO  in-domain recall@1 = %.4f (n=%d miss=%d)\n", recall, total, miss)
	fmt.Printf("PY  in-domain recall@1 = %.4f\n", 0.7750)
	if recall < 0.70 {
		t.Errorf("Go recall too low: %.4f", recall)
	}
}

func listPNGs(dir string) ([]string, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range ents {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".png" {
			out = append(out, e.Name())
		}
	}
	return out, nil
}
