package iconnet

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestFullQueryPipeline runs Go's full query pipeline (segment+trim+letterbox)
// and compares the resulting embedding against Python's reference.
func TestFullQueryPipeline(t *testing.T) {
	refJSON := filepath.Join("..", "test_align", "q0_emb.json")
	if _, err := os.Stat(refJSON); err != nil {
		t.Skip("no reference data")
	}
	net, err := LoadFile(filepath.Join("weights.bin"))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]struct {
		Emb string `json:"emb"`
	}
	raw, _ := os.ReadFile(refJSON)
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	q0, ok := m["q0"]
	if !ok {
		t.Fatal("no q0 in ref")
	}
	qpath := filepath.Join("..", "pynet", "data", "test_cross", "sample_00000.png")
	sp, ok := QuerySprite(qpath, 64)
	if !ok {
		t.Fatal("QuerySprite failed")
	}
	e := net.Embed(sp.Tensor())
	want := make([]float64, 64)
	rb, _ := base64.StdEncoding.DecodeString(q0.Emb)
	for j := 0; j < 64; j++ {
		want[j] = float64(math.Float32frombits(binary.LittleEndian.Uint32(rb[j*4:])))
	}
	fmt.Printf("go full-pipeline embed vs python: cos = %.8f\n", Cosine(e, want))
	if Cosine(e, want) < 0.999999 {
		t.Errorf("full pipeline embedding mismatch: cos=%.8f", Cosine(e, want))
	}
}
