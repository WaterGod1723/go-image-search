// Package iconnet implements a small, dependency-free RGBA icon embedding CNN
// trained in Python (pynet/export_go.py) and executed entirely in Go.
//
// The network is a 4-block CNN: two 3x3 convs + BatchNorm + ReLU per block,
// squeeze-and-excitation (SE) channel attention, 2x2 max pooling, then global
// average pooling, a linear projection to a low-dim embedding, and L2
// normalization. Input is a 64x64 RGBA sprite (channel order R,G,B,A) with
// values in [0,1]; output is an embedding vector on the unit sphere.
//
// Weights are loaded from a binary file exported by pynet/export_go.py.
package iconnet

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// Net is a loaded icon embedding network.
type Net struct {
	InputSize int   // sprite side (N), e.g. 64
	EmbDim    int   // embedding dimension
	Ch        []int // channels per block
	Block     []block
	EmbW      []float64 // embed linear: EmbDim x lastCh (row-major)
	EmbB      []float64 // embed bias: EmbDim
}

// block holds one conv+bn+relu+conv+bn+relu+se+pool stage.
type block struct {
	CIn, COut int
	Conv1W    []float64 // COut x CIn x 3 x 3
	Conv1B    []float64 // COut
	BN1G      []float64 // gamma
	BN1B      []float64 // beta
	BN1M      []float64 // running mean
	BN1V      []float64 // running var
	Conv2W    []float64 // COut x COut x 3 x 3
	Conv2B    []float64 // COut
	BN2G      []float64
	BN2B      []float64
	BN2M      []float64
	BN2V      []float64
	SER       int       // SE bottleneck channels = COut/8
	SE1W      []float64 // SER x COut
	SE1B      []float64 // SER
	SE2W      []float64 // COut x SER
	SE2B      []float64 // COut
}

const bnEps = 1e-5

// Load reads a weight file produced by pynet/export_go.py.
func Load(r io.Reader) (*Net, error) {
	magic := make([]byte, 4)
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, fmt.Errorf("iconnet: read magic: %w", err)
	}
	if string(magic) != "ICN1" {
		return nil, fmt.Errorf("iconnet: bad magic %q (want ICN1)", magic)
	}
	var nU8 uint8
	if err := binary.Read(r, binary.LittleEndian, &nU8); err != nil {
		return nil, fmt.Errorf("iconnet: read nblocks: %w", err)
	}
	n := int(nU8)
	if n < 1 || n > 16 {
		return nil, fmt.Errorf("iconnet: nblocks=%d out of range", n)
	}
	net := &Net{InputSize: 64, EmbDim: 64, Ch: make([]int, n)}
	for i := 0; i < n; i++ {
		ci := readU32(r)
		if ci <= 0 || ci > 512 {
			return nil, fmt.Errorf("iconnet: block %d channel %d out of range", i, ci)
		}
		net.Ch[i] = ci
	}
	prev := 4
	for i := 0; i < n; i++ {
		ci := net.Ch[i]
		b := block{CIn: prev, COut: ci}
		b.Conv1W = readF(r, ci*prev*9)
		b.Conv1B = readF(r, ci)
		b.BN1G = readF(r, ci)
		b.BN1B = readF(r, ci)
		b.BN1M = readF(r, ci)
		b.BN1V = readF(r, ci)
		b.Conv2W = readF(r, ci*ci*9)
		b.Conv2B = readF(r, ci)
		b.BN2G = readF(r, ci)
		b.BN2B = readF(r, ci)
		b.BN2M = readF(r, ci)
		b.BN2V = readF(r, ci)
		b.SER = ci / 8
		b.SE1W = readF(r, b.SER*ci)
		b.SE1B = readF(r, b.SER)
		b.SE2W = readF(r, ci*b.SER)
		b.SE2B = readF(r, ci)
		net.Block = append(net.Block, b)
		prev = int(ci)
	}
	last := net.Ch[n-1]
	net.EmbW = readF(r, 64*last)
	net.EmbB = readF(r, 64)
	return net, nil
}

// LoadFile opens path and loads the network.
func LoadFile(path string) (*Net, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Load(f)
}

func readU32(r io.Reader) int {
	var v uint32
	if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
		panic(fmt.Sprintf("iconnet: read u32: %v", err))
	}
	return int(v)
}

func readF(r io.Reader, n int) []float64 {
	if n <= 0 {
		panic(fmt.Sprintf("iconnet: bad tensor len %d", n))
	}
	buf := make([]byte, n*4)
	if _, err := io.ReadFull(r, buf); err != nil {
		panic(fmt.Sprintf("iconnet: read tensor(%d): %v", n, err))
	}
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:])))
	}
	return out
}
