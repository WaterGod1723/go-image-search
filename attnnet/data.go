package attnnet

import (
	"encoding/binary"
	"errors"
	"io"
	"math/rand"
	"os"
	"path/filepath"
)

// The test set is a flat binary file: 8-byte magic, uint32 sample count, then
// per sample ImageSize*ImageSize*3 bytes of RGB (uint8) followed by nTok bytes
// of per-patch icon mask (0/1).

const setMagic = "ATTNSET1"

// Set holds a generated test set.
type Set struct {
	Cfg    Config
	Images []byte // len = N * ImageSize*ImageSize*3
	Masks  []byte // len = N * nTok
	N      int
}

// NewSet allocates an empty set for cfg.
func NewSet(cfg Config, n int) *Set {
	return &Set{
		Cfg:    cfg,
		Images: make([]byte, n*cfg.ImageSize*cfg.ImageSize*3),
		Masks:  make([]byte, n*cfg.nTok()),
		N:      n,
	}
}

// Img returns the RGB bytes of sample i.
func (s *Set) Img(i int) []byte {
	px := s.Cfg.ImageSize * s.Cfg.ImageSize * 3
	return s.Images[i*px : (i+1)*px]
}

// Mask returns the per-patch icon mask of sample i.
func (s *Set) Mask(i int) []byte {
	nt := s.Cfg.nTok()
	return s.Masks[i*nt : (i+1)*nt]
}

// Save writes the set to path.
func (s *Set) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := f.WriteString(setMagic); err != nil {
		return err
	}
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], uint32(s.N))
	if _, err := f.Write(buf[:]); err != nil {
		return err
	}
	if _, err := f.Write(s.Images); err != nil {
		return err
	}
	return writeAll(f, s.Masks)
}

// LoadSet reads a set written by Save.
func LoadSet(path string) (cfg Config, s *Set, err error) {
	f, err := os.Open(path)
	if err != nil {
		return cfg, nil, err
	}
	defer f.Close()
	magic := make([]byte, len(setMagic))
	if _, err := io.ReadFull(f, magic); err != nil {
		return cfg, nil, err
	}
	if string(magic) != setMagic {
		return cfg, nil, errors.New("not an attnnet set file")
	}
	var buf [4]byte
	if _, err := io.ReadFull(f, buf[:]); err != nil {
		return cfg, nil, err
	}
	cfg = Default()
	n := int(binary.LittleEndian.Uint32(buf[:]))
	s = NewSet(cfg, n)
	px := cfg.ImageSize * cfg.ImageSize * 3
	if _, err := io.ReadFull(f, s.Images); err != nil {
		return cfg, nil, err
	}
	if _, err := io.ReadFull(f, s.Masks); err != nil {
		return cfg, nil, err
	}
	_ = px
	return cfg, s, nil
}

func writeAll(f *os.File, b []byte) error {
	for len(b) > 0 {
		n, err := f.Write(b)
		if err != nil {
			return err
		}
		b = b[n:]
	}
	return nil
}

// Split shuffles the set by seed and returns a training subset (frac of the
// samples) and a validation subset. Indices stay inside the same Set.
func Split(s *Set, frac float64, seed int64) (train, val []int) {
	perm := rand.New(rand.NewSource(seed)).Perm(s.N)
	k := int(float64(s.N) * frac)
	return perm[:k], perm[k:]
}

// DataDir returns the module-local data directory (gitignored by the parent).
func DataDir() string { return "data" }

// SetPath returns the path to the generated test set binary.
func SetPath() string { return filepath.Join(DataDir(), "testset.bin") }
