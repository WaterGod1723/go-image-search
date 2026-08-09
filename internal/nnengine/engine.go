package nnengine

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"image"
	"log"
	"os"
	"path/filepath"
	"time"

	"go-image-search/internal/nnengine/sczl"
)

// Hit is one ranked search result.
type Hit struct {
	Name      string    `json:"name"`
	Rank      int       `json:"rank"`
	Scores    []float64 `json:"scores"`    // 7 per-feature similarities
	MaskScore float64   `json:"maskScore"` // rotation-aligned dice
}

// cacheEntry is one persisted reference: name + the two descriptor sets.
type cacheEntry struct {
	Name  string
	Feat  *Feat
	SCZL  sczl.Descriptor
	MTime time.Time
}

// Engine is the NN search engine (trained MLP fusing hand-crafted features +
// the sczl color-agnostic expert, with a two-stage coarse pre-filter).
type Engine struct {
	refs      []*Feat
	names     []string
	sczlIx    *sczl.Index
	nn        *MLP
	dir       string
	cachePath string
}

// New creates an engine; weightsPath is the trained MLP weights file (may be
// empty; searching then falls back to an uninitialized ranker which returns
// the cheap pre-filter order — the caller should set weights via LoadWeights).
func New(weightsPath string) *Engine {
	e := &Engine{sczlIx: sczl.New()}
	if weightsPath != "" {
		if m, err := loadMLP(weightsPath); err == nil && len(m.W1) == nnH1*nnInput {
			e.nn = m
		} else if err != nil {
			log.Printf("nnengine: weights not loaded (%v)", err)
		}
	}
	return e
}

// LoadWeights loads (or replaces) the trained MLP weights.
func (e *Engine) LoadWeights(path string) error {
	m, err := loadMLP(path)
	if err != nil {
		return err
	}
	if len(m.W1) != nnH1*nnInput {
		return fmt.Errorf("weights shape mismatch (%s)", path)
	}
	e.nn = m
	return nil
}

// Len returns the number of indexed references.
func (e *Engine) Len() int { return len(e.refs) }

// Names returns the indexed reference file names.
func (e *Engine) Names() []string { return append([]string(nil), e.names...) }

// LoadIndex restores a persisted index cache (refs + sczl descriptors).
func (e *Engine) LoadIndex(cachePath string) error {
	e.cachePath = cachePath
	dir := filepath.Dir(cachePath)
	if err := e.loadCache(dir); err != nil {
		return err
	}
	return nil
}

// BuildIndex scans dir for reference sprites and builds the Feat + sczl
// descriptors, reusing a persisted cache when present.
func (e *Engine) BuildIndex(dir string, cachePath string) error {
	files, err := listPNG(dir)
	if err != nil {
		return err
	}
	if cachePath != "" {
		e.cachePath = cachePath
		if err := e.loadCache(dir); err == nil && len(e.refs) == len(files) {
			e.dir = dir
			return nil
		}
	}
	refs := make([]*Feat, 0, len(files))
	names := make([]string, 0, len(files))
	ix := sczl.New()
	for _, fn := range files {
		img, err := loadPNG(filepath.Join(dir, fn))
		if err != nil {
			continue
		}
		px := refPixels(img)
		if len(px) == 0 {
			continue
		}
		refs = append(refs, buildFeat(px))
		// full path (forward slashes) so the GUI can load thumbnails from any
		// user-selected directory; the frontend strips the path for display.
		names = append(names, filepath.ToSlash(filepath.Join(dir, fn)))
		d := sczl.Extract(img)
		ix.AddImage(fn, d)
	}
	e.refs = refs
	e.names = names
	e.sczlIx = ix
	e.dir = dir
	if cachePath != "" {
		e.cachePath = cachePath
		_ = e.saveCache()
	}
	return nil
}

// loadCache restores a persisted index (refs + sczl descriptors).
func (e *Engine) loadCache(dir string) error {
	if e.cachePath == "" {
		return fmt.Errorf("no cache path")
	}
	data, err := os.ReadFile(e.cachePath)
	if err != nil {
		return err
	}
	var entries []cacheEntry
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&entries); err != nil {
		return err
	}
	refs := make([]*Feat, 0, len(entries))
	names := make([]string, 0, len(entries))
	ix := sczl.New()
	for _, en := range entries {
		refs = append(refs, en.Feat)
		names = append(names, en.Name)
		ix.AddImage(en.Name, en.SCZL)
	}
	e.refs = refs
	e.names = names
	e.sczlIx = ix
	return nil
}

func (e *Engine) saveCache() error {
	entries := make([]cacheEntry, len(e.refs))
	for i := range e.refs {
		entries[i] = cacheEntry{Name: e.names[i], Feat: e.refs[i], SCZL: e.sczlIx.Entries[i]}
	}
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(entries); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(e.cachePath), 0o755); err != nil {
		return err
	}
	return os.WriteFile(e.cachePath, buf.Bytes(), 0o644)
}

// Search ranks the references for the query image and returns the top-k hits.
// If the trained weights are missing it degrades to the cheap pre-filter order
// with a note.
func (e *Engine) Search(img image.Image, topK int) ([]Hit, error) {
	if len(e.refs) == 0 {
		return nil, fmt.Errorf("索引为空，请先构建或加载索引")
	}
	if topK <= 0 {
		topK = 5
	}
	if topK > len(e.refs) {
		topK = len(e.refs)
	}
	px := extractQuery(toNRGBA(img))
	if len(px) == 0 {
		return nil, fmt.Errorf("分割未提取到 sprite 像素（背景与前景差异过小？）")
	}
	q := buildFeat(px)
	var ranked []int
	if e.nn != nil {
		ranked = rankNN(q, e.refs, e.nn, e.sczlIx, sczl.Extract(img))
	} else {
		// no weights: fall back to the cheap pre-filter ranking
		ranked = make([]int, len(e.refs))
		pref := make([]float64, len(e.refs))
		for i, r := range e.refs {
			pref[i] = cheapPref(q, r)
		}
		for i := range ranked {
			ranked[i] = i
		}
		insertionSort(ranked, func(a, b int) bool { return pref[a] > pref[b] })
	}
	hits := make([]Hit, 0, topK)
	for i := 0; i < topK; i++ {
		idx := ranked[i]
		hits = append(hits, Hit{
			Name:      e.names[idx],
			Rank:      i + 1,
			Scores:    scores(q, e.refs[idx]),
			MaskScore: maskScore(q, e.refs[idx]),
		})
	}
	return hits, nil
}

// toNRGBA converts any decoded image to *image.NRGBA (the segmentation and
// feature pipeline works on NRGBA).
func toNRGBA(img image.Image) *image.NRGBA {
	if n, ok := img.(*image.NRGBA); ok {
		return n
	}
	b := img.Bounds()
	out := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			out.Set(x, y, img.At(b.Min.X+x, b.Min.Y+y))
		}
	}
	return out
}
