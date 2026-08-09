package nnengine

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"image"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"go-image-search/internal/imageproc"
	"go-image-search/internal/nnengine/sczl"
)

// Hit is one ranked search result.
type Hit struct {
	Name      string    `json:"name"`
	Rank      int       `json:"rank"`
	Scores    []float64 `json:"scores"`    // 7 per-feature similarities
	MaskScore float64   `json:"maskScore"` // rotation-aligned dice
	RankScore float64   `json:"rankScore"` // 实际用于排序的神经网络相关度
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

// BuildIndex 扫描 dir（递归全部子目录）中的全部受支持图片（png/jpg/gif/webp/
// bmp/tiff/avif/svg 等）构建 Feat + sczl 描述子。Go 原生解码失败的特殊格式
// （svg/avif 等）在 GUI 环境中会交给注入的 webview 多线程 Worker 转换器
// （imageproc.SetConverter）转换；解码与特征提取以多 worker 并行以提升效率。
// 已有持久化缓存时复用。
func (e *Engine) BuildIndex(dir string, cachePath string) error {
	files, err := imageproc.LoadSupported(dir)
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

	// 并行 worker 池：解码（含可能的 webview 转换）+ 特征提取并发执行，
	// 按原文件顺序合并结果，保证 refs/names/sczlIx 三者在索引上对齐。
	type result struct {
		idx  int
		ref  *Feat
		name string
		d    sczl.Descriptor
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > 8 {
		workers = 8
	}
	if workers < 1 {
		workers = 1
	}
	if workers > len(files) {
		workers = len(files)
	}
	jobs := make(chan int)
	results := make(chan result, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for fi := range jobs {
				p := files[fi]
				img, err := imageproc.Load(p)
				if err != nil {
					results <- result{idx: fi}
					continue
				}
				nrgba := toNRGBA(img)
				px := refPixels(nrgba)
				if len(px) == 0 {
					results <- result{idx: fi}
					continue
				}
				results <- result{
					idx: fi,
					ref: buildFeat(px),
					// full path (forward slashes) so the GUI can load thumbnails
					// from any user-selected directory.
					name: filepath.ToSlash(p),
					d:    sczl.Extract(nrgba),
				}
			}
		}()
	}
	go func() {
		for fi := range files {
			jobs <- fi
		}
		close(jobs)
		wg.Wait()
		close(results)
	}()

	refs := make([]*Feat, 0, len(files))
	names := make([]string, 0, len(files))
	ix := sczl.New()
	next := 0
	pending := make(map[int]result)
	for r := range results {
		pending[r.idx] = r
		for {
			cur, ok := pending[next]
			if !ok {
				break
			}
			delete(pending, next)
			next++
			if cur.ref == nil {
				continue
			}
			refs = append(refs, cur.ref)
			names = append(names, cur.name)
			ix.AddImage(cur.name, cur.d)
		}
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
	var rankScore []float64
	if e.nn != nil {
		ranked, rankScore = rankNN(q, e.refs, e.nn, e.sczlIx, sczl.Extract(img))
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
		rankScore = pref
	}
	hits := make([]Hit, 0, topK)
	for i := 0; i < topK; i++ {
		idx := ranked[i]
		hits = append(hits, Hit{
			Name:      e.names[idx],
			Rank:      i + 1,
			Scores:    scores(q, e.refs[idx]),
			MaskScore: maskScore(q, e.refs[idx]),
			RankScore: rankScore[i],
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
