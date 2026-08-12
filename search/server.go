package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/gob"
	"encoding/json"
	"image"
	"image/png"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"image-search-test/search/sczl"

	_ "embed"
)

//go:embed web.html
var webHTML []byte

const queryCacheCap = 256

// indexEntry is one cached reference: name + mtime + computed features (both
// the hand-crafted Feat and the color-agnostic sczl descriptor, persisted so
// the neural ranker's second expert needs no PNG re-decode on startup).
type indexEntry struct {
	Name    string
	ModTime time.Time
	Feat    *Feat
	SCZL    sczl.Descriptor
}

// indexCacheFile is the on-disk gob payload for the index cache.
type indexCacheFile struct {
	Dir     string
	Entries []indexEntry
}

type searchResult struct {
	Name      string    `json:"name"`
	Rank      int       `json:"rank"`
	Scores    []float64 `json:"scores"`    // 7 per-feature similarities
	MaskScore float64   `json:"maskScore"` // rotation-aligned dice
}

type searchResponse struct {
	Cached       bool           `json:"cached"`
	QueryW       int            `json:"queryW"`
	QueryH       int            `json:"queryH"`
	SpritePngB64 string         `json:"spritePng,omitempty"`
	Results      []searchResult `json:"results"`
	ElapsedMs    int64          `json:"elapsedMs"`
	Note         string         `json:"note,omitempty"`
}

type statusResponse struct {
	RefDir         string    `json:"refDir"`
	Count          int       `json:"count"`
	LastBuilt      time.Time `json:"lastBuilt"`
	CacheFile      bool      `json:"cacheFile"`
	QueryCacheSize int       `json:"queryCacheSize"`
	Names          []string  `json:"names"`
}

type buildResponse struct {
	Count     int    `json:"count"`
	Rebuilt   int    `json:"rebuilt"`
	Cached    int    `json:"cached"`
	ElapsedMs int64  `json:"elapsedMs"`
	CachePath string `json:"cachePath"`
	Error     string `json:"error,omitempty"`
}

// server holds the index (in memory) plus the query-result LRU cache.
type server struct {
	mu        sync.RWMutex
	root      string
	refDir    string
	cachePath string
	entries   []indexEntry
	lastBuilt time.Time
	cache     *indexCacheFile

	// neural ranker: nn != nil enables the trained attention fusion network
	// (with the sczl color-agnostic expert) in /api/search; otherwise falls back
	// to the hand-tuned compositeAdaptive.
	nn     *AttnNet
	sczlIx *sczl.Index

	qCache map[[32]byte]*searchResponse
	qOrder [][32]byte // FIFO order for LRU eviction
}

func newServer(root string) *server {
	s := &server{
		root:      root,
		refDir:    filepath.Join(root, "test_pngs"),
		cachePath: filepath.Join(root, ".searchcache", "index.gob"),
		qCache:    make(map[[32]byte]*searchResponse, queryCacheCap),
	}
	// Best-effort: reuse a previously persisted index on startup.
	if err := s.loadCache(); err != nil {
		log.Printf("index cache not loaded: %v", err)
	} else if len(s.entries) > 0 {
		log.Printf("index cache loaded: %d entries from %s", len(s.entries), s.cachePath)
	}
	s.loadNN()
	// Auto-build the default reference index on first run so the server is
	// immediately usable; subsequent starts reuse the local cache (fast).
	if len(s.entries) == 0 {
		if resp := s.buildIndex(s.refDir); resp.Error != "" {
			log.Printf("auto-build failed: %s", resp.Error)
		} else {
			log.Printf("auto-built index: %d refs from %s", resp.Count, s.refDir)
		}
	}
	return s
}

// loadNN loads the trained fusion attention-net weights. The path comes from the
// NN_WEIGHTS env var (default <root>/weights.gob); if missing or invalid the
// server degrades gracefully to the heuristic ranker.
func (s *server) loadNN() {
	path := os.Getenv("NN_WEIGHTS")
	if path == "" {
		path = filepath.Join(s.root, "weights.gob")
	}
	m, err := loadAttnNet(path)
	if err != nil {
		log.Printf("neural ranker disabled (no weights at %s): %v", path, err)
		return
	}
	if !m.valid() {
		log.Printf("neural ranker disabled: weights shape mismatch (%s)", path)
		return
	}
	s.nn = m
	log.Printf("neural ranker enabled: %s", path)
}

func (s *server) routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleIndex)
	mux.HandleFunc("/api/status", s.handleStatus)
	mux.HandleFunc("/api/build", s.handleBuild)
	mux.HandleFunc("/api/search", s.handleSearch)
	mux.HandleFunc("/api/refs", s.handleRefs)
	mux.HandleFunc("/api/ref", s.handleRef)
	mux.HandleFunc("/api/samples", s.handleSamples)
	mux.HandleFunc("/api/sample", s.handleSample)
	mux.HandleFunc("/api/clearcache", s.handleClearCache)
	return mux
}

func (s *server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(webHTML)
}

func (s *server) handleStatus(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	names := make([]string, 0, len(s.entries))
	for _, e := range s.entries {
		names = append(names, e.Name)
	}
	writeJSON(w, statusResponse{
		RefDir:         s.refDir,
		Count:          len(s.entries),
		LastBuilt:      s.lastBuilt,
		CacheFile:      fileExists(s.cachePath),
		QueryCacheSize: len(s.qCache),
		Names:          names,
	})
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

func (s *server) handleBuild(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST required", http.StatusMethodNotAllowed)
		return
	}
	dir := r.URL.Query().Get("dir")
	if dir == "" {
		dir = "test_pngs"
	}
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(s.root, dir)
	}
	start := time.Now()
	resp := s.buildIndex(dir)
	resp.ElapsedMs = time.Since(start).Milliseconds()
	resp.CachePath = s.cachePath
	writeJSON(w, resp)
}

// buildIndex scans dir for PNGs and (re)builds features. Files whose mtime
// matches the existing cache are reused; the rest are recomputed. The result is
// persisted to disk so subsequent runs skip the work entirely.
func (s *server) buildIndex(dir string) buildResponse {
	files, err := listPNG(dir)
	if err != nil {
		return buildResponse{Error: err.Error()}
	}
	prev := map[string]indexEntry{}
	s.mu.RLock()
	if s.cache != nil && s.cache.Dir == dir {
		for _, e := range s.cache.Entries {
			prev[e.Name] = e
		}
	}
	s.mu.RUnlock()

	entries := make([]indexEntry, 0, len(files))
	rebuilt, cached := 0, 0
	for _, fn := range files {
		path := filepath.Join(dir, fn)
		fi, err := os.Stat(path)
		if err != nil {
			continue
		}
		if pe, ok := prev[fn]; ok && pe.Feat != nil && pe.ModTime.Equal(fi.ModTime()) {
			entries = append(entries, pe)
			cached++
			continue
		}
		img, err := loadPNG(path)
		if err != nil {
			continue
		}
		px := refPixels(img)
		if len(px) == 0 {
			continue
		}
		f := buildFeat(px)
		entries = append(entries, indexEntry{Name: fn, ModTime: fi.ModTime(), Feat: f, SCZL: sczl.Extract(img)})
		rebuilt++
	}

	cf := &indexCacheFile{Dir: dir, Entries: entries}
	s.mu.Lock()
	s.refDir = dir
	s.entries = entries
	s.lastBuilt = time.Now()
	s.cache = cf
	s.rebuildSCZL()
	// Reference set changed: invalidate the query cache.
	s.qCache = make(map[[32]byte]*searchResponse, queryCacheCap)
	s.qOrder = nil
	s.mu.Unlock()

	if err := s.saveCache(cf); err != nil {
		log.Printf("warn: cache save failed: %v", err)
	}
	return buildResponse{Count: len(entries), Rebuilt: rebuilt, Cached: cached}
}

func (s *server) loadCache() error {
	data, err := os.ReadFile(s.cachePath)
	if err != nil {
		return err
	}
	var c indexCacheFile
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&c); err != nil {
		return err
	}
	s.mu.Lock()
	s.cache = &c
	s.entries = c.Entries
	s.refDir = c.Dir
	s.lastBuilt = time.Now()
	s.rebuildSCZL()
	s.mu.Unlock()
	return nil
}

// rebuildSCZL (re)builds the sczl reference descriptors aligned 1:1 with
// s.entries (same order), as the second expert the neural ranker consumes.
// Persisted descriptors are reused; only missing ones (older cache format) are
// re-extracted and written back so the next save persists them.
// Must be called with s.mu held (write lock).
func (s *server) rebuildSCZL() {
	ix := sczl.New()
	extracted := false
	for i := range s.entries {
		e := &s.entries[i]
		d := e.SCZL
		if !d.Valid {
			if img, err := loadPNG(filepath.Join(s.refDir, e.Name)); err == nil {
				d = sczl.Extract(img)
				e.SCZL = d
				extracted = true
			}
		}
		// Keep 1:1 alignment with entries even if extraction fails (zero
		// descriptor simply contributes a low similarity).
		ix.AddImage(e.Name, d)
	}
	s.sczlIx = ix
	if extracted && s.cache != nil {
		_ = s.saveCache(s.cache)
	}
}

func (s *server) saveCache(c *indexCacheFile) error {
	if c == nil {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(s.cachePath), 0o755); err != nil {
		return err
	}
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(c); err != nil {
		return err
	}
	return os.WriteFile(s.cachePath, buf.Bytes(), 0o644)
}

func (s *server) handleRefs(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]string, 0, len(s.entries))
	for _, e := range s.entries {
		out = append(out, e.Name)
	}
	writeJSON(w, map[string]any{"dir": s.refDir, "names": out})
}

func (s *server) handleRef(w http.ResponseWriter, r *http.Request) {
	name := filepath.Base(r.URL.Query().Get("name"))
	if name == "" || name == "." {
		http.Error(w, "bad name", 400)
		return
	}
	s.mu.RLock()
	dir := s.refDir
	s.mu.RUnlock()
	http.ServeFile(w, r, filepath.Join(dir, name))
}

func (s *server) handleSamples(w http.ResponseWriter, r *http.Request) {
	dir := filepath.Join(s.root, "test_set")
	files, err := listPNG(dir)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	writeJSON(w, map[string]any{"dir": dir, "samples": files})
}

func (s *server) handleSample(w http.ResponseWriter, r *http.Request) {
	name := filepath.Base(r.URL.Query().Get("name"))
	if name == "" || name == "." {
		http.Error(w, "bad name", 400)
		return
	}
	http.ServeFile(w, r, filepath.Join(s.root, "test_set", name))
}

func (s *server) handleClearCache(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST required", http.StatusMethodNotAllowed)
		return
	}
	s.mu.Lock()
	s.qCache = make(map[[32]byte]*searchResponse, queryCacheCap)
	s.qOrder = nil
	s.mu.Unlock()
	writeJSON(w, map[string]any{"cleared": true})
}

func (s *server) handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST required", http.StatusMethodNotAllowed)
		return
	}
	imgBytes, err := readQueryImage(r, s.root)
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	if len(imgBytes) == 0 {
		http.Error(w, "no image provided (upload 'image' or pass ?sample=NAME)", 400)
		return
	}

	// Query-result cache: hash the raw bytes so identical queries are instant.
	key := sha256.Sum256(imgBytes)
	s.mu.RLock()
	if cached, ok := s.qCache[key]; ok {
		s.mu.RUnlock()
		resp := *cached
		resp.Cached = true
		writeJSON(w, resp)
		return
	}
	s.mu.RUnlock()

	s.mu.RLock()
	entries := s.entries
	s.mu.RUnlock()
	if len(entries) == 0 {
		writeJSON(w, searchResponse{Note: "索引为空，请先在“索引管理”中构建索引"})
		return
	}

	start := time.Now()
	img, err := decodePNGBytes(imgBytes)
	if err != nil {
		http.Error(w, "decode: "+err.Error(), 400)
		return
	}
	px := extractQuery(img)
	spriteB64 := ""
	if len(px) > 0 {
		spriteB64 = renderSpritePng(img, px)
	}
	if len(px) == 0 {
		writeJSON(w, searchResponse{
			QueryW:       img.Bounds().Dx(),
			QueryH:       img.Bounds().Dy(),
			SpritePngB64: spriteB64,
			Note:         "分割未提取到 sprite 像素（背景与前景差异过小？）",
			ElapsedMs:    time.Since(start).Milliseconds(),
		})
		return
	}
	q := buildFeat(px)
	qd := sczl.Extract(img)
	s.mu.RLock()
	refs := make([]*Feat, len(s.entries))
	for i := range s.entries {
		refs[i] = s.entries[i].Feat
	}
	nn := s.nn
	sczlIx := s.sczlIx
	s.mu.RUnlock()

	var ranked []int
	if nn != nil {
		// neural ranker: trained attention net fusing our features + the sczl expert,
		// with shape-context shortlist refinement.
		ranked = rankNN(q, refs, nn, sczlIx, qd)
	} else {
		ranked = compositeAdaptive(q, refs)
	}

	topK := 12
	if len(ranked) < topK {
		topK = len(ranked)
	}
	results := make([]searchResult, 0, topK)
	for i := 0; i < topK; i++ {
		idx := ranked[i]
		results = append(results, searchResult{
			Name:      entries[idx].Name,
			Rank:      i + 1,
			Scores:    scores(q, refs[idx]),
			MaskScore: maskScore(q, refs[idx]),
		})
	}
	resp := searchResponse{
		QueryW:       img.Bounds().Dx(),
		QueryH:       img.Bounds().Dy(),
		SpritePngB64: spriteB64,
		Results:      results,
		ElapsedMs:    time.Since(start).Milliseconds(),
	}

	s.mu.Lock()
	if _, ok := s.qCache[key]; !ok {
		cp := resp
		s.qCache[key] = &cp
		s.qOrder = append(s.qOrder, key)
		for len(s.qOrder) > queryCacheCap {
			old := s.qOrder[0]
			s.qOrder = s.qOrder[1:]
			delete(s.qCache, old)
		}
	}
	s.mu.Unlock()

	writeJSON(w, resp)
}

// readQueryImage accepts either a multipart upload ("image" field) or a
// ?sample=NAME query referring to a file in test_set/.
func readQueryImage(r *http.Request, root string) ([]byte, error) {
	if err := r.ParseMultipartForm(32 << 20); err == nil {
		if f, _, err := r.FormFile("image"); err == nil {
			defer f.Close()
			return io.ReadAll(f)
		}
	}
	if sample := r.URL.Query().Get("sample"); sample != "" {
		name := filepath.Base(sample)
		return os.ReadFile(filepath.Join(root, "test_set", name))
	}
	return nil, nil
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func decodePNGBytes(b []byte) (*image.NRGBA, error) {
	img, err := png.Decode(bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	bd := img.Bounds()
	out := image.NewNRGBA(image.Rect(0, 0, bd.Dx(), bd.Dy()))
	for y := 0; y < bd.Dy(); y++ {
		for x := 0; x < bd.Dx(); x++ {
			out.Set(x, y, img.At(bd.Min.X+x, bd.Min.Y+y))
		}
	}
	return out, nil
}

// renderSpritePng renders the segmented sprite as green-on-black PNG (same
// convention as dumpSegmentation) and returns it base64-encoded so the UI can
// inline it without a second round-trip.
func renderSpritePng(img *image.NRGBA, px []Px) string {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	mask := make([]bool, w*h)
	for _, p := range px {
		if p.Y >= 0 && p.Y < h && p.X >= 0 && p.X < w {
			mask[p.Y*w+p.X] = true
		}
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := (y*w + x) * 4
			if mask[y*w+x] {
				out.Pix[i], out.Pix[i+1], out.Pix[i+2], out.Pix[i+3] = 0, 255, 0, 255
			} else {
				out.Pix[i], out.Pix[i+1], out.Pix[i+2], out.Pix[i+3] = 0, 0, 0, 255
			}
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, out); err != nil {
		return ""
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

// runServer starts the HTTP server. addr defaults to :8080.
func runServer(root string, args []string) {
	addr := ":8080"
	if len(args) > 0 {
		addr = args[0]
	}
	srv := newServer(root)
	log.Printf("image-search web server on http://localhost%s  (root=%s)", addr, root)
	log.Fatal(http.ListenAndServe(addr, srv.routes()))
}
