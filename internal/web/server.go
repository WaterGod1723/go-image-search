// Package web 提供 iOS 风格图片搜索的 Web 界面与 API。
package web

import (
	"embed"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io/fs"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"go-image-search/internal/imageproc"
	"go-image-search/internal/index"
	"go-image-search/internal/phash"
	"go-image-search/internal/segment"
)

//go:embed assets
var embeddedAssets embed.FS

// Options 服务器配置。
type Options struct {
	IndexPath string // 默认索引文件路径
	Root      string // 图像库根目录，用于浏览/缩略图
}

// Server Web 服务。
type Server struct {
	opts   Options
	logger *log.Logger

	mu    sync.Mutex
	index *index.Index
	build *buildJob
}

// buildJob 后台构建任务的进度状态。
type buildJob struct {
	ID      string    `json:"id"`
	State   string    `json:"state"` // running / done / error
	Message string    `json:"message,omitempty"`
	Error   string    `json:"error,omitempty"`
	Current string    `json:"current,omitempty"`
	Done    int       `json:"done"`
	Total   int       `json:"total"`
	Images  int       `json:"images"`
	Regions int       `json:"regions"`
	Index   string    `json:"index"`
	Started time.Time `json:"started"`
}

// buildRequest 构建索引的请求参数。
type buildRequest struct {
	Dir             string  `json:"dir"`
	Out             string  `json:"out"`
	ThresholdPct    float64 `json:"thresholdPct"`
	ThresholdFactor float64 `json:"thresholdFactor"`
	MinAreaRatio    float64 `json:"minAreaRatio"`
	MedianFilterK   int     `json:"medianFilterK"`
	Connectivity    int     `json:"connectivity"`
	MergeHashDist   int     `json:"mergeHashDist"`
	MergeColorDist  float64 `json:"mergeColorDist"`
	NoMerge         bool    `json:"noMerge"`
}

// New 创建一个新的 Web 服务器。
func New(opts Options) *Server {
	if opts.Root == "" {
		opts.Root = "."
	}
	return &Server{
		opts:   opts,
		logger: log.New(os.Stdout, "[web] ", log.LstdFlags),
	}
}

// Handler 返回 HTTP 路由。
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	sub, err := fs.Sub(embeddedAssets, "assets")
	if err != nil {
		panic(err)
	}
	mux.Handle("/assets/", http.StripPrefix("/assets/", http.FileServer(http.FS(sub))))
	mux.HandleFunc("/", s.handleIndex)
	mux.HandleFunc("/api/status", s.handleStatus)
	mux.HandleFunc("/api/build", s.handleBuild)
	mux.HandleFunc("/api/build/status", s.handleBuildStatus)
	mux.HandleFunc("/api/load", s.handleLoad)
	mux.HandleFunc("/api/query", s.handleQuery)
	mux.HandleFunc("/api/images", s.handleImages)
	mux.HandleFunc("/api/image", s.handleImage)
	mux.HandleFunc("/api/segments", s.handleSegments)
	mux.HandleFunc("/api/segments.png", s.handleSegmentsPNG)
	return mux
}

// LoadIndex 从默认索引路径加载索引（若文件存在）。
func (s *Server) LoadIndex() error {
	ix, err := index.Load(s.opts.IndexPath)
	if err != nil {
		return err
	}
	s.mu.Lock()
	s.index = ix
	s.mu.Unlock()
	s.logger.Printf("已加载索引: %d 图像, %d 区域", len(ix.Images), ix.Len())
	return nil
}

// hashImage 对图像做区域划分、感知哈希，并将相似相邻区域合并为组合区域。
func hashImage(img image.Image, cfg segment.Config, merge segment.MergeConfig) ([]index.RegionHash, error) {
	res, err := segment.Segment(img, cfg)
	if err != nil {
		return nil, err
	}
	infos := make([]segment.RegionInfo, 0, len(res.Regions))
	for _, reg := range res.Regions {
		crop := res.Crop(img, reg.ID)
		if crop == nil {
			continue
		}
		shape := phash.Hash(imageproc.StructuralMask(crop))
		infos = append(infos, segment.RegionInfo{
			ID: reg.ID, Hash: phash.Hash(crop), Shape: shape,
			Area: reg.Area, Color: reg.MeanColor, BBox: reg.BBox,
		})
	}
	merged := segment.MergeSimilar(img, res, infos, merge)
	hashes := make([]index.RegionHash, 0, len(merged))
	for _, m := range merged {
		bw, bh := m.BBox.Dx(), m.BBox.Dy()
		fill, aspect := 0.0, 0.0
		if bw > 0 && bh > 0 {
			fill = float64(m.Area) / float64(bw*bh)
			aspect = float64(bw) / float64(bh)
		}
		nx, ny := 0.0, 0.0
		if res.Width > 0 {
			nx = float64(m.BBox.Min.X+m.BBox.Max.X) / (2 * float64(res.Width))
		}
		if res.Height > 0 {
			ny = float64(m.BBox.Min.Y+m.BBox.Max.Y) / (2 * float64(res.Height))
		}
		hashes = append(hashes, index.RegionHash{
			RegionID: m.ID,
			Hash:     m.Hash,
			Shape:    m.Shape,
			Area:     m.Area,
			BBox:     m.BBox,
			Color:    m.MeanColor,
			NX:       nx,
			NY:       ny,
			Fill:     fill,
			Aspect:   aspect,
		})
	}
	return hashes, nil
}

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	data, err := embeddedAssets.ReadFile("assets/index.html")
	if err != nil {
		http.Error(w, "index.html 缺失", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(data)
}

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	images, regions := 0, 0
	if s.index != nil {
		images = len(s.index.Images)
		regions = s.index.Len()
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"indexPath": s.opts.IndexPath,
		"root":      s.opts.Root,
		"loaded":    s.index != nil,
		"images":    images,
		"regions":   regions,
		"build":     s.build,
	})
}

func (s *Server) handleBuild(w http.ResponseWriter, r *http.Request) {
	var req buildRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	if req.Dir == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "缺少 dir（图像库目录）"})
		return
	}
	if req.Out == "" {
		req.Out = s.opts.IndexPath
	}

	s.mu.Lock()
	if s.build != nil && s.build.State == "running" {
		s.mu.Unlock()
		writeJSON(w, http.StatusConflict, map[string]string{"error": "已有构建任务正在运行"})
		return
	}
	job := &buildJob{
		ID:      fmt.Sprintf("build-%d", time.Now().UnixNano()),
		State:   "running",
		Started: time.Now(),
		Index:   req.Out,
	}
	s.build = job
	s.mu.Unlock()

	go s.runBuild(job, req)
	writeJSON(w, http.StatusAccepted, job)
}

func (s *Server) handleBuildStatus(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.build == nil {
		writeJSON(w, http.StatusOK, map[string]any{"state": "idle"})
		return
	}
	writeJSON(w, http.StatusOK, s.build)
}

func (s *Server) runBuild(job *buildJob, req buildRequest) {
	cfg := segment.DefaultConfig()
	if req.ThresholdPct > 0 {
		cfg.ThresholdPct = req.ThresholdPct
	}
	if req.ThresholdFactor > 0 {
		cfg.ThresholdFactor = req.ThresholdFactor
	}
	if req.MinAreaRatio > 0 {
		cfg.MinAreaRatio = req.MinAreaRatio
	}
	if req.MedianFilterK > 0 {
		cfg.MedianFilterK = req.MedianFilterK
	}
	if req.Connectivity != 0 {
		cfg.Connectivity = req.Connectivity
	}

	mc := segment.DefaultMergeConfig()
	if req.MergeHashDist > 0 {
		mc.HashDist = req.MergeHashDist
	}
	if req.MergeColorDist > 0 {
		mc.ColorDist = req.MergeColorDist
	}
	if req.NoMerge {
		mc.Enabled = false
	}

	files, err := imageproc.LoadSupported(req.Dir)
	if err != nil {
		s.failBuild(job, err)
		return
	}
	if len(files) == 0 {
		s.failBuild(job, fmt.Errorf("目录 %s 中无支持的图片", req.Dir))
		return
	}

	ix := index.New()
	job.Total = len(files)
	for i, f := range files {
			job.Current = f
			job.Done = i + 1
			img, err := imageproc.Load(f)
			if err != nil {
				s.logger.Printf("跳过 %s: %v", f, err)
				continue
			}
			id := strings.ReplaceAll(f, "\\", "/")
			// 原图 + 骨架图分别入索引，让查询的任一衍生图都能命中对应表示。
			for _, v := range imageproc.QueryVariants(img) {
				hashes, err := hashImage(v, cfg, mc)
				if err != nil {
					continue
				}
				ix.AddImage(id, hashes)
			}
		}

	if err := ix.Save(req.Out); err != nil {
		s.failBuild(job, err)
		return
	}

	s.mu.Lock()
	s.index = ix
	s.opts.IndexPath = req.Out
	job.State = "done"
	job.Images = len(ix.Images)
	job.Regions = ix.Len()
	job.Message = "构建完成"
	s.mu.Unlock()
	s.logger.Printf("索引构建完成: %d 图像, %d 区域 -> %s", len(ix.Images), ix.Len(), req.Out)
}

func (s *Server) failBuild(job *buildJob, err error) {
	s.mu.Lock()
	job.State = "error"
	job.Error = err.Error()
	s.mu.Unlock()
	s.logger.Printf("索引构建失败: %v", err)
}

func (s *Server) handleLoad(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Index string `json:"index"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	p := strings.TrimSpace(req.Index)
	if p == "" {
		p = s.opts.IndexPath
	}
	ix, err := index.Load(p)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	s.mu.Lock()
	s.index = ix
	s.opts.IndexPath = p
	s.mu.Unlock()
	s.logger.Printf("已加载索引: %s", p)
	writeJSON(w, http.StatusOK, map[string]any{"images": len(ix.Images), "regions": ix.Len()})
}

type regionMatchJSON struct {
	RegionID int    `json:"regionId"`
	Hash     string `json:"hash"`
	Dist     int    `json:"dist"`
	Area     int    `json:"area"`
	Color    string `json:"color"`
}

type matchJSON struct {
	ImageID string            `json:"imageId"`
	Score   float64           `json:"score"`
	Cover   float64           `json:"cover"`
	Count   int               `json:"count"`
	Regions []regionMatchJSON `json:"regions"`
}

func (s *Server) handleQuery(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	ix := s.index
	s.mu.Unlock()
	if ix == nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "尚未加载索引，请先构建或加载索引"})
		return
	}
	if err := r.ParseMultipartForm(64 << 20); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	file, _, err := r.FormFile("image")
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "缺少查询图像"})
		return
	}
	defer file.Close()
	img, _, err := image.Decode(file)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "无法解析图像: " + err.Error()})
		return
	}

	cfg := segment.DefaultConfig()
	// 对查询图像生成原图 + 骨架图两张衍生图分别检索，
	// 任一衍生图与索引图像相似即认为该图像相似。
	querySets := make([][]index.QueryRegion, 0, 2)
	for _, v := range imageproc.QueryVariants(img) {
		hashes, err := hashImage(v, cfg, segment.DefaultMergeConfig())
		if err != nil {
			continue
		}
		qs := make([]index.QueryRegion, 0, len(hashes))
		for _, h := range hashes {
			qs = append(qs, index.QueryRegion{
				Hash: h.Hash, Shape: h.Shape, Area: h.Area, Color: h.Color,
				NX: h.NX, NY: h.NY, Fill: h.Fill, Aspect: h.Aspect,
			})
		}
		querySets = append(querySets, qs)
	}
	top := atoiDefault(r.FormValue("top"), 5)
	if top <= 0 {
		top = 5
	}
	maxDist := atoiDefault(r.FormValue("maxdist"), 12)
	colorWeight := parseColorWeight(r.FormValue("colorWeight"))

	matches := ix.SearchMulti(querySets, index.SearchOptions{MaxDist: maxDist, ColorWeight: colorWeight})
	if len(matches) > top {
		matches = matches[:top]
	}

	totalRegions := 0
	for _, qs := range querySets {
		totalRegions += len(qs)
	}
	resp := struct {
		Regions int         `json:"regions"`
		Matches []matchJSON `json:"matches"`
	}{Regions: totalRegions}
	for _, m := range matches {
		mm := matchJSON{
			ImageID: m.ImageID,
			Score:   round4(m.Score),
			Cover:   round4(m.CoverRatio),
			Count:   len(m.Matches),
		}
		for _, rm := range m.Matches {
			mm.Regions = append(mm.Regions, regionMatchJSON{
				RegionID: rm.Entry.RegionID,
				Hash:     fmt.Sprintf("%016x", rm.Entry.Hash),
				Dist:     rm.Dist,
				Area:     rm.Entry.Area,
				Color:    fmt.Sprintf("#%02x%02x%02x", rm.Entry.Color.R, rm.Entry.Color.G, rm.Entry.Color.B),
			})
		}
		resp.Matches = append(resp.Matches, mm)
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleImages(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	ix := s.index
	s.mu.Unlock()
	if ix == nil {
		writeJSON(w, http.StatusOK, []string{})
		return
	}
	paths := make([]string, 0, len(ix.Images))
	for p := range ix.Images {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	writeJSON(w, http.StatusOK, paths)
}

func (s *Server) handleImage(w http.ResponseWriter, r *http.Request) {
	p := r.URL.Query().Get("path")
	if p == "" {
		http.Error(w, "缺少 path", http.StatusBadRequest)
		return
	}
	ext := strings.ToLower(filepath.Ext(p))
	switch ext {
	case ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp":
	default:
		http.Error(w, "不支持的文件类型", http.StatusBadRequest)
		return
	}
	http.ServeFile(w, r, filepath.FromSlash(p))
}

type regionJSON struct {
	ID    int    `json:"id"`
	Area  int    `json:"area"`
	BBox  [4]int `json:"bbox"`
	Color string `json:"color"`
}

func (s *Server) handleSegments(w http.ResponseWriter, r *http.Request) {
	img, ok := s.loadImageForSegments(w, r)
	if !ok {
		return
	}
	cfg := segConfigFromQuery(r.URL.Query())
	res, err := segment.Segment(img, cfg)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	regions := make([]regionJSON, 0, len(res.Regions))
	for _, reg := range res.Regions {
		regions = append(regions, regionJSON{
			ID:    reg.ID,
			Area:  reg.Area,
			BBox:  [4]int{reg.BBox.Min.X, reg.BBox.Min.Y, reg.BBox.Max.X, reg.BBox.Max.Y},
			Color: fmt.Sprintf("#%02x%02x%02x", reg.MeanColor.R, reg.MeanColor.G, reg.MeanColor.B),
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"width":   res.Width,
		"height":  res.Height,
		"regions": regions,
	})
}

func (s *Server) handleSegmentsPNG(w http.ResponseWriter, r *http.Request) {
	img, ok := s.loadImageForSegments(w, r)
	if !ok {
		return
	}
	cfg := segConfigFromQuery(r.URL.Query())
	res, err := segment.Segment(img, cfg)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	vis := visualize(res, img)
	data, err := imageproc.EncodePNG(vis)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("Cache-Control", "no-store")
	w.Write(data)
}

func (s *Server) loadImageForSegments(w http.ResponseWriter, r *http.Request) (image.Image, bool) {
	p := r.URL.Query().Get("path")
	if p == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "缺少 path"})
		return nil, false
	}
	img, err := imageproc.Load(filepath.FromSlash(p))
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return nil, false
	}
	return img, true
}

// visualize 在图像上叠加区域边界框。
func visualize(res *segment.Result, src image.Image) *image.RGBA {
	b := src.Bounds()
	dst := image.NewRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.Set(x, y, src.At(x, y))
		}
	}
	for _, reg := range res.Regions {
		box := reg.BBox
		for x := box.Min.X; x < box.Max.X; x++ {
			dst.Set(x, box.Min.Y, color.RGBA{255, 0, 0, 255})
			dst.Set(x, box.Max.Y-1, color.RGBA{255, 0, 0, 255})
		}
		for y := box.Min.Y; y < box.Max.Y; y++ {
			dst.Set(box.Min.X, y, color.RGBA{255, 0, 0, 255})
			dst.Set(box.Max.X-1, y, color.RGBA{255, 0, 0, 255})
		}
	}
	return dst
}

func segConfigFromQuery(q url.Values) segment.Config {
	cfg := segment.DefaultConfig()
	if v := q.Get("thresholdPct"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 {
			cfg.ThresholdPct = f
		}
	}
	if v := q.Get("thresholdFactor"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 {
			cfg.ThresholdFactor = f
		}
	}
	if v := q.Get("minAreaRatio"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 {
			cfg.MinAreaRatio = f
		}
	}
	if v := q.Get("median"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.MedianFilterK = n
		}
	}
	if v := q.Get("connectivity"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.Connectivity = n
		}
	}
	return cfg
}

func round4(v float64) float64 {
	return math.Round(v*10000) / 10000
}

func atoiDefault(s string, def int) int {
	if s == "" {
		return def
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return def
	}
	return n
}

func parseColorWeight(s string) float64 {
	if s == "" {
		return 0.1
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0.1
	}
	return f
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}
