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
	"io"
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
	"go-image-search/internal/sczl"
	"go-image-search/internal/segment"
)

//go:embed assets
var embeddedAssets embed.FS

// Options 服务器配置。
type Options struct {
	IndexPath string // 默认索引文件路径
	Root      string // 图像库根目录，用于浏览/缩略图
	Writer    io.Writer // 日志输出，nil 时默认 os.Stdout
}

// Server Web 服务。
type Server struct {
	opts   Options
	logger *log.Logger

	mu        sync.Mutex
	index     *index.Index   // 区域感知哈希索引
	sczlIndex *sczl.Index    // SCZL 形状索引（NCC+HOG）
	algorithm string         // "region" 或 "sczl"，当前激活的检索策略
	build     *buildJob
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
	w := opts.Writer
	if w == nil {
		w = os.Stdout
	}
	return &Server{
		opts:      opts,
		logger:    log.New(w, "[web] ", log.LstdFlags),
		algorithm: "sczl",
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
	mux.HandleFunc("/api/algorithm", s.handleAlgorithm)
	mux.HandleFunc("/api/build", s.handleBuild)
	mux.HandleFunc("/api/build/status", s.handleBuildStatus)
	mux.HandleFunc("/api/load", s.handleLoad)
	mux.HandleFunc("/api/query", s.handleQuery)
	mux.HandleFunc("/api/images", s.handleImages)
	mux.HandleFunc("/api/image", s.handleImage)
	mux.HandleFunc("/api/segments", s.handleSegments)
	mux.HandleFunc("/api/segments.png", s.handleSegmentsPNG)
	mux.HandleFunc("/api/segments.map.png", s.handleSegmentsMapPNG)
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

// processRegions 由 segment.RunDefault（统一入口）替代：像素划分 → 相似合并 →
// 引力聚合 → 辅助区域构建，返回（res, partition, aux）。

// hashImage 对图像执行区域划分（默认流水线），并将区域产出转换为待入库的哈希区域。
func hashImage(img image.Image, cfg segment.Config, mergeCfg segment.MergeConfig, grav segment.GravityConfig) ([]index.RegionHash, error) {
	res, out, err := segment.RunDefault(img, cfg, mergeCfg, grav)
	if err != nil {
		return nil, err
	}
	return index.RegionHashes(res, out), nil
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
	if s.algorithm == "sczl" && s.sczlIndex != nil {
		images = len(s.sczlIndex.Images)
		regions = s.sczlIndex.Len()
	} else if s.index != nil {
		images = len(s.index.Images)
		regions = s.index.Len()
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"indexPath":  s.opts.IndexPath,
		"root":       s.opts.Root,
		"loaded":     (s.algorithm == "sczl" && s.sczlIndex != nil) || (s.algorithm == "region" && s.index != nil),
		"images":     images,
		"regions":    regions,
		"algorithm":  s.algorithm,
		"sczlLoaded": s.sczlIndex != nil,
		"build":      s.build,
	})
}

func (s *Server) handleAlgorithm(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		s.mu.Lock()
		alg := s.algorithm
		s.mu.Unlock()
		writeJSON(w, http.StatusOK, map[string]string{"algorithm": alg})
		return
	}
	var req struct {
		Algorithm string `json:"algorithm"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	alg := strings.ToLower(strings.TrimSpace(req.Algorithm))
	if alg != "region" && alg != "sczl" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "algorithm 必须是 region 或 sczl"})
		return
	}
	s.mu.Lock()
	s.algorithm = alg
	s.mu.Unlock()
	s.logger.Printf("检索策略切换为: %s", alg)
	writeJSON(w, http.StatusOK, map[string]string{"algorithm": alg})
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
	alg := s.algorithm
	job := &buildJob{
		ID:      fmt.Sprintf("build-%d", time.Now().UnixNano()),
		State:   "running",
		Started: time.Now(),
		Index:   req.Out,
	}
	s.build = job
	s.mu.Unlock()

	if alg == "sczl" {
		go s.runBuildSczl(job, req)
	} else {
		go s.runBuild(job, req)
	}
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
	added := ix.BuildParallel(files, 0, func(f string) (string, []index.RegionHash, error) {
		img, err := imageproc.Load(f)
		if err != nil {
			s.logger.Printf("跳过 %s: %v", f, err)
			return "", nil, err
		}
		id := strings.ReplaceAll(f, "\\", "/")
		// 原图 + 骨架图分别入索引，让查询的任一衍生图都能命中对应表示。
		// 索引阶段也启用引力聚合：区域过多时聚合出辅助组合区域一并入索引，
		// 与查询阶段的辅助查询区域对应，提升碎片化图像的召回。
		var regions []index.RegionHash
		for _, v := range imageproc.QueryVariants(img) {
			hashes, err := hashImage(v, cfg, mc, segment.DefaultGravityConfig())
			if err != nil {
				continue
			}
			regions = append(regions, hashes...)
		}
		return id, regions, nil
	}, func(p index.BuildProgress) {
		s.mu.Lock()
		job.Current = p.File
		job.Done = p.Done
		s.mu.Unlock()
	})

	if err := ix.Save(req.Out); err != nil {
		s.failBuild(job, err)
		return
	}

	s.mu.Lock()
	s.index = ix
	s.opts.IndexPath = req.Out
	job.State = "done"
	job.Images = added
	job.Regions = ix.Len()
	job.Message = "构建完成"
	s.mu.Unlock()
	s.logger.Printf("索引构建完成: %d 图像, %d 区域 -> %s", added, ix.Len(), req.Out)
}

func (s *Server) failBuild(job *buildJob, err error) {
	s.mu.Lock()
	job.State = "error"
	job.Error = err.Error()
	s.mu.Unlock()
	s.logger.Printf("索引构建失败: %v", err)
}

// runBuildSczl 用 SCZL 算法（NCC+HOG 形状描述子）构建索引。
func (s *Server) runBuildSczl(job *buildJob, req buildRequest) {
	files, err := imageproc.LoadSupported(req.Dir)
	if err != nil {
		s.failBuild(job, err)
		return
	}
	if len(files) == 0 {
		s.failBuild(job, fmt.Errorf("目录 %s 中无支持的图片", req.Dir))
		return
	}

	out := req.Out
	if out == "" || out == s.opts.IndexPath {
		out = s.opts.IndexPath + ".sczl"
	}

	ix := sczl.New()
	job.Total = len(files)
	added := ix.BuildParallel(files, 0, func(f string) (sczl.Descriptor, error) {
		img, err := imageproc.Load(f)
		if err != nil {
			s.logger.Printf("跳过 %s: %v", f, err)
			return sczl.Descriptor{}, err
		}
		return sczl.Extract(img), nil
	}, func(p sczl.BuildProgress) {
		s.mu.Lock()
		job.Current = p.File
		job.Done = p.Done
		s.mu.Unlock()
	})

	if err := ix.Save(out); err != nil {
		s.failBuild(job, err)
		return
	}

	s.mu.Lock()
	s.sczlIndex = ix
	job.State = "done"
	job.Images = added
	job.Regions = ix.Len()
	job.Index = out
	job.Message = "SCZL 构建完成"
	s.mu.Unlock()
	s.logger.Printf("SCZL 索引构建完成: %d 图像, %d 描述子 -> %s", added, ix.Len(), out)
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
	s.mu.Lock()
	alg := s.algorithm
	s.mu.Unlock()
	if alg == "sczl" {
		if p == "" {
			p = s.opts.IndexPath + ".sczl"
		}
		ix, err := sczl.Load(p)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		s.mu.Lock()
		s.sczlIndex = ix
		s.mu.Unlock()
		s.logger.Printf("已加载 SCZL 索引: %s", p)
		writeJSON(w, http.StatusOK, map[string]any{"images": len(ix.Images), "regions": ix.Len()})
		return
	}
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
	alg := s.algorithm
	ix := s.index
	sczlIx := s.sczlIndex
	s.mu.Unlock()

	if alg == "sczl" {
		if sczlIx == nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "尚未加载 SCZL 索引，请先构建或加载索引"})
			return
		}
		s.handleQuerySczl(w, r, sczlIx)
		return
	}
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
	grav := gravityConfigFromQuery(r.Form)
	// 查询阶段无条件追加整图辅助区域，提升整图级别的召回
	grav.CombineAlways = true
	// 对查询图像生成 原图 + 骨架图 分别检索；若检测到统一背景色与附属文字带，
	// 再追加归一化裁剪内容的整图辅助区域（白底+主体，与图库透明底图标可比）。
	// 任一衍生图的任一区域与索引图像相似即认为该图像相似。
	querySets := make([][]index.QueryRegion, 0, 3)
	for _, v := range imageproc.QueryVariants(img) {
		hashes, err := hashImage(v, cfg, segment.DefaultMergeConfig(), grav)
		if err != nil {
			continue
		}
		qs := make([]index.QueryRegion, 0, len(hashes))
		for _, h := range hashes {
			qs = append(qs, index.QueryRegion{
				Hash: h.Hash, Shape: h.Shape, Area: h.Area, Color: h.Color,
				NX: h.NX, NY: h.NY, Fill: h.Fill, Aspect: h.Aspect, Global: h.Global,
			})
		}
		querySets = append(querySets, qs)
	}
	if whole := imageproc.QueryNormalizedWholeHashes(img, cfg, segment.DefaultMergeConfig(), grav); len(whole) > 0 {
		qs := make([]index.QueryRegion, 0, len(whole))
		for _, h := range whole {
			qs = append(qs, index.QueryRegion{
				Hash: h.Hash, Shape: h.Shape, Area: h.Area, Color: h.Color,
				NX: h.NX, NY: h.NY, Fill: h.Fill, Aspect: h.Aspect, Global: h.Global,
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

// handleQuerySczl 用 SCZL 算法检索。
func (s *Server) handleQuerySczl(w http.ResponseWriter, r *http.Request, ix *sczl.Index) {
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
	q := sczl.Extract(img)
	if !q.Valid {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "无法提取前景"})
		return
	}
	top := atoiDefault(r.FormValue("top"), 5)
	if top <= 0 {
		top = 5
	}
	matches := ix.Query(q, sczl.DefaultOptions())
	if len(matches) > top {
		matches = matches[:top]
	}
	resp := struct {
		Regions int         `json:"regions"`
		Matches []matchJSON `json:"matches"`
	}{Regions: 1}
	for _, m := range matches {
		resp.Matches = append(resp.Matches, matchJSON{
			ImageID: m.ImageID,
			Score:   round4(m.Score),
			Cover:   1.0,
			Count:   1,
		})
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleImages(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	alg := s.algorithm
	ix := s.index
	sczlIx := s.sczlIndex
	s.mu.Unlock()
	var images map[string]bool
	if alg == "sczl" && sczlIx != nil {
		images = sczlIx.Images
	} else if ix != nil {
		images = ix.Images
	}
	if images == nil {
		writeJSON(w, http.StatusOK, []string{})
		return
	}
	paths := make([]string, 0, len(images))
	for p := range images {
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
	if _, ok := mimeByExt[ext]; !ok {
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
	Whole bool   `json:"whole"` // 是否为"整图"辅助区域（绿框）
}

func (s *Server) handleSegments(w http.ResponseWriter, r *http.Request) {
	img, ok := s.loadImageForSegments(w, r)
	if !ok {
		return
	}
	q := r.URL.Query()
	cfg := segConfigFromQuery(q)
	grav := gravityConfigFromQuery(q)
	res, out, err := segment.RunDefault(img, cfg, segment.DefaultMergeConfig(), grav)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	all := out.All()
	regions := make([]regionJSON, 0, len(all))
	for _, reg := range all {
		regions = append(regions, regionJSON{
			ID:    reg.ID,
			Area:  reg.Area,
			BBox:  [4]int{reg.BBox.Min.X, reg.BBox.Min.Y, reg.BBox.Max.X, reg.BBox.Max.Y},
			Color: fmt.Sprintf("#%02x%02x%02x", reg.MeanColor.R, reg.MeanColor.G, reg.MeanColor.B),
			Whole: reg.Whole,
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
	q := r.URL.Query()
	cfg := segConfigFromQuery(q)
	grav := gravityConfigFromQuery(q)
	res, out, err := segment.RunDefault(img, cfg, segment.DefaultMergeConfig(), grav)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	partition := out.Partition
	var vis *image.RGBA
	if q.Get("mode") == "fill" {
		vis = visualizeFill(res, partition, img)
	} else {
		vis = visualize(partition, out.Aux, img)
	}
	data, err := imageproc.EncodePNG(vis)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("Cache-Control", "no-store")
	w.Write(data)
}

func (s *Server) handleSegmentsMapPNG(w http.ResponseWriter, r *http.Request) {
	img, ok := s.loadImageForSegments(w, r)
	if !ok {
		return
	}
	q := r.URL.Query()
	cfg := segConfigFromQuery(q)
	grav := gravityConfigFromQuery(q)
	res, out, err := segment.RunDefault(img, cfg, segment.DefaultMergeConfig(), grav)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	vis := visualizeLabels(res, out.Partition)
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

// visualize 在图像上叠加区域边界框（检索实际使用的区域，含引力辅助组合区域）。
// 划分区域用红色标注，引力辅助组合区域用蓝色标注，合并"整图"辅助区域用绿色标注。
func visualize(partition, aux []segment.MergedRegion, src image.Image) *image.RGBA {
	b := src.Bounds()
	dst := image.NewRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.Set(x, y, src.At(x, y))
		}
	}
	for _, reg := range partition {
		drawBox(dst, reg.BBox, color.RGBA{255, 0, 0, 255})
	}
	for _, reg := range aux {
		c := color.RGBA{0, 0, 255, 255}
		if reg.Whole {
			c = color.RGBA{0, 255, 0, 255}
		}
		drawBox(dst, reg.BBox, c)
	}
	return dst
}

func drawBox(dst *image.RGBA, box image.Rectangle, c color.Color) {
	for x := box.Min.X; x < box.Max.X; x++ {
		dst.Set(x, box.Min.Y, c)
		dst.Set(x, box.Max.Y-1, c)
	}
	for y := box.Min.Y; y < box.Max.Y; y++ {
		dst.Set(box.Min.X, y, c)
		dst.Set(box.Max.X-1, y, c)
	}
}

// buildRawToMerged 构建原始区域 ID → 合并区域 ID 的映射（基于 partition 的 Members）。
func buildRawToMerged(partition []segment.MergedRegion) map[int32]int {
	m := make(map[int32]int, len(partition))
	for _, mr := range partition {
		for _, mid := range mr.Members {
			m[int32(mid)] = mr.ID
		}
	}
	return m
}

// visualizeFill 用每个区域（合并后）的平均色填充该区域像素，背景像素保留原图，
// 直观展示实际参与检索的空间划分。
func visualizeFill(res *segment.Result, partition []segment.MergedRegion, src image.Image) *image.RGBA {
	b := src.Bounds()
	dst := image.NewRGBA(b)
	rawToMerged := buildRawToMerged(partition)
	colors := make(map[int]color.RGBA, len(partition))
	for _, mr := range partition {
		colors[mr.ID] = mr.MeanColor
	}
	for y := 0; y < res.Height; y++ {
		for x := 0; x < res.Width; x++ {
			px, py := b.Min.X+x, b.Min.Y+y
			label := res.Labels[y*res.Width+x]
			if label > 0 {
				if mergedID, ok := rawToMerged[label]; ok {
					if c, ok2 := colors[mergedID]; ok2 {
						dst.Set(px, py, c)
						continue
					}
				}
			}
			dst.Set(px, py, src.At(px, py))
		}
	}
	return dst
}

// visualizeLabels 生成标签图：每个区域像素以唯一颜色编码其合并区域 ID
// （R=高8位、G=中8位、B=低8位），背景像素为 (0,0,0)。
// 客户端可读取像素反查区域 ID，实现悬停展示等交互。
func visualizeLabels(res *segment.Result, partition []segment.MergedRegion) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, res.Width, res.Height))
	rawToMerged := buildRawToMerged(partition)
	for i, label := range res.Labels {
		off := i * 4
		if label > 0 {
			mergedID := uint32(label)
			if id, ok := rawToMerged[label]; ok {
				mergedID = uint32(id)
			}
			dst.Pix[off+0] = uint8((mergedID >> 16) & 0xff)
			dst.Pix[off+1] = uint8((mergedID >> 8) & 0xff)
			dst.Pix[off+2] = uint8(mergedID & 0xff)
		} else {
			dst.Pix[off+0] = 0
			dst.Pix[off+1] = 0
			dst.Pix[off+2] = 0
		}
		dst.Pix[off+3] = 255
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

// gravityConfigFromQuery 从表单参数构建引力聚合配置（检索阶段辅助检索）。
func gravityConfigFromQuery(q url.Values) segment.GravityConfig {
	gc := segment.DefaultGravityConfig()
	if v := q.Get("gravity"); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			gc.Enabled = b
		}
	}
	if v := q.Get("gravityMin"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			gc.MinRegions = n
		}
	}
	if v := q.Get("gravityTrigger"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			gc.TriggerCount = n
		}
	}
	if v := q.Get("gravityAdditive"); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			gc.Additive = b
		}
	}
	if v := q.Get("gravityCombineFew"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			gc.CombineFew = n
		}
	}
	if v := q.Get("gravityFrameRatio"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 {
			gc.FrameRatio = f
		}
	}
	return gc
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
