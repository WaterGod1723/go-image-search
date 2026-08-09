// Package web 提供 iOS 风格图片搜索的 Web 界面与 API。
// 底层引擎为神经网络排序（image-search-test 引擎），见 internal/nnengine。
package web

import (
	"embed"
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"go-image-search/internal/nnengine"
)

//go:embed assets
var embeddedAssets embed.FS

// Options 服务器配置。
type Options struct {
	IndexPath   string // 默认索引（引擎缓存）文件路径
	Root        string // 图像库根目录，用于浏览/缩略图
	Writer      io.Writer
	WeightsPath string // 训练好的神经网络权重文件（可为空）
}

// Server Web 服务器。
type Server struct {
	opts   Options
	logger *log.Logger

	mu      sync.Mutex
	engine  *nnengine.Engine
	build   *buildJob
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

// New 创建一个新的 Web 服务器（底层为神经网络引擎）。
func New(opts Options) *Server {
	if opts.Root == "" {
		opts.Root = "."
	}
	w := opts.Writer
	if w == nil {
		w = os.Stdout
	}
	return &Server{
		opts:   opts,
		logger: log.New(w, "[web] ", log.LstdFlags),
		engine: nnengine.New(opts.WeightsPath),
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

// LoadIndex 从默认索引路径加载引擎缓存（若文件存在）。
func (s *Server) LoadIndex() error {
	if err := s.engine.LoadIndex(s.opts.IndexPath); err != nil {
		return err
	}
	s.logger.Printf("已加载索引: %d 张 (%s)", s.engine.Len(), s.opts.IndexPath)
	return nil
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
	images := 0
	if s.engine != nil {
		images = s.engine.Len()
	}
	loaded := images > 0
	build := s.build
	s.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{
		"indexPath":  s.opts.IndexPath,
		"root":       s.opts.Root,
		"loaded":     loaded,
		"images":     images,
		"regions":    images,
		"algorithm":  "nn",
		"sczlLoaded": loaded,
		"build":      build,
	})
}

func (s *Server) handleAlgorithm(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		writeJSON(w, http.StatusOK, map[string]string{"algorithm": "nn"})
		return
	}
	var req struct {
		Algorithm string `json:"algorithm"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	_ = req
	writeJSON(w, http.StatusOK, map[string]string{"algorithm": "nn"})
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

	go func() {
		if err := s.engine.BuildIndex(req.Dir, req.Out); err != nil {
			s.failBuild(job, err)
			return
		}
		s.mu.Lock()
		s.opts.IndexPath = req.Out
		job.State = "done"
		job.Done = s.engine.Len()
		job.Total = s.engine.Len()
		job.Images = s.engine.Len()
		job.Regions = s.engine.Len()
		job.Message = "构建完成"
		s.mu.Unlock()
		s.logger.Printf("NN 索引构建完成: %d 张 -> %s", s.engine.Len(), req.Out)
	}()
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
	if err := s.engine.LoadIndex(p); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	s.mu.Lock()
	s.opts.IndexPath = p
	s.mu.Unlock()
	s.logger.Printf("已加载索引: %s (%d 张)", p, s.engine.Len())
	writeJSON(w, http.StatusOK, map[string]any{"images": s.engine.Len(), "regions": s.engine.Len()})
}

func (s *Server) handleQuery(w http.ResponseWriter, r *http.Request) {
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
	top := atoiDefault(r.FormValue("top"), 5)
	res, err := s.queryImage(img, top)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func (s *Server) handleImages(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	names := s.engine.Names()
	s.mu.Unlock()
	sort.Strings(names)
	writeJSON(w, http.StatusOK, names)
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

// handleSegments / handleSegmentsPNG / handleSegmentsMapPNG：旧"区域划分"调试
// 功能已随算法引擎替换移除，返回明确错误以提示前端。
func (s *Server) handleSegments(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusBadRequest, map[string]string{"error": "区域划分调试已随旧算法移除，当前引擎为神经网络排序"})
}

func (s *Server) handleSegmentsPNG(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusBadRequest, map[string]string{"error": "区域划分调试已随旧算法移除，当前引擎为神经网络排序"})
}

func (s *Server) handleSegmentsMapPNG(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusBadRequest, map[string]string{"error": "区域划分调试已随旧算法移除，当前引擎为神经网络排序"})
}

func atoiDefault(s string, def int) int {
	if v, err := strconv.Atoi(strings.TrimSpace(s)); err == nil {
		return v
	}
	return def
}

func round4(v float64) float64 {
	return float64(int(v*10000+0.5)) / 10000
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}
