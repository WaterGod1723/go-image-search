// Package web 提供 iOS 风格图片搜索的 Web 界面与 API。
// service.go 提供面向桌面 GUI（Wails）的文件/字节级操作方法。
//
// 底层引擎为神经网络排序（image-search-test 引擎：MLP 融合手工特征 +
// sczl 颜色无关专家 + 两阶段粗筛），旧"区域哈希/SCZL 单独"检索策略已替换。
package web

import (
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"go-image-search/internal/imageproc"
)

// StatusInfo 索引与服务器的整体状态。
type StatusInfo struct {
	IndexPath  string        `json:"indexPath"`
	Root       string        `json:"root"`
	Loaded     bool          `json:"loaded"`
	Images     int           `json:"images"`
	Regions    int           `json:"regions"`
	Algorithm  string        `json:"algorithm"`
	SczlLoaded bool          `json:"sczlLoaded"`
	Build      *BuildJobInfo `json:"build"`
}

// BuildJobInfo 后台构建任务的进度状态（向 GUI 暴露的公共视图）。
type BuildJobInfo struct {
	ID      string    `json:"id"`
	State   string    `json:"state"`
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

func (j *buildJob) info() *BuildJobInfo {
	if j == nil {
		return &BuildJobInfo{State: "idle"}
	}
	return &BuildJobInfo{
		ID: j.ID, State: j.State, Message: j.Message, Error: j.Error,
		Current: j.Current, Done: j.Done, Total: j.Total,
		Images: j.Images, Regions: j.Regions, Index: j.Index, Started: j.Started,
	}
}

// BuildParams 构建索引的请求参数（与 HTTP buildRequest 字段一致）。
type BuildParams struct {
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

// LoadResult 加载索引后的统计。
type LoadResult struct {
	Images  int `json:"images"`
	Regions int `json:"regions"`
}

// QueryResult 检索结果。
type QueryResult struct {
	Regions int         `json:"regions"`
	Matches []MatchInfo `json:"matches"`
}

// MatchInfo 单张图像的匹配信息（score 为旋转对齐的掩膜相似度）。
type MatchInfo struct {
	ImageID string        `json:"imageId"`
	Score   float64       `json:"score"`
	Cover   float64       `json:"cover"`
	Count   int           `json:"count"`
	Regions []RegionMatch `json:"regions"`
}

// RegionMatch 单个命中区域（神经网络引擎不再产出区域划分，保留字段以兼容前端）。
type RegionMatch struct {
	RegionID int    `json:"regionId"`
	Hash     string `json:"hash"`
	Dist     int    `json:"dist"`
	Area     int    `json:"area"`
	Color    string `json:"color"`
}

// SegParams 区域划分（调试）的参数（保留以兼容前端，引擎不再使用）。
type SegParams struct {
	ThresholdPct      float64 `json:"thresholdPct"`
	ThresholdFactor   float64 `json:"thresholdFactor"`
	MinAreaRatio      float64 `json:"minAreaRatio"`
	MedianFilterK     int     `json:"medianFilterK"`
	Connectivity      int     `json:"connectivity"`
	Gravity           bool    `json:"gravity"`
	GravityMin        int     `json:"gravityMin"`
	GravityTrigger    int     `json:"gravityTrigger"`
	GravityAdditive   bool    `json:"gravityAdditive"`
	GravityCombineFew int     `json:"gravityCombineFew"`
	GravityFrameRatio float64 `json:"gravityFrameRatio"`
}

// SegResult 区域划分结果。
type SegResult struct {
	Width   int         `json:"width"`
	Height  int         `json:"height"`
	Regions []SegRegion `json:"regions"`
}

// SegRegion 单个划分区域。
type SegRegion struct {
	ID    int    `json:"id"`
	Area  int    `json:"area"`
	BBox  [4]int `json:"bbox"`
	Color string `json:"color"`
	Whole bool   `json:"whole"`
}

// Status 返回服务器与已加载索引的整体状态。
func (s *Server) Status() StatusInfo {
	s.mu.Lock()
	defer s.mu.Unlock()
	var st StatusInfo
	st.IndexPath = s.opts.IndexPath
	st.Root = s.opts.Root
	st.Algorithm = "nn"
	if s.engine != nil && s.engine.Len() > 0 {
		st.Loaded = true
		st.Images = s.engine.Len()
		st.Regions = s.engine.Len()
	}
	st.SczlLoaded = s.engine != nil && s.engine.Len() > 0
	st.Build = s.build.info()
	return st
}

// GetAlgorithm 返回当前检索策略（恒为 nn）。
func (s *Server) GetAlgorithm() string { return "nn" }

// SetAlgorithm 保留以兼容前端；仅接受 "nn"（引擎已固定为神经网络排序）。
func (s *Server) SetAlgorithm(alg string) error {
	a := strings.ToLower(strings.TrimSpace(alg))
	if a != "nn" && a != "sczl" && a != "region" {
		return errors.New("未知算法")
	}
	s.logger.Printf("检索策略固定为: nn（神经网络排序引擎）")
	return nil
}

// Build 启动后台构建索引任务，立即返回任务视图（由 BuildStatus 轮询进度）。
func (s *Server) Build(req BuildParams) (*BuildJobInfo, error) {
	if req.Dir == "" {
		return nil, errors.New("缺少 dir（图像库目录）")
	}
	if req.Out == "" {
		req.Out = s.opts.IndexPath
	}
	s.mu.Lock()
	if s.build != nil && s.build.State == "running" {
		s.mu.Unlock()
		return nil, errors.New("已有构建任务正在运行")
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
		job.State = "done"
		job.Message = "构建完成"
		job.Done = s.engine.Len()
		job.Total = s.engine.Len()
		job.Images = s.engine.Len()
		job.Regions = s.engine.Len()
		s.mu.Unlock()
		s.logger.Printf("NN 索引构建完成: %d 张", s.engine.Len())
	}()
	return job.info(), nil
}

// BuildStatus 返回当前构建任务状态（无任务时为 idle）。
func (s *Server) BuildStatus() *BuildJobInfo {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.build.info()
}

// Load 加载指定路径的（引擎缓存的）索引文件。
func (s *Server) Load(indexPath string) (LoadResult, error) {
	p := strings.TrimSpace(indexPath)
	if p == "" {
		p = s.opts.IndexPath
	}
	if err := s.engine.LoadIndex(p); err != nil {
		return LoadResult{}, err
	}
	s.mu.Lock()
	s.opts.IndexPath = p
	s.mu.Unlock()
	s.logger.Printf("已加载 NN 索引: %s (%d 张)", p, s.engine.Len())
	return LoadResult{Images: s.engine.Len(), Regions: s.engine.Len()}, nil
}

// QueryPath 以本地文件路径作为查询图像检索。
func (s *Server) QueryPath(path string, top, maxdist int, colorWeight float64) (QueryResult, error) {
	img, err := imageproc.Load(filepath.FromSlash(path))
	if err != nil {
		return QueryResult{}, err
	}
	return s.queryImage(img, top)
}

// QueryData 以 base64 data URL 形式传入查询图像检索。
func (s *Server) QueryData(dataURL string, top, maxdist int, colorWeight float64) (QueryResult, error) {
	if i := strings.IndexByte(dataURL, ','); i >= 0 {
		dataURL = dataURL[i+1:]
	}
	data, err := base64.StdEncoding.DecodeString(dataURL)
	if err != nil {
		data, err = base64.RawStdEncoding.DecodeString(dataURL)
		if err != nil {
			return QueryResult{}, fmt.Errorf("无法解析图片数据: %v", err)
		}
	}
	img, err := imageproc.DecodeAny(data, "")
	if err != nil {
		return QueryResult{}, fmt.Errorf("无法解析图像: %v", err)
	}
	return s.queryImage(img, top)
}

// queryImage 用神经网络引擎检索。
func (s *Server) queryImage(img image.Image, top int) (QueryResult, error) {
	if top < 1 {
		top = 5
	}
	if s.engine.Len() == 0 {
		return QueryResult{}, errors.New("尚未加载索引，请先构建或加载索引")
	}
	hits, err := s.engine.Search(img, top)
	if err != nil {
		return QueryResult{}, err
	}
	res := QueryResult{Regions: s.engine.Len()}
	for _, h := range hits {
		res.Matches = append(res.Matches, MatchInfo{
			ImageID: h.Name,
			Score:   round4(h.MaskScore),
			Cover:   1.0,
			Count:   1,
		})
	}
	return res, nil
}

// ImageList 返回当前索引的图像路径列表。
func (s *Server) ImageList() []string {
	if s.engine == nil {
		return nil
	}
	paths := s.engine.Names()
	sort.Strings(paths)
	return paths
}

var mimeByExt = map[string]string{
	".png":  "image/png",
	".jpg":  "image/jpeg",
	".jpeg": "image/jpeg",
	".gif":  "image/gif",
	".webp": "image/webp",
	".bmp":  "image/bmp",
	".tif":  "image/tiff",
	".tiff": "image/tiff",
	".avif": "image/avif",
	".svg":  "image/svg+xml",
}

// ImageDataURI 将本地图像编码为 data URI，供桌面 WebView 展示。
func (s *Server) ImageDataURI(path string) (string, error) {
	ext := strings.ToLower(filepath.Ext(path))
	mime, ok := mimeByExt[ext]
	if !ok {
		return "", fmt.Errorf("不支持的文件类型: %s", ext)
	}
	data, err := os.ReadFile(filepath.FromSlash(path))
	if err != nil {
		return "", err
	}
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data), nil
}

// Segments 计算图像的区域划分（调试）——旧区域哈希算法的调试功能已随引擎替换移除。
func (s *Server) Segments(path string, p SegParams) (SegResult, error) {
	return SegResult{}, errors.New("区域划分调试已随旧算法移除，当前引擎为神经网络排序")
}

// SegmentsPNG 返回区域可视化 PNG 的 data URI（已移除）。
func (s *Server) SegmentsPNG(path string, p SegParams, mode string) (string, error) {
	return "", errors.New("区域划分调试已随旧算法移除，当前引擎为神经网络排序")
}

// SegmentsMapPNG 返回像素编码区域 ID 的标签图 data URI（已移除）。
func (s *Server) SegmentsMapPNG(path string, p SegParams) (string, error) {
	return "", errors.New("区域划分调试已随旧算法移除，当前引擎为神经网络排序")
}
