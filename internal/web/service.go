// Package web 提供 iOS 风格图片搜索的 Web 界面与 API。
// service.go 提供面向桌面 GUI（Wails）的文件/字节级操作方法，
// 复用 server.go 的索引、构建与检索逻辑，不依赖 HTTP 上下文。
package web

import (
	"bytes"
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
	"go-image-search/internal/index"
	"go-image-search/internal/sczl"
	"go-image-search/internal/segment"
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

// BuildJobInfo 后台构建任务的进度状态（对 GUI 暴露的公共视图）。
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

// QueryResult 检索结果（与 HTTP handleQuery 的响应一致）。
type QueryResult struct {
	Regions int         `json:"regions"`
	Matches []MatchInfo `json:"matches"`
}

// MatchInfo 单张图像的匹配信息。
type MatchInfo struct {
	ImageID string        `json:"imageId"`
	Score   float64       `json:"score"`
	Cover   float64       `json:"cover"`
	Count   int           `json:"count"`
	Regions []RegionMatch `json:"regions"`
}

// RegionMatch 单个命中区域。
type RegionMatch struct {
	RegionID int    `json:"regionId"`
	Hash     string `json:"hash"`
	Dist     int    `json:"dist"`
	Area     int    `json:"area"`
	Color    string `json:"color"`
}

// SegParams 区域划分（调试）的参数。
type SegParams struct {
	ThresholdPct     float64 `json:"thresholdPct"`
	ThresholdFactor  float64 `json:"thresholdFactor"`
	MinAreaRatio     float64 `json:"minAreaRatio"`
	MedianFilterK    int     `json:"medianFilterK"`
	Connectivity     int     `json:"connectivity"`
	Gravity          bool    `json:"gravity"`
	GravityMin       int     `json:"gravityMin"`
	GravityTrigger   int     `json:"gravityTrigger"`
	GravityAdditive  bool    `json:"gravityAdditive"`
	GravityCombineFew int    `json:"gravityCombineFew"`
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
	st.Algorithm = s.algorithm
	if s.algorithm == "sczl" && s.sczlIndex != nil {
		st.SczlLoaded = true
		st.Loaded = true
		st.Images = len(s.sczlIndex.Images)
		st.Regions = s.sczlIndex.Len()
	} else if s.index != nil {
		st.Loaded = true
		st.Images = len(s.index.Images)
		st.Regions = s.index.Len()
	}
	st.Build = s.build.info()
	return st
}

// GetAlgorithm 返回当前检索策略。
func (s *Server) GetAlgorithm() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.algorithm
}

// SetAlgorithm 切换检索策略（region 或 sczl）。
func (s *Server) SetAlgorithm(alg string) error {
	a := strings.ToLower(strings.TrimSpace(alg))
	if a != "region" && a != "sczl" {
		return errors.New("algorithm 必须是 region 或 sczl")
	}
	s.mu.Lock()
	s.algorithm = a
	s.mu.Unlock()
	s.logger.Printf("检索策略切换为: %s", a)
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
	alg := s.algorithm
	job := &buildJob{
		ID:      fmt.Sprintf("build-%d", time.Now().UnixNano()),
		State:   "running",
		Started: time.Now(),
		Index:   req.Out,
	}
	s.build = job
	s.mu.Unlock()

	br := buildRequest{
		Dir: req.Dir, Out: req.Out,
		ThresholdPct:    req.ThresholdPct,
		ThresholdFactor: req.ThresholdFactor,
		MinAreaRatio:    req.MinAreaRatio,
		MedianFilterK:   req.MedianFilterK,
		Connectivity:    req.Connectivity,
		MergeHashDist:   req.MergeHashDist,
		MergeColorDist:  req.MergeColorDist,
		NoMerge:         req.NoMerge,
	}
	if alg == "sczl" {
		go s.runBuildSczl(job, br)
	} else {
		go s.runBuild(job, br)
	}
	return job.info(), nil
}

// BuildStatus 返回当前构建任务状态（无任务时为 idle）。
func (s *Server) BuildStatus() *BuildJobInfo {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.build.info()
}

// Load 加载指定路径的索引文件。
func (s *Server) Load(indexPath string) (LoadResult, error) {
	s.mu.Lock()
	alg := s.algorithm
	def := s.opts.IndexPath
	s.mu.Unlock()

	if alg == "sczl" {
		p := strings.TrimSpace(indexPath)
		if p == "" {
			p = def + ".sczl"
		}
		ix, err := sczl.Load(p)
		if err != nil {
			return LoadResult{}, err
		}
		s.mu.Lock()
		s.sczlIndex = ix
		s.mu.Unlock()
		s.logger.Printf("已加载 SCZL 索引: %s", p)
		return LoadResult{Images: len(ix.Images), Regions: ix.Len()}, nil
	}

	p := strings.TrimSpace(indexPath)
	if p == "" {
		p = def
	}
	ix, err := index.Load(p)
	if err != nil {
		return LoadResult{}, err
	}
	s.mu.Lock()
	s.index = ix
	s.opts.IndexPath = p
	s.mu.Unlock()
	s.logger.Printf("已加载索引: %s", p)
	return LoadResult{Images: len(ix.Images), Regions: ix.Len()}, nil
}

// QueryPath 以本地文件路径作为查询图像检索。
func (s *Server) QueryPath(path string, top, maxdist int, colorWeight float64) (QueryResult, error) {
	img, err := imageproc.Load(filepath.FromSlash(path))
	if err != nil {
		return QueryResult{}, err
	}
	return s.queryImage(img, top, maxdist, colorWeight)
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
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return QueryResult{}, fmt.Errorf("无法解析图像: %v", err)
	}
	return s.queryImage(img, top, maxdist, colorWeight)
}

// queryImage 按当前检索策略执行检索。
func (s *Server) queryImage(img image.Image, top, maxdist int, colorWeight float64) (QueryResult, error) {
	if top < 1 {
		top = 5
	}
	s.mu.Lock()
	alg := s.algorithm
	ix := s.index
	sczlIx := s.sczlIndex
	s.mu.Unlock()

	if alg == "sczl" {
		if sczlIx == nil {
			return QueryResult{}, errors.New("尚未加载 SCZL 索引，请先构建或加载索引")
		}
		return s.querySczl(sczlIx, img, top)
	}
	if ix == nil {
		return QueryResult{}, errors.New("尚未加载索引，请先构建或加载索引")
	}
	if maxdist < 1 {
		maxdist = 12
	}
	return s.queryRegion(ix, img, top, maxdist, colorWeight)
}

// queryRegion 区域感知哈希检索（pHash 方案）。
func (s *Server) queryRegion(ix *index.Index, img image.Image, top, maxdist int, colorWeight float64) (QueryResult, error) {
	cfg := segment.DefaultConfig()
	grav := segment.DefaultGravityConfig()
	grav.CombineAlways = true
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
	matches := ix.SearchMulti(querySets, index.SearchOptions{MaxDist: maxdist, ColorWeight: colorWeight})
	if len(matches) > top {
		matches = matches[:top]
	}
	total := 0
	for _, qs := range querySets {
		total += len(qs)
	}
	return buildQueryResult(total, matches), nil
}

// querySczl SCZL 方案（NCC+HOG+形状上下文）。
func (s *Server) querySczl(ix *sczl.Index, img image.Image, top int) (QueryResult, error) {
	q := sczl.Extract(img)
	if !q.Valid {
		return QueryResult{}, errors.New("无法提取前景")
	}
	matches := ix.Query(q, sczl.DefaultOptions())
	if len(matches) > top {
		matches = matches[:top]
	}
	res := QueryResult{Regions: 1}
	for _, m := range matches {
		res.Matches = append(res.Matches, MatchInfo{
			ImageID: m.ImageID,
			Score:   round4(m.Score),
			Cover:   1.0,
			Count:   1,
		})
	}
	return res, nil
}

// buildQueryResult 将底层命中转换为对外 JSON 结构。
func buildQueryResult(totalRegions int, matches []index.Match) QueryResult {
	res := QueryResult{Regions: totalRegions}
	for _, m := range matches {
		mm := MatchInfo{
			ImageID: m.ImageID,
			Score:   round4(m.Score),
			Cover:   round4(m.CoverRatio),
			Count:   len(m.Matches),
		}
		for _, rm := range m.Matches {
			mm.Regions = append(mm.Regions, RegionMatch{
				RegionID: rm.Entry.RegionID,
				Hash:     fmt.Sprintf("%016x", rm.Entry.Hash),
				Dist:     rm.Dist,
				Area:     rm.Entry.Area,
				Color:    fmt.Sprintf("#%02x%02x%02x", rm.Entry.Color.R, rm.Entry.Color.G, rm.Entry.Color.B),
			})
		}
		res.Matches = append(res.Matches, mm)
	}
	return res
}

// ImageList 返回当前索引的图像路径列表。
func (s *Server) ImageList() []string {
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
	paths := make([]string, 0, len(images))
	for p := range images {
		paths = append(paths, p)
	}
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

// Segments 计算图像的区域划分（调试）。
func (s *Server) Segments(path string, p SegParams) (SegResult, error) {
	img, err := imageproc.Load(filepath.FromSlash(path))
	if err != nil {
		return SegResult{}, err
	}
	cfg := segConfigFromParams(p)
	grav := gravConfigFromParams(p)
	res, out, err := segment.RunDefault(img, cfg, segment.DefaultMergeConfig(), grav)
	if err != nil {
		return SegResult{}, err
	}
	all := out.All()
	regions := make([]SegRegion, 0, len(all))
	for _, reg := range all {
		regions = append(regions, SegRegion{
			ID:    reg.ID,
			Area:  reg.Area,
			BBox:  [4]int{reg.BBox.Min.X, reg.BBox.Min.Y, reg.BBox.Max.X, reg.BBox.Max.Y},
			Color: fmt.Sprintf("#%02x%02x%02x", reg.MeanColor.R, reg.MeanColor.G, reg.MeanColor.B),
			Whole: reg.Whole,
		})
	}
	return SegResult{Width: res.Width, Height: res.Height, Regions: regions}, nil
}

// SegmentsPNG 返回区域可视化 PNG 的 data URI（mode=bbox/fill）。
func (s *Server) SegmentsPNG(path string, p SegParams, mode string) (string, error) {
	img, err := imageproc.Load(filepath.FromSlash(path))
	if err != nil {
		return "", err
	}
	cfg := segConfigFromParams(p)
	grav := gravConfigFromParams(p)
	res, out, err := segment.RunDefault(img, cfg, segment.DefaultMergeConfig(), grav)
	if err != nil {
		return "", err
	}
	var vis *image.RGBA
	if mode == "fill" {
		vis = visualizeFill(res, out.Partition, img)
	} else {
		vis = visualize(out.Partition, out.Aux, img)
	}
	data, err := imageproc.EncodePNG(vis)
	if err != nil {
		return "", err
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(data), nil
}

// SegmentsMapPNG 返回像素编码区域 ID 的标签图 data URI。
func (s *Server) SegmentsMapPNG(path string, p SegParams) (string, error) {
	img, err := imageproc.Load(filepath.FromSlash(path))
	if err != nil {
		return "", err
	}
	cfg := segConfigFromParams(p)
	grav := gravConfigFromParams(p)
	res, out, err := segment.RunDefault(img, cfg, segment.DefaultMergeConfig(), grav)
	if err != nil {
		return "", err
	}
	vis := visualizeLabels(res, out.Partition)
	data, err := imageproc.EncodePNG(vis)
	if err != nil {
		return "", err
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(data), nil
}

func segConfigFromParams(p SegParams) segment.Config {
	cfg := segment.DefaultConfig()
	if p.ThresholdPct > 0 {
		cfg.ThresholdPct = p.ThresholdPct
	}
	if p.ThresholdFactor > 0 {
		cfg.ThresholdFactor = p.ThresholdFactor
	}
	if p.MinAreaRatio > 0 {
		cfg.MinAreaRatio = p.MinAreaRatio
	}
	if p.MedianFilterK != 0 {
		cfg.MedianFilterK = p.MedianFilterK
	}
	if p.Connectivity != 0 {
		cfg.Connectivity = p.Connectivity
	}
	return cfg
}

func gravConfigFromParams(p SegParams) segment.GravityConfig {
	gc := segment.DefaultGravityConfig()
	if p.GravityMin >= 1 {
		gc.MinRegions = p.GravityMin
	}
	if p.GravityTrigger >= 1 {
		gc.TriggerCount = p.GravityTrigger
	}
	if p.GravityCombineFew >= 0 {
		gc.CombineFew = p.GravityCombineFew
	}
	if p.GravityFrameRatio >= 0 {
		gc.FrameRatio = p.GravityFrameRatio
	}
	if p.Gravity && p.GravityCombineFew > 0 {
		gc.Additive = p.GravityAdditive
	}
	return gc
}