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

	"go-image-search/internal/detect"
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
	det   *detect.Detector
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
	Detect          bool    `json:"detect"`
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
	mux.HandleFunc("/api/segments.map.png", s.handleSegmentsMapPNG)
	return mux
}

func (s *Server) getDetector() *detect.Detector {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.det != nil {
		return s.det
	}
	d, err := detect.NewDetector(detect.DefaultModelPath(), detect.DefaultLibPath())
	if err != nil {
		s.logger.Printf("检测模型不可用: %v", err)
		return nil
	}
	s.det = d
	return s.det
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

// processRegions 是区域处理的唯一入口（segment → merge → gravity），返回：
//   - res:      原始分段结果（提供像素标签与尺寸，供像素级可视化）
//   - partition: 空间不重叠的划分区域（additive 模式=merge 结果，replace 模式=引力替换后集合）
//   - all:      全部参与检索的区域（partition + 引力辅助区域），顺序编号 1..N
//
// hashImage 与调试端点共用此函数，确保检索行为与调试视图完全一致，避免漏改。
func processRegions(img image.Image, cfg segment.Config, mergeCfg segment.MergeConfig, grav segment.GravityConfig) (*segment.Result, []segment.MergedRegion, []segment.MergedRegion, error) {
	res, err := segment.Segment(img, cfg)
	if err != nil {
		return nil, nil, nil, err
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
	merged := segment.MergeSimilar(img, res, infos, mergeCfg)

	// 引力聚合：区域过多时按引力模型聚合。Additive 模式产出辅助组合区域追加；
	// replace 模式用聚合后的完整集合替换原区域。
	partition := merged
	var aux []segment.MergedRegion
	if grav.Enabled {
		gravity := segment.GravityMerge(img, merged, grav)
		if grav.Additive {
			aux = gravity
		} else {
			partition = gravity
		}
	}

	// 丢弃与图像边界几乎重合的区域（无效边框/背景）
	if grav.FrameRatio > 0 {
		partition = segment.FilterFullFrame(partition, img.Bounds().Dx(), img.Bounds().Dy(), grav.FrameRatio)
	}

	// 整图辅助（绿框）：触发依据优先取蓝色辅助区域数量，其次才是最初划分区域（红框）数量。
	// 在蓝框合并/红框转蓝之前判定，避免 relabel 后 aux 数量变化影响触发。
	var whole *segment.MergedRegion
	if segment.ShouldAddWholeAux(grav, partition, aux) {
		w := segment.MergeAll(img, partition)
		whole = &w
	}

	// 蓝框相交、相互包含 → 合并为新的辅助区域，提升每个索引区域的全局信息。
	aux = segment.MergeContainedOverlapping(img, aux)

	// 红框自动变蓝：划分区域逐个标记为蓝色辅助区域（仅标记，不合并成一个蓝色区域），
	// 使索引只包含蓝框与绿框，红框不再直接入库。
	if grav.Enabled && grav.CombineFew > 0 && len(partition) >= 2 {
		for i := range partition {
			p := partition[i]
			p.Whole = false
			aux = append(aux, p)
		}
	}

	if whole != nil {
		aux = append(aux, *whole)
	}

	// 顺序编号：partition 1..P，aux P+1..P+A
	id := 0
	for i := range partition {
		id++
		partition[i].ID = id
	}
	for i := range aux {
		id++
		aux[i].ID = id
	}
	all := make([]segment.MergedRegion, 0, len(partition)+len(aux))
	all = append(all, partition...)
	all = append(all, aux...)
	return res, partition, all, nil
}

// addDetectedRegion 使用目标检测模型识别图像中的主物体区域，
// 将其作为高权重辅助区域追加到 hashes 列表（Global=2.0）。
func addDetectedRegion(hashes []index.RegionHash, img image.Image, det *detect.Detector) []index.RegionHash {
	if det == nil {
		return hashes
	}
	box, err := det.Detect(img)
	if err != nil {
		return hashes
	}
	b := img.Bounds()
	rect := box.PixelRect(b.Dx(), b.Dy())
	if rect.Empty() || rect.Dx() < 2 || rect.Dy() < 2 {
		return hashes
	}
	crop := detect.CropImage(img, rect)
	if crop == nil {
		return hashes
	}
	h := phash.Hash(segment.BgNormalize(crop))
	shape := phash.Hash(imageproc.StructuralMask(crop))
	bw, bh := rect.Dx(), rect.Dy()
	var sumR, sumG, sumB, cnt int64
	cb := crop.Bounds()
	for y := cb.Min.Y; y < cb.Max.Y; y++ {
		for x := cb.Min.X; x < cb.Max.X; x++ {
			r, g, bl, a := crop.At(x, y).RGBA()
			if a < 32768 {
				continue
			}
			sumR += int64(r >> 8)
			sumG += int64(g >> 8)
			sumB += int64(bl >> 8)
			cnt++
		}
	}
	meanColor := color.RGBA{A: 255}
	if cnt > 0 {
		meanColor = color.RGBA{R: uint8(sumR / cnt), G: uint8(sumG / cnt), B: uint8(sumB / cnt), A: 255}
	}
	nextID := 0
	for _, hh := range hashes {
		if hh.RegionID > nextID {
			nextID = hh.RegionID
		}
	}
	hashes = append(hashes, index.RegionHash{
		RegionID: nextID + 1,
		Hash:     h,
		Shape:    shape,
		Area:     bw * bh,
		BBox:     rect,
		Color:    meanColor,
		NX:       float64(rect.Min.X+rect.Max.X) / (2 * float64(b.Dx())),
		NY:       float64(rect.Min.Y+rect.Max.Y) / (2 * float64(b.Dy())),
		Fill:     1.0,
		Aspect:   float64(bw) / float64(bh),
		Global:   2.0,
	})
	return hashes
}

// hashImage 对图像做区域划分、感知哈希，并将相似相邻区域合并为组合区域。
// 当 grav.Enabled 且区域数过多时，按引力模型聚合出辅助区域（Additive 模式追加，
// replace 模式替换），用于辅助检索。
func hashImage(img image.Image, cfg segment.Config, mergeCfg segment.MergeConfig, grav segment.GravityConfig) ([]index.RegionHash, error) {
	res, partition, all, err := processRegions(img, cfg, mergeCfg, grav)
	if err != nil {
		return nil, err
	}
	idx := all
	if len(all) > len(partition) {
		idx = all[len(partition):] // 仅蓝框+绿框入库
	}
	hashes := make([]index.RegionHash, 0, len(idx))
	for _, m := range idx {
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
			Global:   segment.GlobalWeight(m),
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

	var det *detect.Detector
	if req.Detect {
		d, err := detect.NewDetector(detect.DefaultModelPath(), detect.DefaultLibPath())
		if err != nil {
			s.logger.Printf("检测模型不可用: %v", err)
		} else {
			det = d
			defer det.Close()
		}
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
		// 索引阶段也启用引力聚合：区域过多时聚合出辅助组合区域一并入索引，
		// 与查询阶段的辅助查询区域对应，提升碎片化图像的召回。
		for _, v := range imageproc.QueryVariants(img) {
			hashes, err := hashImage(v, cfg, mc, segment.DefaultGravityConfig())
			if err != nil {
				continue
			}
			ix.AddImage(id, hashes)
		}
		if det != nil {
			detHashes := addDetectedRegion(nil, img, det)
			if len(detHashes) > 0 {
				ix.AddImage(id, detHashes)
			}
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
	grav := gravityConfigFromQuery(r.Form)
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
	if r.FormValue("detect") == "true" {
		if det := s.getDetector(); det != nil {
			detHashes := addDetectedRegion(nil, img, det)
			if len(detHashes) > 0 {
				qs := make([]index.QueryRegion, 0, len(detHashes))
				for _, h := range detHashes {
					qs = append(qs, index.QueryRegion{
						Hash: h.Hash, Shape: h.Shape, Area: h.Area, Color: h.Color,
						NX: h.NX, NY: h.NY, Fill: h.Fill, Aspect: h.Aspect, Global: h.Global,
					})
				}
				querySets = append(querySets, qs)
			}
		}
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
	ID       int    `json:"id"`
	Area     int    `json:"area"`
	BBox     [4]int `json:"bbox"`
	Color    string `json:"color"`
	Whole    bool   `json:"whole"`
	Detected bool   `json:"detected"`
}

func (s *Server) handleSegments(w http.ResponseWriter, r *http.Request) {
	img, ok := s.loadImageForSegments(w, r)
	if !ok {
		return
	}
	q := r.URL.Query()
	cfg := segConfigFromQuery(q)
	grav := gravityConfigFromQuery(q)
	res, _, all, err := processRegions(img, cfg, segment.DefaultMergeConfig(), grav)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
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
	if q.Get("detect") == "true" {
		if det := s.getDetector(); det != nil {
			if box, err := det.Detect(img); err == nil {
				b := img.Bounds()
				rect := box.PixelRect(b.Dx(), b.Dy())
				if !rect.Empty() && rect.Dx() >= 2 && rect.Dy() >= 2 {
					crop := detect.CropImage(img, rect)
					var meanColor color.RGBA
					if crop != nil {
						var sumR, sumG, sumB, cnt int64
						cb := crop.Bounds()
						for y := cb.Min.Y; y < cb.Max.Y; y++ {
							for x := cb.Min.X; x < cb.Max.X; x++ {
								cr, cg, cbl, a := crop.At(x, y).RGBA()
								if a < 32768 {
									continue
								}
								sumR += int64(cr >> 8)
								sumG += int64(cg >> 8)
								sumB += int64(cbl >> 8)
								cnt++
							}
						}
						if cnt > 0 {
							meanColor = color.RGBA{R: uint8(sumR / cnt), G: uint8(sumG / cnt), B: uint8(sumB / cnt), A: 255}
						} else {
							meanColor = color.RGBA{A: 255}
						}
					}
					nextID := 0
					for _, r := range regions {
						if r.ID > nextID {
							nextID = r.ID
						}
					}
					regions = append(regions, regionJSON{
						ID:       nextID + 1,
						Area:     rect.Dx() * rect.Dy(),
						BBox:     [4]int{rect.Min.X, rect.Min.Y, rect.Max.X, rect.Max.Y},
						Color:    fmt.Sprintf("#%02x%02x%02x", meanColor.R, meanColor.G, meanColor.B),
						Detected: true,
					})
				}
			}
		}
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
	res, partition, all, err := processRegions(img, cfg, segment.DefaultMergeConfig(), grav)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	var detRect *image.Rectangle
	if q.Get("detect") == "true" {
		if det := s.getDetector(); det != nil {
			if box, err := det.Detect(img); err == nil {
				b := img.Bounds()
				rect := box.PixelRect(b.Dx(), b.Dy())
				if !rect.Empty() && rect.Dx() >= 2 && rect.Dy() >= 2 {
					detRect = &rect
				}
			}
		}
	}
	var vis *image.RGBA
	if q.Get("mode") == "fill" {
		vis = visualizeFill(res, partition, img)
	} else {
		vis = visualize(partition, all[len(partition):], img, detRect)
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
	res, partition, _, err := processRegions(img, cfg, segment.DefaultMergeConfig(), grav)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	vis := visualizeLabels(res, partition)
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
func visualize(partition, aux []segment.MergedRegion, src image.Image, detRect *image.Rectangle) *image.RGBA {
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
	if detRect != nil {
		drawBox(dst, *detRect, color.RGBA{255, 165, 0, 255})
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
