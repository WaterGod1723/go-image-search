package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"net/http"
	"os"
	"strings"

	"go-image-search/internal/imageproc"
	"go-image-search/internal/index"
	"go-image-search/internal/phash"
	"go-image-search/internal/segment"
	"go-image-search/internal/web"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "build":
		runBuild(os.Args[2:])
	case "query":
		runQuery(os.Args[2:])
	case "segments":
		runSegments(os.Args[2:])
	case "serve":
		runServe(os.Args[2:])
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `图片反向搜索（感知哈希 + 区域划分）

用法:
  go-image-search build -dir <图像库目录> -out <索引文件> [分段参数...]
  go-image-search query -index <索引文件> -q <查询图像> [-top N] [-maxdist D]
  go-image-search segments -img <图像> [-out <可视化png>]   # 调试：查看区域划分
  go-image-search serve [-addr <host:port>] [-root <图像库目录>] [-index <索引文件>]  # iOS 风格 Web 界面

分段参数: -threshold-pct <0~1> -factor <x> -min-area-ratio <r> -median <k> -connectivity <4|8>
`)
}

func buildFlagSet(fs *flag.FlagSet) *segment.Config {
	cfg := segment.DefaultConfig()
	fs.Float64Var(&cfg.ThresholdPct, "threshold-pct", cfg.ThresholdPct, "相邻色差分位数(0~1)")
	fs.Float64Var(&cfg.ThresholdFactor, "factor", cfg.ThresholdFactor, "阈值因子")
	fs.Float64Var(&cfg.MinAreaRatio, "min-area-ratio", cfg.MinAreaRatio, "噪声区域面积比例阈值")
	fs.IntVar(&cfg.MedianFilterK, "median", cfg.MedianFilterK, "预处理中值滤波核(0关闭)")
	fs.IntVar(&cfg.Connectivity, "connectivity", cfg.Connectivity, "连通性(4或8)")
	return &cfg
}

// hashImage 对图像做区域划分并为每个区域计算感知哈希。
func hashImage(img image.Image, cfg segment.Config) ([]index.RegionHash, error) {
	res, err := segment.Segment(img, cfg)
	if err != nil {
		return nil, err
	}
	hashes := make([]index.RegionHash, 0, len(res.Regions))
	for _, reg := range res.Regions {
		crop := res.Crop(img, reg.ID)
		if crop == nil {
			continue
		}
		h := phash.Hash(crop)
		hashes = append(hashes, index.RegionHash{
			RegionID: reg.ID,
			Hash:     h,
			Area:     reg.Area,
			BBox:     reg.BBox,
			Color:    reg.MeanColor,
		})
	}
	return hashes, nil
}

func runBuild(args []string) {
	fs := flag.NewFlagSet("build", flag.ExitOnError)
	dir := fs.String("dir", "", "图像库目录")
	out := fs.String("out", "index.bin", "输出索引文件")
	cfg := buildFlagSet(fs)
	fs.Parse(args)

	if *dir == "" {
		fmt.Fprintln(os.Stderr, "缺少 -dir")
		os.Exit(2)
	}
	files, err := imageproc.LoadSupported(*dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "扫描目录失败: %v\n", err)
		os.Exit(1)
	}
	if len(files) == 0 {
		fmt.Fprintln(os.Stderr, "目录中无支持的图片")
		os.Exit(1)
	}

	ix := index.New()
	for i, f := range files {
		img, err := imageproc.Load(f)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[跳过] %s: %v\n", f, err)
			continue
		}
		hashes, err := hashImage(img, *cfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[跳过] %s: %v\n", f, err)
			continue
		}
		id := strings.ReplaceAll(f, "\\", "/")
		ix.AddImage(id, hashes)
		fmt.Printf("[%d/%d] %s (%d 区域)\n", i+1, len(files), id, len(hashes))
	}

	if err := ix.Save(*out); err != nil {
		fmt.Fprintf(os.Stderr, "保存索引失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("完成: %d 张图像, %d 个区域 -> %s\n", len(ix.Images), ix.Len(), *out)
}

func runQuery(args []string) {
	fs := flag.NewFlagSet("query", flag.ExitOnError)
	idxPath := fs.String("index", "index.bin", "索引文件")
	q := fs.String("q", "", "查询图像")
	top := fs.Int("top", 5, "返回 top-N")
	maxDist := fs.Int("maxdist", 12, "区域哈希最大汉明距离")
	colorWeight := fs.Float64("color-weight", 0.8, "颜色相似度权重(0关闭)")
	cfg := buildFlagSet(fs)
	fs.Parse(args)

	if *q == "" {
		fmt.Fprintln(os.Stderr, "缺少 -q")
		os.Exit(2)
	}
	ix, err := index.Load(*idxPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "加载索引失败: %v\n", err)
		os.Exit(1)
	}
	img, err := imageproc.Load(*q)
	if err != nil {
		fmt.Fprintf(os.Stderr, "加载查询图像失败: %v\n", err)
		os.Exit(1)
	}
	hashes, err := hashImage(img, *cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "查询图像处理失败: %v\n", err)
		os.Exit(1)
	}

	query := make([]index.QueryRegion, 0, len(hashes))
	for _, h := range hashes {
		query = append(query, index.QueryRegion{Hash: h.Hash, Area: h.Area, Color: h.Color})
	}

	matches := ix.Search(query, index.SearchOptions{MaxDist: *maxDist, ColorWeight: *colorWeight})

	fmt.Printf("查询区域数: %d\n", len(query))
	if len(matches) == 0 {
		fmt.Println("无匹配结果")
		return
	}
	for i, m := range matches {
		if i >= *top {
			break
		}
		fmt.Printf("#%d score=%.4f cover=%.2f%% %s\n", i+1, m.Score, m.CoverRatio*100, m.ImageID)
		for _, rm := range m.Matches {
			fmt.Printf("    region %d dist=%d hash=%016x bbox=%v\n",
				rm.Entry.RegionID, rm.Dist, rm.Entry.Hash, rm.Entry.BBox)
		}
	}
}

// visualize 在图像上叠加区域边界框，便于调试。
func runServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	addr := fs.String("addr", "127.0.0.1:8080", "HTTP 监听地址")
	root := fs.String("root", ".", "图像库根目录（用于图库浏览/缩略图）")
	idxPath := fs.String("index", "index.bin", "索引文件路径（存在则自动加载）")
	fs.Parse(args)

	srv := web.New(web.Options{IndexPath: *idxPath, Root: *root})
	if err := srv.LoadIndex(); err != nil {
		fmt.Fprintf(os.Stderr, "提示: 加载索引失败(%v)，可在界面中构建或加载索引\n", err)
	}
	fmt.Printf("Web 界面: http://%s\n", *addr)
	fmt.Printf("图像库根目录: %s\n", *root)
	if err := http.ListenAndServe(*addr, srv.Handler()); err != nil {
		fmt.Fprintf(os.Stderr, "服务启动失败: %v\n", err)
		os.Exit(1)
	}
}

func visualize(res *segment.Result, src image.Image) *image.RGBA {
	b := src.Bounds()
	dst := image.NewRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.Set(x, y, src.At(x, y))
		}
	}
	for _, r := range res.Regions {
		box := r.BBox
		drawBox(dst, box, color.RGBA{255, 0, 0, 255})
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

func runSegments(args []string) {
	fs := flag.NewFlagSet("segments", flag.ExitOnError)
	imgPath := fs.String("img", "", "图像路径")
	outPath := fs.String("out", "segments.png", "可视化输出")
	cfg := buildFlagSet(fs)
	fs.Parse(args)

	if *imgPath == "" {
		fmt.Fprintln(os.Stderr, "缺少 -img")
		os.Exit(2)
	}
	img, err := imageproc.Load(*imgPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "加载图像失败: %v\n", err)
		os.Exit(1)
	}
	res, err := segment.Segment(img, *cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "区域划分失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("图像 %dx%d, 区域数 %d, 各区域: 面积 bbox\n", res.Width, res.Height, len(res.Regions))
	for _, r := range res.Regions {
		fmt.Printf("  #%d area=%d bbox=%v mean=%v\n", r.ID, r.Area, r.BBox, r.MeanColor)
	}
	if *outPath != "" {
		vis := visualize(res, img)
		data, err := imageproc.EncodePNG(vis)
		if err != nil {
			fmt.Fprintf(os.Stderr, "可视化失败: %v\n", err)
			os.Exit(1)
		}
		if err := os.WriteFile(*outPath, data, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "写入可视化失败: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("可视化已保存: %s\n", *outPath)
	}
}
