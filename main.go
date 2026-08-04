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
	case "sczl-build":
		runSCZLBuild(os.Args[2:])
	case "sczl-query":
		runSCZLQuery(os.Args[2:])
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
  go-image-search sczl-build -dir <图像库目录> -out <索引文件> [-jobs N]  # SCZL 算法：形状上下文+Fourier+LBP
  go-image-search sczl-query -index <索引文件> -q <查询图像> [-top N]  # SCZL 检索

分段参数: -threshold-pct <0~1> -factor <x> -min-area-ratio <r> -median <k> -connectivity <4|8>
合并参数: -merge-dist <汉明距离阈值> -merge-color <颜色阈值> -no-merge  # 相似相邻区域合并为组合区域
引力参数(检索): -gravity <bool> -gravity-min <保留区域数> -gravity-trigger <触发阈值> -gravity-additive <bool> -gravity-combine-few <阈值>  # 区域过多时引力聚合；区域过少时合并为整图辅助区域；查询阶段始终追加整图辅助区域
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

// mergeFlagSet 注册相似区域合并参数。
func mergeFlagSet(fs *flag.FlagSet) *segment.MergeConfig {
	mc := segment.DefaultMergeConfig()
	fs.BoolVar(&mc.Enabled, "no-merge", mc.Enabled, "关闭相似区域合并")
	fs.IntVar(&mc.HashDist, "merge-dist", mc.HashDist, "合并的 pHash 汉明距离阈值")
	fs.Float64Var(&mc.ColorDist, "merge-color", mc.ColorDist, "合并的平均色归一化距离阈值")
	fs.Float64Var(&mc.GapFactor, "merge-gap", mc.GapFactor, "合并的空间邻近系数(0=仅邻接)")
	return &mc
}

// gravityFlagSet 注册引力聚合参数（检索阶段：区域过多时按引力模型聚合辅助检索）。
func gravityFlagSet(fs *flag.FlagSet) *segment.GravityConfig {
	gc := segment.DefaultGravityConfig()
	fs.BoolVar(&gc.Enabled, "gravity", gc.Enabled, "启用引力聚合（区域过多时聚合出辅助区域）")
	fs.IntVar(&gc.MinRegions, "gravity-min", gc.MinRegions, "引力聚合至少保留的区域数（停止阈值）")
	fs.IntVar(&gc.TriggerCount, "gravity-trigger", gc.TriggerCount, "区域数超过该值才启动引力聚合")
	fs.BoolVar(&gc.Additive, "gravity-additive", gc.Additive, "true=追加辅助组合区域；false=替换原区域")
	fs.IntVar(&gc.CombineFew, "gravity-combine-few", gc.CombineFew, "最终区域数少于该值时合并为一个整体辅助索引区域（0=禁用）")
	fs.Float64Var(&gc.FrameRatio, "gravity-frame-ratio", gc.FrameRatio, "丢弃与图像边界几乎重合的边框区域覆盖比例（0=禁用）")
	return &gc
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

func runBuild(args []string) {
	fs := flag.NewFlagSet("build", flag.ExitOnError)
	dir := fs.String("dir", "", "图像库目录")
	out := fs.String("out", "index.bin", "输出索引文件")
	jobs := fs.Int("jobs", 0, "并发构建线程数(0=CPU 核数)")
	cfg := buildFlagSet(fs)
	mc := mergeFlagSet(fs)
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
	added := ix.BuildParallel(files, *jobs, func(f string) (string, []index.RegionHash, error) {
		img, err := imageproc.Load(f)
		if err != nil {
			return "", nil, err
		}
		id := strings.ReplaceAll(f, "\\", "/")
		// 原图 + 骨架图分别入索引，让查询的任一衍生图都能命中对应表示。
		// 索引阶段也启用引力聚合：区域过多时聚合出辅助组合区域一并入索引，
		// 与查询阶段的辅助查询区域对应，提升碎片化图像的召回。
		var regions []index.RegionHash
		for _, v := range imageproc.QueryVariants(img) {
			hashes, err := hashImage(v, *cfg, *mc, segment.DefaultGravityConfig())
			if err != nil {
				continue
			}
			regions = append(regions, hashes...)
		}
		return id, regions, nil
	}, func(p index.BuildProgress) {
		if p.Err != nil {
			fmt.Fprintf(os.Stderr, "[跳过] %s: %v\n", p.File, p.Err)
			return
		}
		fmt.Printf("[%d/%d] %s (%d 区域, 含骨架)\n", p.Done, len(files), p.ID, p.Regions)
	})

	if err := ix.Save(*out); err != nil {
		fmt.Fprintf(os.Stderr, "保存索引失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("完成: %d 张图像, %d 个区域 -> %s\n", added, ix.Len(), *out)
}

func runQuery(args []string) {
	fs := flag.NewFlagSet("query", flag.ExitOnError)
	idxPath := fs.String("index", "index.bin", "索引文件")
	q := fs.String("q", "", "查询图像")
	top := fs.Int("top", 5, "返回 top-N")
	maxDist := fs.Int("maxdist", 12, "区域哈希最大汉明距离")
	colorWeight := fs.Float64("color-weight", 0.1, "颜色相似度权重(0关闭)")
	cfg := buildFlagSet(fs)
	mc := mergeFlagSet(fs)
	gc := gravityFlagSet(fs)
	fs.Parse(args)
	// 查询阶段无条件追加整图辅助区域，提升整图级别的召回
	gc.CombineAlways = true

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
	// 对查询图像生成 原图 + 骨架图 分别检索；若检测到统一背景色与附属文字带，
	// 再追加归一化裁剪内容的整图辅助区域（白底+主体，与图库透明底图标可比）。
	// 任一衍生图的任一区域与索引图像相似即认为该图像相似。
	querySets := make([][]index.QueryRegion, 0, 3)
	for _, v := range imageproc.QueryVariants(img) {
		hashes, err := hashImage(v, *cfg, *mc, *gc)
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
	if whole := imageproc.QueryNormalizedWholeHashes(img, *cfg, *mc, *gc); len(whole) > 0 {
		qs := make([]index.QueryRegion, 0, len(whole))
		for _, h := range whole {
			qs = append(qs, index.QueryRegion{
				Hash: h.Hash, Shape: h.Shape, Area: h.Area, Color: h.Color,
				NX: h.NX, NY: h.NY, Fill: h.Fill, Aspect: h.Aspect, Global: h.Global,
			})
		}
		querySets = append(querySets, qs)
	}
	matches := ix.SearchMulti(querySets, index.SearchOptions{MaxDist: *maxDist, ColorWeight: *colorWeight})

	totalRegions := 0
	for _, qs := range querySets {
		totalRegions += len(qs)
	}
	fmt.Printf("衍生图像数: %d（原图+骨架）, 查询区域总数: %d\n", len(querySets), totalRegions)
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
	for _, r := range partition {
		drawBox(dst, r.BBox, color.RGBA{255, 0, 0, 255})
	}
	for _, r := range aux {
		c := color.RGBA{0, 0, 255, 255}
		if r.Whole {
			c = color.RGBA{0, 255, 0, 255}
		}
		drawBox(dst, r.BBox, c)
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
	// segments 子命令默认启用引力聚合，与 build/query 检索行为一致。
	grav := gravityFlagSet(fs)
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
	res, out, err := segment.RunDefault(img, *cfg, segment.DefaultMergeConfig(), *grav)
	if err != nil {
		fmt.Fprintf(os.Stderr, "区域划分失败: %v\n", err)
		os.Exit(1)
	}
	all := out.All()
	fmt.Printf("图像 %dx%d, 检索区域数 %d (划分 %d + 引力辅助 %d), 各区域: 面积 bbox\n",
		res.Width, res.Height, len(all), len(out.Partition), len(out.Aux))
	for _, r := range all {
		fmt.Printf("  #%d area=%d bbox=%v mean=%v\n", r.ID, r.Area, r.BBox, r.MeanColor)
	}
	if *outPath != "" {
		vis := visualize(out.Partition, out.Aux, img)
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
