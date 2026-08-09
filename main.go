// Command go-image-search 是基于神经网络排序引擎（image-search-test）的
// 图片反向搜索桌面应用/CLI。无参数时启动 Wails 桌面 GUI；带子命令时进入
// 命令行模式。底层算法与索引全部由 internal/nnengine（MLP 融合手工特征 +
// sczl 颜色无关专家 + 两阶段粗筛）承担。
package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"

	"go-image-search/internal/imageproc"
	"go-image-search/internal/nnengine"
	"go-image-search/internal/web"
)

func main() {
	// 不带参数时默认启动桌面 GUI 界面；带子命令时进入命令行模式。
	if len(os.Args) < 2 {
		runGUI()
		return
	}
	switch os.Args[1] {
	case "build":
		runBuild(os.Args[2:])
	case "query":
		runQuery(os.Args[2:])
	case "serve":
		runServe(os.Args[2:])
	case "help", "-h", "--help":
		usage()
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `图片反向搜索（神经网络排序引擎）

用法:
  不带参数（默认）启动桌面 GUI 界面：
  go-image-search
  go-image-search help                       # 查看本帮助

  go-image-search build -dir <图像库目录> -out <索引文件>
  go-image-search query -index <索引文件> -q <查询图像> [-top N]
  go-image-search serve [-addr <host:port>] [-root <图像库目录>] [-index <索引文件>]

神经网络权重：默认读取当前目录/程序目录 weights.gob，可用环境变量 NN_WEIGHTS 覆盖。
`)
}

func runBuild(args []string) {
	fs := flag.NewFlagSet("build", flag.ExitOnError)
	dir := fs.String("dir", "", "图像库目录")
	out := fs.String("out", "index.bin", "输出索引（引擎缓存）文件")
	fs.Parse(args)

	if *dir == "" {
		fmt.Fprintln(os.Stderr, "缺少 -dir")
		os.Exit(2)
	}
	e := nnengine.New(defaultWeights())
	if err := e.BuildIndex(*dir, *out); err != nil {
		fmt.Fprintf(os.Stderr, "构建索引失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("完成: %d 张图像 -> %s\n", e.Len(), *out)
}

func runQuery(args []string) {
	fs := flag.NewFlagSet("query", flag.ExitOnError)
	idxPath := fs.String("index", "index.bin", "索引文件")
	q := fs.String("q", "", "查询图像")
	top := fs.Int("top", 5, "返回 top-N")
	fs.Parse(args)

	if *q == "" {
		fmt.Fprintln(os.Stderr, "缺少 -q")
		os.Exit(2)
	}
	e := nnengine.New(defaultWeights())
	if err := e.LoadIndex(*idxPath); err != nil {
		fmt.Fprintf(os.Stderr, "加载索引失败: %v\n", err)
		os.Exit(1)
	}
	img, err := imageproc.Load(*q)
	if err != nil {
		fmt.Fprintf(os.Stderr, "加载查询图像失败: %v\n", err)
		os.Exit(1)
	}
	hits, err := e.Search(img, *top)
	if err != nil {
		fmt.Fprintf(os.Stderr, "检索失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("索引图像数: %d\n", e.Len())
	if len(hits) == 0 {
		fmt.Println("无匹配结果")
		return
	}
	for i, h := range hits {
		fmt.Printf("#%d score=%.4f mask=%.4f %s\n", i+1, h.RankScore, h.MaskScore, h.Name)
	}
}

// defaultWeights 返回神经网络权重路径（NN_WEIGHTS 或当前目录 weights.gob）。
func defaultWeights() string {
	if v := os.Getenv("NN_WEIGHTS"); v != "" {
		return v
	}
	return "weights.gob"
}

func runServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	addr := fs.String("addr", "127.0.0.1:8080", "HTTP 监听地址")
	root := fs.String("root", ".", "图像库根目录（用于图库浏览/缩略图）")
	idxPath := fs.String("index", "index.bin", "索引文件路径（存在则自动加载）")
	fs.Parse(args)

	srv := web.New(web.Options{IndexPath: *idxPath, Root: *root, WeightsPath: defaultWeights()})
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
