package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"go-image-search/internal/imageproc"
	"go-image-search/internal/sczl"
)

// sczl_cli.go SCZL 算法的 CLI 子命令：sczl-build / sczl-query。
// 与默认的区域感知哈希方案并行，互不影响。

func runSCZLBuild(args []string) {
	fs := flag.NewFlagSet("sczl-build", flag.ExitOnError)
	dir := fs.String("dir", "", "图像库目录")
	out := fs.String("out", "sczl.bin", "输出索引文件")
	jobs := fs.Int("jobs", 0, "并发构建线程数(0=CPU 核数)")
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

	ix := sczl.New()
	added := ix.BuildParallel(files, *jobs, func(f string) (sczl.Descriptor, error) {
		img, err := imageproc.Load(f)
		if err != nil {
			return sczl.Descriptor{}, err
		}
		return sczl.Extract(img), nil
	}, func(p sczl.BuildProgress) {
		if p.Err != nil {
			fmt.Fprintf(os.Stderr, "[跳过] %s: %v\n", p.File, p.Err)
			return
		}
		fmt.Printf("[%d/%d] %s\n", p.Done, len(files), p.ID)
	})

	if err := ix.Save(*out); err != nil {
		fmt.Fprintf(os.Stderr, "保存索引失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("完成: %d 张图像, %d 条描述子 -> %s\n", added, ix.Len(), *out)
}

func runSCZLQuery(args []string) {
	fs := flag.NewFlagSet("sczl-query", flag.ExitOnError)
	idxPath := fs.String("index", "sczl.bin", "索引文件")
	q := fs.String("q", "", "查询图像")
	top := fs.Int("top", 5, "返回 top-N")
	prefilter := fs.Int("prefilter", 200, "FD 全局粗排保留候选数")
	fdCut := fs.Float64("fd-cut", 0.4, "FD 相似度截断(低于此值淘汰)")
	scPts := fs.Int("sc-points", 48, "SC 匈牙利匹配每侧点数上限")
	fs.Parse(args)

	if *q == "" {
		fmt.Fprintln(os.Stderr, "缺少 -q")
		os.Exit(2)
	}
	ix, err := sczl.Load(*idxPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "加载索引失败: %v\n", err)
		os.Exit(1)
	}
	img, err := imageproc.Load(*q)
	if err != nil {
		fmt.Fprintf(os.Stderr, "加载查询图像失败: %v\n", err)
		os.Exit(1)
	}
	qd := sczl.Extract(img)
	if !qd.Valid {
		fmt.Fprintln(os.Stderr, "无法从查询图像提取前景")
		os.Exit(1)
	}
	matches := ix.Query(qd, sczl.Options{
		TopK:        *top,
		FDPreFilter: *prefilter,
		FDCut:       *fdCut,
		SCMaxPoints: *scPts,
	})
	if len(matches) == 0 {
		fmt.Println("无匹配结果")
		return
	}
	for i, m := range matches {
		fmt.Printf("#%d score=%.4f fd=%.4f sc=%.4f lbp=%.4f %s\n",
			i+1, m.Score, m.FDSim, m.SCSim, m.LBPSim, base(m.ImageID))
	}
}

func base(p string) string {
	i := strings.LastIndex(p, "/")
	if i < 0 {
		return p
	}
	return p[i+1:]
}
