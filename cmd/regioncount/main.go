// 临时调试程序：输出指定图像在各阶段（segment / merge / gravity）的区域数。
// 用法: go run ./cmd/regioncount <图像路径>
package main

import (
	"fmt"
	"os"
	"path/filepath"

	"go-image-search/internal/imageproc"
	"go-image-search/internal/segment"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "用法: regioncount <图像路径>")
		os.Exit(2)
	}
	path, err := filepath.Abs(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	img, err := imageproc.Load(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "加载失败: %v\n", err)
		os.Exit(1)
	}

	cfg := segment.DefaultConfig()
	mc := segment.DefaultMergeConfig()
	gc := segment.DefaultGravityConfig()

	// 对每个衍生图（原图 + 骨架）分别统计
	for vi, v := range imageproc.QueryVariants(img) {
		name := "原图"
		if vi == 1 {
			name = "骨架"
		}
		res, err := segment.Segment(v, cfg)
		if err != nil {
			fmt.Printf("[%s] segment 失败: %v\n", name, err)
			continue
		}
		nSeg := len(res.Regions)

		infos := make([]segment.RegionInfo, 0, nSeg)
		for _, reg := range res.Regions {
			crop := res.Crop(v, reg.ID)
			if crop == nil {
				continue
			}
			infos = append(infos, segment.RegionInfo{
				ID: reg.ID, Area: reg.Area, Color: reg.MeanColor, BBox: reg.BBox,
			})
		}
		merged := segment.MergeSimilar(v, res, infos, mc)
		nMerged := len(merged)

		// 引力聚合（Additive 模式：返回辅助组合区域数）
		gravAux := segment.GravityMerge(v, merged, gc)
		// replace 模式下的最终区域数
		gcReplace := gc
		gcReplace.Additive = false
		gravReplace := segment.GravityMerge(v, merged, gcReplace)
		nGravReplace := len(gravReplace)

		fmt.Printf("[%s] segment=%d  merge=%d  gravity辅助(additive)=%d  gravity替换后=%d  索引入库总数(merge+辅助)=%d\n",
			name, nSeg, nMerged, len(gravAux), nGravReplace, nMerged+len(gravAux))
	}
}
