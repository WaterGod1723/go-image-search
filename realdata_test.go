package main

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"go-image-search/internal/imageproc"
	"go-image-search/internal/index"
	"go-image-search/internal/segment"
)

// TestRealDatasetSearch 使用真实测试图片做端到端检索：
//   - 索引库: test_pngs
//   - 查询图: test_pngs_target/TEST<N>_FROM_<源图名>.png（为原图裁剪/缩放得到的局部视图）
//
// 断言：期望图必须命中 top-3。裁剪过狠或算法不足的用例会如实失败，用于驱动算法优化。
func TestRealDatasetSearch(t *testing.T) {
	const (
		libDir = "test_pngs"
		qDir   = "test_pngs_target"
	)

	files, err := imageproc.LoadSupported(libDir)
	if err != nil {
		t.Fatalf("扫描索引库失败: %v", err)
	}
	if len(files) == 0 {
		t.Fatalf("索引库 %s 中无支持的图片", libDir)
	}

	ix := index.New()
	cfg := segment.DefaultConfig()
	for _, f := range files {
		img, err := loadTestImage(f)
		if err != nil {
			t.Fatalf("加载索引图片失败 %s: %v", f, err)
		}
		// 原图 + 骨架图分别入索引（与构建一致）。
		// 索引阶段启用引力聚合：区域过多时聚合出辅助组合区域一并入索引，
		// 与查询阶段的辅助查询区域对应，提升碎片化图像的召回。
		for _, v := range imageproc.QueryVariants(img) {
			hashes, err := hashImage(v, cfg, segment.DefaultMergeConfig(), segment.DefaultGravityConfig())
			if err != nil {
				t.Fatalf("索引图片处理失败 %s: %v", f, err)
			}
			ix.AddImage(f, hashes)
		}
	}

	qfiles, err := filepath.Glob(filepath.Join(qDir, "*.png"))
	if err != nil || len(qfiles) == 0 {
		t.Fatalf("查询目录 %s 中无图片", qDir)
	}

	re := regexp.MustCompile(`^TEST\d+_FROM_(.+?)\.png$`)
	pass, total := 0, 0
	for _, qf := range qfiles {
		m := re.FindStringSubmatch(filepath.Base(qf))
		if m == nil {
			t.Logf("[跳过] 无法从文件名解析期望目标: %s", filepath.Base(qf))
			continue
		}
		want := normalizeLibName(m[1]) + ".png"
		total++

		img, err := loadTestImage(qf)
		if err != nil {
			t.Errorf("[%s] 加载失败: %v", filepath.Base(qf), err)
			continue
		}
		// 生成 原图 + 骨架图 两组查询区域，用 SearchMulti 合并（与查询流程一致）。
		// 查询阶段启用引力聚合：区域过多时按引力模型聚合出辅助组合区域参与检索。
		querySets := make([][]index.QueryRegion, 0, 2)
		for _, v := range imageproc.QueryVariants(img) {
			hashes, err := hashImage(v, cfg, segment.DefaultMergeConfig(), segment.DefaultGravityConfig())
			if err != nil {
				t.Errorf("[%s] 处理失败: %v", filepath.Base(qf), err)
				continue
			}
			var qs []index.QueryRegion
			for _, h := range hashes {
				qs = append(qs, index.QueryRegion{
					Hash: h.Hash, Shape: h.Shape, Area: h.Area, Color: h.Color,
					NX: h.NX, NY: h.NY, Fill: h.Fill, Aspect: h.Aspect, Global: h.Global,
				})
			}
			querySets = append(querySets, qs)
		}
		matches := ix.SearchMulti(querySets, index.SearchOptions{MaxDist: 12, ColorWeight: 0.1})

		gotRank, gotScore := -1, 0.0
		for i, m := range matches {
			if filepath.Base(m.ImageID) == want {
				gotRank, gotScore = i, m.Score
				break
			}
		}
		if gotRank >= 0 && gotRank < 3 {
			pass++
			t.Logf("[PASS] %s -> %s (rank=#%d, score=%.3f)",
				filepath.Base(qf), filepath.Base(matches[0].ImageID), gotRank+1, gotScore)
			continue
		}

		msg := fmt.Sprintf("[FAIL] %s 期望 top3=%s，实际未命中", filepath.Base(qf), want)
		if len(matches) > 0 {
			msg = fmt.Sprintf("[FAIL] %s 期望 top3=%s，实际 top1=%s(score=%.3f)",
				filepath.Base(qf), want, filepath.Base(matches[0].ImageID), matches[0].Score)
		}
		if gotRank > 0 {
			msg += fmt.Sprintf("，期望图排名 #%d(score=%.3f)", gotRank+1, gotScore)
		}
		t.Errorf("%s", msg)
	}

	if total == 0 {
		t.Fatal("未解析到任何测试用例")
	}
	t.Logf("汇总: %d/%d 命中 top-3", pass, total)
}

// normalizeLibName 规整文件名中 FROM_ 后的部分：
// TEST7_FROM_.grid_view.png 期望对应 test_pngs/grid_view.png。
func normalizeLibName(name string) string {
	return strings.TrimLeft(name, "._")
}
