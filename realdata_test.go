package main

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"go-image-search/internal/imageproc"
	"go-image-search/internal/sczl"
)

// TestRealDatasetSearch 使用真实测试图片做端到端 SCZL 检索：
//   - 索引库: test_pngs
//   - 查询图: test_pngs_target/TEST<N>_FROM_<源图名>.png（为原图裁剪/缩放得到的局部视图）
//
// 断言：期望图必须命中 top-3。默认启用自适应权重（基于各维度得分分布动态调整，
// 而非写死固定权重）。
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

	ix := sczl.New()
	for _, f := range files {
		img, err := imageproc.Load(f)
		if err != nil {
			// 纯 Go 测试环境无 webview 转换器，svg/avif 等格式无法解码：
			// 与构建流水线一致，跳过而非中断。
			t.Logf("[跳过] 无法解码 %s: %v", f, err)
			continue
		}
		d := sczl.Extract(img)
		if !d.Valid {
			t.Logf("[跳过] 无效描述子: %s", f)
			continue
		}
		ix.AddImage(f, d)
	}

	qfiles, err := filepath.Glob(filepath.Join(qDir, "*.png"))
	if err != nil || len(qfiles) == 0 {
		t.Fatalf("查询目录 %s 中无图片", qDir)
	}

	opts := sczl.DefaultOptions() // 默认 Adaptive=true，启用自适应权重
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

		img, err := imageproc.Load(qf)
		if err != nil {
			t.Errorf("[%s] 加载失败: %v", filepath.Base(qf), err)
			continue
		}
		qd := sczl.Extract(img)
		if !qd.Valid {
			t.Errorf("[%s] 无法提取前景", filepath.Base(qf))
			continue
		}
		matches := ix.Query(qd, opts)

		gotRank, gotScore := -1, 0.0
		for i, mm := range matches {
			if filepath.Base(mm.ImageID) == want {
				gotRank, gotScore = i, mm.Score
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
