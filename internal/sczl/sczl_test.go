package sczl

import (
	"image"
	_ "image/png"
	"os"
	"path/filepath"
	"testing"
)

// sczl_test.go 基础冒烟测试：确保描述子提取、索引构建、检索、持久化端到端可用。

func mustLoad(t *testing.T, rel string) image.Image {
	t.Helper()
	paths := []string{
		rel,
		filepath.Join("../../", rel),
	}
	var lastErr error
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			lastErr = err
			continue
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			lastErr = err
			continue
		}
		return img
	}
	t.Skip("跳过：找不到测试图", lastErr)
	return nil
}

func TestExtractProducesValidDescriptor(t *testing.T) {
	img := mustLoad(t, "test_pngs/dashboard_76dp.png")
	d := Extract(img)
	if !d.Valid {
		t.Fatal("Extract 返回无效描述子")
	}
	if len(d.Occupancy32) != occBin32*occBin32 {
		t.Fatalf("Occupancy32 维度异常: %d", len(d.Occupancy32))
	}
	if len(d.Occupancy64) != normSize*normSize {
		t.Fatalf("Occupancy64 维度异常: %d", len(d.Occupancy64))
	}
	if len(d.Patch) != normSize*normSize {
		t.Fatalf("Patch 维度异常: %d", len(d.Patch))
	}
	if len(d.HOG) == 0 {
		t.Fatal("HOG 为空")
	}
	if len(d.Fourier) != fourierDims {
		t.Fatalf("Fourier 维度异常: %d", len(d.Fourier))
	}
	if len(d.SC) != contourSamples {
		t.Fatalf("SC 点数异常: %d", len(d.SC))
	}
}

func TestBuildQuerySaveLoad(t *testing.T) {
	img := mustLoad(t, "test_pngs/dashboard_76dp.png")
	ix := New()
	ix.AddImage("lib/dashboard_76dp.png", Extract(img))
	if ix.Len() != 1 {
		t.Fatalf("索引条目数异常: %d", ix.Len())
	}
	q := Extract(img)
	matches := ix.Query(q, DefaultOptions())
	if len(matches) == 0 {
		t.Fatal("Query 返回空结果")
	}
	if matches[0].ImageID != "lib/dashboard_76dp.png" {
		t.Fatalf("top1 ImageID 异常: %s", matches[0].ImageID)
	}

	// 持久化往返。
	path := filepath.Join(t.TempDir(), "sczl.bin")
	if err := ix.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	ix2, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if ix2.Len() != ix.Len() {
		t.Fatalf("加载后条目数不一致: %d vs %d", ix2.Len(), ix.Len())
	}
	matches2 := ix2.Query(q, DefaultOptions())
	if len(matches2) == 0 || matches2[0].ImageID != matches[0].ImageID {
		t.Fatalf("加载后检索结果不一致: %+v vs %+v", matches2, matches)
	}
}

// TestSelfMatchTop1 自检索：一张图检索自身必须 top-1 命中。
func TestSelfMatchTop1(t *testing.T) {
	img := mustLoad(t, "test_pngs/dashboard_76dp.png")
	ix := New()
	ix.AddImage("lib/dashboard_76dp.png", Extract(img))
	q := Extract(img)
	m := ix.Query(q, DefaultOptions())
	if len(m) == 0 || m[0].ImageID != "lib/dashboard_76dp.png" {
		t.Fatalf("自检索未命中: %+v", m)
	}
}
