package imageproc

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDecodeAnyNativeBMP(t *testing.T) {
	img, err := DecodeAny(mustPngBytes(t, 8, 6), "png")
	if err != nil {
		t.Fatalf("png decode: %v", err)
	}
	if b := img.Bounds(); b.Dx() != 8 || b.Dy() != 6 {
		t.Fatalf("unexpected bounds: %v", b)
	}
}

func TestDecodeAnyConverterFallback(t *testing.T) {
	// SVG 无法原生解码，应走注入的 Converter 转换后还原。
	var gotFormat string
	SetConverter(func(data []byte, format string) ([]byte, error) {
		gotFormat = format
		return mustPngBytes(t, 4, 4), nil
	})
	defer SetConverter(nil)

	svg := []byte(`<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"></svg>`)
	// 从文件路径加载时 formatHint 带点（".svg"），转换器应收到规范化的 "svg"。
	img, err := DecodeAny(svg, ".svg")
	if err != nil {
		t.Fatalf("decode via converter: %v", err)
	}
	if gotFormat != "svg" {
		t.Fatalf("expected format svg, got %q", gotFormat)
	}
	if b := img.Bounds(); b.Dx() != 4 || b.Dy() != 4 {
		t.Fatalf("unexpected bounds after convert: %v", b)
	}
}

func TestDecodeAnyConverterNilFails(t *testing.T) {
	// 无 converter 时，svg/avif 应报错而非 panic。
	SetConverter(nil)
	svg := []byte(`<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"></svg>`)
	if _, err := DecodeAny(svg, "svg"); err == nil {
		t.Fatal("expected error without converter")
	}
}

func TestDetectFormat(t *testing.T) {
	cases := map[string]string{
		"<svg xmlns=...></svg>": "svg",
		"<?xml version><svg></svg>": "svg",
		"RIFFxxxxWEBPVP8 ":     "webp",
		"BM":                   "bmp",
	}
	for data, want := range cases {
		if got := detectFormat([]byte(data)); got != want {
			t.Errorf("detectFormat(%q) = %q, want %q", data, got, want)
		}
	}
}

func TestLoadSupportedExtensions(t *testing.T) {
	dir := t.TempDir()
	exts := []string{".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".avif", ".svg", ".txt"}
	for _, e := range exts {
		if err := os.WriteFile(filepath.Join(dir, "a"+e), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	files, err := LoadSupported(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != len(exts)-1 { // 排除 .txt
		t.Fatalf("expected %d supported, got %d: %v", len(exts)-1, len(files), files)
	}
}

func TestLoadSupportedRecursive(t *testing.T) {
	root := t.TempDir()
	// 顶层 + 多级子目录 + 隐藏目录
	dirs := []string{".git", "sub1", "sub1/deep", "sub2"}
	for _, d := range dirs {
		if err := os.MkdirAll(filepath.Join(root, d), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	// 各目录放一张 png（.git 里的应被跳过）
	places := []string{".", "sub1", "sub1/deep", "sub2", ".git"}
	for i, p := range places {
		name := fmt.Sprintf("img%d.png", i)
		if err := os.WriteFile(filepath.Join(root, p, name), mustPngBytes(t, 2, 2), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// 顶层放一个非图片，确保被忽略
	if err := os.WriteFile(filepath.Join(root, "note.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	files, err := LoadSupported(root)
	if err != nil {
		t.Fatal(err)
	}
	// 期望 4 张：顶层 + sub1 + sub1/deep + sub2（.git 里的被跳过）
	if len(files) != 4 {
		t.Fatalf("expected 4 images (recursive, hidden skipped), got %d: %v", len(files), files)
	}
	for _, f := range files {
		if strings.Contains(filepath.ToSlash(f), "/.git/") {
			t.Errorf("hidden dir should be skipped, got %s", f)
		}
	}
}

func mustPngBytes(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	if w > 0 && h > 0 {
		img.Set(0, 0, image.White)
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}