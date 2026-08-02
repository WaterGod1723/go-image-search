package imageproc

import (
	"bytes"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"
)

// Load 读取图片文件并解码为 image.Image，支持 png/jpeg/gif。
func Load(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("decode %s: %w", path, err)
	}
	return img, nil
}

// LoadSupported 扫描目录，返回所有受支持图片文件的绝对路径（排序稳定）。
func LoadSupported(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var files []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := strings.ToLower(e.Name())
		if strings.HasSuffix(name, ".png") || strings.HasSuffix(name, ".jpg") ||
			strings.HasSuffix(name, ".jpeg") || strings.HasSuffix(name, ".gif") {
			files = append(files, filepath.Join(dir, e.Name()))
		}
	}
	return files, nil
}

// EncodePNG 将图片编码为 PNG 字节，用于可视化调试。
func EncodePNG(img image.Image) ([]byte, error) {
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
