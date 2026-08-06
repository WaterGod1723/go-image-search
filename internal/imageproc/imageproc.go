package imageproc

import (
	"bytes"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"
)

// supportedExts 支持扫描与解码的图片扩展名（小写）。
var supportedExts = map[string]bool{
	".png": true, ".jpg": true, ".jpeg": true, ".gif": true,
	".webp": true, ".bmp": true, ".tif": true, ".tiff": true,
	".avif": true, ".svg": true,
}

// Converter 将任意图片字节转换为 PNG 字节。data 为原始文件内容，format 为
// 探测出的格式（小写扩展名，如 "webp"/"svg"/"avif"，未知为空串）。
// Go 原生解码失败的格式（avif/svg 等）会交给它处理；实现上通常复用 webview
// 的多线程 Web Worker 完成，避免引入原生（cgo）解码依赖。
type Converter func(data []byte, format string) ([]byte, error)

var (
	convMu   sync.RWMutex
	converter Converter
)

// SetConverter 注入 webview 格式转换器。仅在有能力转换的环境中调用一次（如
// Wails GUI 启动时）；nil 表示禁用（CLI/纯 Go 环境仅用原生+x/image 解码）。
func SetConverter(c Converter) {
	convMu.Lock()
	defer convMu.Unlock()
	converter = c
}

// Load 读取图片文件并解码为 image.Image。优先原生解码（png/jpeg/gif/bmp/tiff/
// webp），失败时若已注入 Converter 则经由其转换后再解码。
func Load(path string) (image.Image, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return DecodeAny(data, extOf(path))
}

// LoadAny 与 Load 等价，便于语义区分（按扩展名加速探测）。
func LoadAny(path string) (image.Image, error) {
	return Load(path)
}

// DecodeAny 从字节解码任意受支持格式。formatHint 为可选的扩展名（小写，可空）。
func DecodeAny(data []byte, formatHint string) (image.Image, error) {
	img, err := decodeNative(data)
	if err == nil {
		return img, nil
	}
	convMu.RLock()
	c := converter
	convMu.RUnlock()
	if c == nil {
		return nil, err
	}
	format := strings.TrimPrefix(formatHint, ".")
	if format == "" {
		format = detectFormat(data)
	}
	pngBytes, cerr := c(data, format)
	if cerr != nil {
		return nil, fmt.Errorf("decode %s: %w (webview convert: %v)", format, err, cerr)
	}
	img, err = png.Decode(bytes.NewReader(pngBytes))
	if err != nil {
		return nil, fmt.Errorf("decode webview png: %w", err)
	}
	return img, nil
}

// decodeNative 用注册的格式解码器解码（png/jpeg/gif/bmp/tiff/webp）。
func decodeNative(data []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return img, nil
}

// extOf 返回扩展名（小写，含点），无扩展名时为空串。
func extOf(path string) string {
	return strings.ToLower(filepath.Ext(path))
}

// detectFormat 通过魔数探测无法原生解码的格式（svg/avif/webp 等）。
func detectFormat(data []byte) string {
	head := data
	if len(head) > 1024 {
		head = head[:1024]
	}
	trimmed := bytes.TrimLeft(head, " \t\r\n\xef\xbb\xbf")
	lower := bytes.ToLower(trimmed)
	switch {
	case len(lower) > 4 && string(lower[:4]) == "riff" && len(lower) > 12 &&
		string(lower[8:12]) == "webp":
		return "webp"
	case len(lower) > 8 && string(lower[4:8]) == "ftyp" &&
		(string(lower[8:12]) == "avif" || string(lower[8:12]) == "avis"):
		return "avif"
	case bytes.Contains(lower, []byte("<svg")) || bytes.Contains(lower, []byte("<!doctype svg")):
		return "svg"
	case len(lower) > 1 && string(lower[:2]) == "bm":
		return "bmp"
	}
	return ""
}

// LoadSupported 递归扫描目录及其全部子目录，返回所有受支持图片文件的路径
// （按词法序稳定排列）。跳过以 "." 开头的隐藏目录（如 .git），避免误入库元数据。
func LoadSupported(dir string) ([]string, error) {
	var files []string
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if path != dir && strings.HasPrefix(d.Name(), ".") {
				return fs.SkipDir
			}
			return nil
		}
		if supportedExts[extOf(d.Name())] {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
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