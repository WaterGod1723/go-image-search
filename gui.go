//go:build wails

// 桌面 GUI（Wails）后端绑定与入口。仅当以 -tags wails 构建时编译，
// 因此普通 `go build` / `go test` 与跨平台构建不受 WebView2/WebKit 依赖影响。
package main

import (
	"context"
	"embed"
	"encoding/base64"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
	"github.com/wailsapp/wails/v2/pkg/runtime"

	"go-image-search/internal/imageproc"
	"go-image-search/internal/web"
)

//go:embed all:frontend/dist
var guiAssets embed.FS

//go:embed weights_split.gob
var embeddedWeights []byte

// App 桌面 GUI 的后端绑定。通过内嵌 *web.Server 复用其全部操作方法，
// 前端通过 window.go.main.App.* 调用；原生文件对话框经 Pick* 绑定暴露。
type App struct {
	*web.Server
	ctx context.Context

	convMu     sync.Mutex
	convNextID atomic.Int64
	convParts  map[string]chan convResult
	convTO     time.Duration
}

// convResult webview 转换结果。
type convResult struct {
	png []byte
	err error
}

// NewApp 创建桌面应用后端。
func NewApp() *App {
	srv := web.New(web.Options{
		IndexPath:   defaultIndexPath(),
		Root:        ".",
		Writer:      io.Discard,
		WeightsPath: defaultWeightsPath(),
	})
	return &App{Server: srv, convParts: make(map[string]chan convResult), convTO: 30 * time.Second}
}

// defaultWeightsPath 返回神经网络权重文件路径：优先 NN_WEIGHTS 环境变量，
// 其次程序所在目录 / 当前目录的 weights_split.gob，最后用户配置目录。若
// 外部均不存在，则将内嵌的 weights_split.gob 解出到用户配置目录并返回该路径。
func defaultWeightsPath() string {
	if v := os.Getenv("NN_WEIGHTS"); v != "" {
		return v
	}
	candidates := []string{"weights_split.gob"}
	if exe, err := os.Executable(); err == nil {
		candidates = append([]string{filepath.Join(filepath.Dir(exe), "weights_split.gob")}, candidates...)
	}
	if dir, err := os.UserConfigDir(); err == nil && dir != "" {
		candidates = append(candidates, filepath.Join(dir, "go-image-search", "weights_split.gob"))
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	if dir, err := os.UserConfigDir(); err == nil && dir != "" {
		dir = filepath.Join(dir, "go-image-search")
		if err := os.MkdirAll(dir, 0o755); err == nil {
			p := filepath.Join(dir, "weights_split.gob")
			if _, err := os.Stat(p); err != nil {
				if err := os.WriteFile(p, embeddedWeights, 0o644); err == nil {
					return p
				}
			} else {
				return p
			}
		}
	}
	return "weights_split.gob"
}

// defaultIndexPath 返回跨平台可写的默认索引路径（用户配置目录）。
func defaultIndexPath() string {
	dir, err := os.UserConfigDir()
	if err != nil || dir == "" {
		dir = "."
	}
	dir = filepath.Join(dir, "go-image-search")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "index.bin"
	}
	return filepath.Join(dir, "index.bin")
}

// Startup 保存 Wails 上下文并尽力加载默认索引，不阻塞窗口。
func (a *App) Startup(ctx context.Context) {
	a.ctx = ctx
	// 注入 webview 格式转换器：Go 原生解码失败（avif/svg 等）时交由前端
	// 多线程 Worker 转换为 PNG 后回传。
	imageproc.SetConverter(a.webviewConvert)
	_ = a.LoadIndex()
}

// webviewConvert 将任意图片字节交给前端 Worker 池转换为 PNG 字节。
// 通过事件 img:convert 派发，前端转换完成后调用 Converted/ConvertError 回传。
func (a *App) webviewConvert(data []byte, format string) ([]byte, error) {
	id := a.convNextID.Add(1) - 1
	tag := fmt.Sprint(id)
	ch := make(chan convResult, 1)
	a.convMu.Lock()
	a.convParts[tag] = ch
	a.convMu.Unlock()
	defer func() {
		a.convMu.Lock()
		delete(a.convParts, tag)
		a.convMu.Unlock()
	}()
	runtime.EventsEmit(a.ctx, "img:convert", map[string]any{
		"id":     tag,
		"format": format,
		"data":   base64.StdEncoding.EncodeToString(data),
	})
	select {
	case r := <-ch:
		return r.png, r.err
	case <-time.After(a.convTO):
		return nil, fmt.Errorf("webview 转换超时 (%s)", format)
	}
}

// Converted 前端经 Worker 转换完成后回传 PNG（base64）。
func (a *App) Converted(id string, pngBase64 string) {
	a.deliverConvert(id, func() ([]byte, error) {
		if pngBase64 == "" {
			return nil, fmt.Errorf("空转换结果")
		}
		png, err := base64.StdEncoding.DecodeString(pngBase64)
		if err != nil {
			return nil, fmt.Errorf("解码转换结果: %v", err)
		}
		return png, nil
	})
}

// ConvertError 前端上报转换失败。
func (a *App) ConvertError(id, msg string) {
	a.deliverConvert(id, func() ([]byte, error) { return nil, fmt.Errorf("%s", msg) })
}

func (a *App) deliverConvert(id string, fn func() ([]byte, error)) {
	a.convMu.Lock()
	ch, ok := a.convParts[id]
	if ok {
		delete(a.convParts, id)
	}
	a.convMu.Unlock()
	if !ok {
		return
	}
	png, err := fn()
	ch <- convResult{png: png, err: err}
}

// PickImage 弹出原生"选择图片"对话框，返回文件路径（取消时为空字符串）。
func (a *App) PickImage() (string, error) {
	return runtime.OpenFileDialog(a.ctx, runtime.OpenDialogOptions{
		Title:   "选择图片",
		Filters: []runtime.FileFilter{{DisplayName: "图片", Pattern: "*.png;*.jpg;*.jpeg;*.gif;*.webp;*.bmp;*.tif;*.tiff;*.avif;*.svg"}},
	})
}

// PickDirectory 弹出原生"选择目录"对话框，返回目录路径（取消时为空字符串）。
func (a *App) PickDirectory() (string, error) {
	return runtime.OpenDirectoryDialog(a.ctx, runtime.OpenDialogOptions{
		Title: "选择目录",
	})
}

// PickIndex 弹出原生"选择索引文件"对话框，返回文件路径（取消时为空字符串）。
func (a *App) PickIndex() (string, error) {
	return runtime.OpenFileDialog(a.ctx, runtime.OpenDialogOptions{
		Title:   "选择索引文件",
		Filters: []runtime.FileFilter{{DisplayName: "索引", Pattern: "*.bin;*.sczl"}},
	})
}

// runGUI 启动 Wails 桌面界面。
func runGUI() {
	app := NewApp()
	err := wails.Run(&options.App{
		Title:     "图片搜索",
		Width:     1280,
		Height:    840,
		MinWidth:  960,
		MinHeight: 640,
		AssetServer: &assetserver.Options{
			Assets: guiAssets,
		},
		BackgroundColour: &options.RGBA{R: 255, G: 143, B: 176, A: 1},
		OnStartup:        app.Startup,
		Bind:             []interface{}{app},
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "GUI 启动失败: %v\n", err)
		os.Exit(1)
	}
}
