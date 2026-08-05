//go:build wails

// 桌面 GUI（Wails）后端绑定与入口。仅当以 -tags wails 构建时编译，
// 因此普通 `go build` / `go test` 与跨平台构建不受 WebView2/WebKit 依赖影响。
package main

import (
	"context"
	"embed"
	"fmt"
	"os"

	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
	"github.com/wailsapp/wails/v2/pkg/runtime"

	"go-image-search/internal/web"
)

//go:embed all:frontend/dist
var guiAssets embed.FS

// App 桌面 GUI 的后端绑定。通过内嵌 *web.Server 复用其全部操作方法，
// 前端通过 window.go.main.App.* 调用；原生文件对话框经 Pick* 绑定暴露。
type App struct {
	*web.Server
	ctx context.Context
}

// NewApp 创建桌面应用后端。
func NewApp() *App {
	srv := web.New(web.Options{IndexPath: "index.bin", Root: "."})
	return &App{Server: srv}
}

// Startup 保存 Wails 上下文并尽力加载默认索引，不阻塞窗口。
func (a *App) Startup(ctx context.Context) {
	a.ctx = ctx
	_ = a.LoadIndex()
}

// PickImage 弹出原生"选择图片"对话框，返回文件路径（取消时为空字符串）。
func (a *App) PickImage() (string, error) {
	return runtime.OpenFileDialog(a.ctx, runtime.OpenDialogOptions{
		Title:   "选择图片",
		Filters: []runtime.FileFilter{{DisplayName: "图片", Pattern: "*.png;*.jpg;*.jpeg;*.gif;*.webp;*.bmp"}},
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