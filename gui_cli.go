//go:build !wails

// 未以 -tags wails 构建时（CLI / 跨平台构建），无桌面 GUI 依赖。
// 不带参数时保持原有行为：打印用法并退出。
package main

import "os"

func runGUI() {
	usage()
	os.Exit(2)
}