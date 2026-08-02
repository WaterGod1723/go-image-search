//go:build windows

package main

import "syscall"

func init() {
	k32 := syscall.NewLazyDLL("kernel32.dll")
	if p := k32.NewProc("SetConsoleOutputCP"); p != nil {
		p.Call(65001)
	}
	if p := k32.NewProc("SetConsoleCP"); p != nil {
		p.Call(65001)
	}
}
