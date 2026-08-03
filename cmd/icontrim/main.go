package main

import (
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"

	"go-image-search/internal/imageproc"
)

func trimBounds(img image.Image) image.Rectangle {
	b := img.Bounds()
	minX, minY := b.Max.X, b.Max.Y
	maxX, maxY := b.Min.X, b.Min.Y

	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			_, _, _, a := img.At(x, y).RGBA()
			if a > 0 {
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}

	if minX > maxX || minY > maxY {
		return image.Rect(0, 0, 0, 0)
	}
	return image.Rect(minX, minY, maxX+1, maxY+1)
}

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "用法: icontrim <输入目录> <输出目录>")
		os.Exit(2)
	}

	srcDir, err := filepath.Abs(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	dstDir, err := filepath.Abs(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	if err := os.MkdirAll(dstDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "创建输出目录失败: %v\n", err)
		os.Exit(1)
	}

	files, err := imageproc.LoadSupported(srcDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "扫描目录失败: %v\n", err)
		os.Exit(1)
	}

	trimmed, skipped := 0, 0
	for _, path := range files {
		img, err := imageproc.Load(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "加载失败 %s: %v\n", path, err)
			continue
		}

		rect := trimBounds(img)
		if rect.Empty() {
			fmt.Fprintf(os.Stderr, "跳过全透明: %s\n", filepath.Base(path))
			skipped++
			continue
		}

		if rect.Eq(img.Bounds()) {
			skipped++
			continue
		}

		cropped := img.(interface {
			SubImage(r image.Rectangle) image.Image
		}).SubImage(rect)

		outPath := filepath.Join(dstDir, filepath.Base(path))
		if !strings.HasSuffix(strings.ToLower(outPath), ".png") {
			outPath += ".png"
		}

		f, err := os.Create(outPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "创建失败 %s: %v\n", outPath, err)
			continue
		}
		if err := png.Encode(f, cropped); err != nil {
			fmt.Fprintf(os.Stderr, "编码失败 %s: %v\n", outPath, err)
			f.Close()
			continue
		}
		f.Close()
		trimmed++
	}

	fmt.Printf("完成: 裁剪 %d 张, 跳过 %d 张\n", trimmed, skipped)
}
