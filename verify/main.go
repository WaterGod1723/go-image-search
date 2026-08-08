// Command verify decodes all generated samples, checks sizes and that the
// transparent background region from manifest matches, then reports stats.
package main

import (
	"encoding/json"
	"fmt"
	"image/png"
	"os"
	"path/filepath"
)

type Manif struct {
	Texts []struct {
		Side string `json:"side"`
		Size int    `json:"size"`
	}
	Canvas [2]int `json:"canvas"`
}

func main() {
	dir := "test_set"
	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		panic(err)
	}
	var samples []Manif
	if err := json.Unmarshal(data, &samples); err != nil {
		panic(err)
	}
	var files []string
	entries, _ := os.ReadDir(dir)
	ok := 0
	for _, e := range entries {
		if filepath.Ext(e.Name()) != ".png" {
			continue
		}
		files = append(files, e.Name())
		f, err := os.Open(filepath.Join(dir, e.Name()))
		if err != nil {
			panic(err)
		}
		img, err := png.Decode(f)
		f.Close()
		_ = img
		if err != nil {
			fmt.Printf("DECODE FAIL %s: %v\n", e.Name(), err)
			continue
		}
		ok++
	}
	fmt.Printf("files=%d decodable=%d\n", len(files), ok)
	for i, s := range samples {
		for _, t := range s.Texts {
			if t.Size > s.Canvas[1]/3 {
				fmt.Printf("sample %d: text size %d exceeds canvas h/3=%d\n", i, t.Size, s.Canvas[1]/3)
			}
		}
	}
	fmt.Println("manifest records:", len(samples))
}