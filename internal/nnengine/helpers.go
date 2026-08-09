package nnengine

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// min returns the smaller of two ints.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// listPNG returns the sorted .png file names in dir (engine's reference scan).
func listPNG(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if strings.HasSuffix(strings.ToLower(e.Name()), ".png") {
			out = append(out, e.Name())
		}
	}
	sort.Strings(out)
	return out, nil
}

// rel is a no-op kept for parity with callers that used filepath.Join with a
// root; it is unused but documents intent.
var _ = filepath.Join
