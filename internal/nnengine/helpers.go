package nnengine

import (
	"path/filepath"
)

// min returns the smaller of two ints.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// rel is a no-op kept for parity with callers that used filepath.Join with a
// root; it is unused but documents intent.
var _ = filepath.Join
