package main

import (
	"errors"
	"fmt"
	"os"

	"golang.org/x/image/font"
	"golang.org/x/image/font/opentype"
)

// fontLib wraps a parsed font so faces can be created at arbitrary pixel sizes.
type fontLib struct {
	parsed *opentype.Font
}

// loadFont opens the first usable system font from fontCandidates.
func loadFont() (*fontLib, error) {
	var lastErr error
	for _, p := range fontCandidates {
		data, err := os.ReadFile(p)
		if err != nil {
			lastErr = err
			continue
		}
		var f *opentype.Font
		if col, err := opentype.ParseCollection(data); err == nil {
			if col.NumFonts() == 0 {
				lastErr = errors.New("empty collection: " + p)
				continue
			}
			f, err = col.Font(0)
			if err != nil {
				lastErr = err
				continue
			}
		} else if f, err = opentype.Parse(data); err != nil {
			lastErr = err
			continue
		}
		return &fontLib{parsed: f}, nil
	}
	return nil, fmt.Errorf("no usable system font (%v)", lastErr)
}

// faceAt returns a face that renders text at pixel height sizePx.
func (l *fontLib) faceAt(sizePx float64) (font.Face, error) {
	return opentype.NewFace(l.parsed, &opentype.FaceOptions{
		Size:    sizePx,
		DPI:     72, // 1px == 1 point at 72 DPI
		Hinting: font.HintingNone,
	})
}
