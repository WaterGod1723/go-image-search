package main

import (
	"fmt"
	"image"
	"image/color"
	"math/rand"

	"golang.org/x/image/font"
	"golang.org/x/image/math/fixed"
)

// TextOut describes one text element drawn on a side of the canvas.
type TextOut struct {
	Side  string `json:"side"`
	Text  string `json:"text"`
	Size  int    `json:"size"`
	X     int    `json:"x"`
	Y     int    `json:"y"`
	Color string `json:"color"`
}

var fontWords = []string{
	"商店", "选购", "特价", "新品", "热卖", "爆款", "福利", "旗舰",
	"SALE", "HOT", "NEW", "FREE", "50", "99", "满减",
	"点击", "查看", "详情", "领券", "会员",
}

func drawEdgeText(rng *rand.Rand, lib *fontLib, canvas *image.NRGBA, all bool, sprite [4]float64) []TextOut {
	w, h := canvas.Bounds().Dx(), canvas.Bounds().Dy()
	var out []TextOut

	sides := []string{"top", "bottom", "left", "right"}
	if !all {
		rng.Shuffle(len(sides), func(i, j int) { sides[i], sides[j] = sides[j], sides[i] })
		n := rng.Intn(len(sides) + 1)
		sides = sides[:n]
	}

	for _, side := range sides {
		size := int(float64(h) * (0.10 + rng.Float64()*0.18))
		if size < 12 {
			size = 12
		}
		if size > h*3/10 {
			size = h * 3 / 10
		}
		word := fontWord(rng)
		tc := textColor(rng)

		f, err := lib.faceAt(float64(size))
		if err != nil {
			continue
		}
		adv := int(font.MeasureString(f, word) >> 6)

		var spec TextOut
		spec.Text = word
		spec.Size = size
		spec.Color = hexColor(tc)
		switch side {
		case "top":
			if sprite[1] < float64(size) {
				continue
			}
			spec.X = rng.Intn(maxi(1, w-adv))
			spec.Y = rng.Intn(maxi(1, size/3))
			textDrawH(canvas, f, tc, word, spec.X, spec.Y+size)
		case "bottom":
			if sprite[3] > float64(h)-float64(size) {
				continue
			}
			spec.X = rng.Intn(maxi(1, w-adv))
			spec.Y = h - size + rng.Intn(maxi(1, size/2))
			textDrawH(canvas, f, tc, word, spec.X, spec.Y+size)
		case "left":
			if sprite[0] < float64(size) {
				continue
			}
			spec.Y = rng.Intn(maxi(1, h-adv))
			spec.X = rng.Intn(maxi(1, size))
			textDrawV(canvas, f, tc, word, spec.X, spec.Y, false)
		case "right":
			if sprite[2] > float64(w)-float64(size) {
				continue
			}
			spec.Y = rng.Intn(maxi(1, h-adv))
			spec.X = w - size + rng.Intn(maxi(1, size))
			textDrawV(canvas, f, tc, word, spec.X, spec.Y, true)
		}
		spec.Side = side
		out = append(out, spec)
	}
	return out
}

func fontWord(rng *rand.Rand) string {
	w := fontWords[rng.Intn(len(fontWords))]
	if rng.Intn(2) == 0 {
		return fmt.Sprintf("%s%d", w, rng.Intn(1000))
	}
	return w
}

func textColor(rng *rand.Rand) color.NRGBA {
	lum := rng.Intn(5)
	switch lum {
	case 0, 1:
		return color.NRGBA{20, 20, 20, 255}
	default:
		return color.NRGBA{255, 255, 255, 255}
	}
}

func textDrawH(canvas *image.NRGBA, f font.Face, c color.NRGBA, word string, x, y int) {
	dr := &font.Drawer{
		Dst:  canvas,
		Src:  image.NewUniform(c),
		Face: f,
		Dot:  fixed.P(x, y),
	}
	dr.DrawString(word)
}

func textDrawV(dst *image.NRGBA, f font.Face, c color.NRGBA, word string, x, y int, cw bool) {
	adv := int(font.MeasureString(f, word) >> 6)
	m := f.Metrics()
	glyph := int(m.Ascent) + int(m.Descent)
	if adv < 1 || glyph < 1 {
		return
	}
	lw := adv + 2
	lh := glyph + 2
	layer := image.NewNRGBA(image.Rect(0, 0, lw, lh))
	dr := &font.Drawer{
		Dst:  layer,
		Src:  image.NewUniform(c),
		Face: f,
		Dot:  fixed.P(1, int(m.Ascent)+1),
	}
	dr.DrawString(word)
	for sy := 0; sy < lh; sy++ {
		for sx := 0; sx < lw; sx++ {
			c2 := layer.NRGBAAt(sx, sy)
			if c2.A == 0 {
				continue
			}
			var dx, dy int
			if cw {
				dx, dy = x+sy, y+(lw-1-sx)
			} else {
				dx, dy = x+(lh-1-sy), y+sx
			}
			if dx >= 0 && dy >= 0 && dx < dst.Bounds().Dx() && dy < dst.Bounds().Dy() {
				dst.SetNRGBA(dx, dy, c)
			}
		}
	}
}
