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

// drawEdgeText adds a random number of short texts near the border of the
// canvas. Text height never exceeds 0.3 * canvas height. Top/bottom texts are
// horizontal; left/right texts are rotated 90 degrees. sprite is the screen
// bbox [x0,y0,x1,y1] of the drawn icon: a side is skipped when its text would
// overlap the icon (thin gray glyphs must stay fully readable).
func drawEdgeText(rng *rand.Rand, lib *fontLib, canvas *image.NRGBA, all bool, sprite [4]float64) []TextOut {
	w, h := canvas.Bounds().Dx(), canvas.Bounds().Dy()
	var out []TextOut

	sides := []string{"top", "bottom", "left", "right"}
	if !all {
		rng.Shuffle(len(sides), func(i, j int) { sides[i], sides[j] = sides[j], sides[i] })
		n := rng.Intn(len(sides) + 1) // 0..4 sides
		sides = sides[:n]
	}

	for _, side := range sides {
		size := int(float64(h) * (0.10 + rng.Float64()*0.18)) // 10%..28% of H
		if size < 12 {
			size = 12
		}
		if size > h*3/10 {
			size = h * 3 / 10 // hard cap: 0.3*H
		}
		word := fontWord(rng)
		tc := textColor(rng)

		f, err := lib.faceAt(float64(size))
		if err != nil {
			continue
		}
		adv := int(font.MeasureString(f, word) >> 6) // advance width in px

		var spec TextOut
		spec.Text = word
		spec.Size = size
		spec.Color = hexColorFmt(tc)
		switch side {
		case "top":
			// vertical band occupied by the text: [0, size]. Skip if the icon
			// intrudes into it.
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

// fontWord returns a short random word, sometimes with a numeric suffix.
func fontWord(rng *rand.Rand) string {
	w := fontWords[rng.Intn(len(fontWords))]
	if rng.Intn(2) == 0 {
		return fmt.Sprintf("%s%d", w, rng.Intn(1000))
	}
	return w
}

// textColor picks a legible ink color for the (light/dark) background.
func textColor(rng *rand.Rand) color.NRGBA {
	lum := rng.Intn(5) // 0..4
	switch lum {
	case 0, 1:
		return color.NRGBA{20, 20, 20, 255}
	default:
		return color.NRGBA{255, 255, 255, 255}
	}
}

func hexColorFmt(c color.NRGBA) string {
	return fmt.Sprintf("#%02x%02x%02x", c.R, c.G, c.B)
}

// textDrawH draws word with baseline at (x, y) into canvas.
func textDrawH(canvas *image.NRGBA, f font.Face, c color.NRGBA, word string, x, y int) {
	dr := &font.Drawer{
		Dst:  canvas,
		Src:  image.NewUniform(c),
		Face: f,
		Dot:  fixed.P(x, y),
	}
	dr.DrawString(word)
}

// textDrawV draws word vertically (rotated 90°) around the corner (x, y).
// cw==true rotates clockwise (right edge), otherwise counter-clockwise.
func textDrawV(dst *image.NRGBA, f font.Face, c color.NRGBA, word string, x, y int, cw bool) {
	adv := int(font.MeasureString(f, word) >> 6)
	m := f.Metrics()
	glyph := int(m.Ascent) + int(m.Descent)
	if adv < 1 || glyph < 1 {
		return
	}
	// offscreen horizontal layer big enough to hold the word
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

	// rotate 90 deg: dst(x,y)=layer(y, lw-1-x) for cw, or (lh-1-y, x) for ccw
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
