package iconnet

import (
	"math"
)

// Embed computes the L2-normalized embedding of a sprite tensor.
//
// x is in channel-major NCHW layout (order R,G,B,A), N = net.InputSize, values
// in [0,1]. Returns the embedding vector of length EmbDim, already normalized.
func (net *Net) Embed(x []float64) []float64 {
	n := net.InputSize
	prev := x
	h := n
	for i := range net.Block {
		b := &net.Block[i]
		prev = b.convReLUBN(prev, h, b.CIn, b.Conv1W, b.Conv1B, b.BN1G, b.BN1B, b.BN1M, b.BN1V)
		prev = b.convReLUBN(prev, h, b.COut, b.Conv2W, b.Conv2B, b.BN2G, b.BN2B, b.BN2M, b.BN2V)
		prev = b.se(prev, h)
		prev = maxPool2x2(prev, h, b.COut)
		h /= 2
	}
	// global average pool over h x h
	gap := make([]float64, net.Ch[len(net.Ch)-1])
	perCh := h * h
	for c := range gap {
		var s float64
		for k := 0; k < perCh; k++ {
			s += prev[c*perCh+k]
		}
		gap[c] = s / float64(perCh)
	}
	// linear projection
	emb := make([]float64, net.EmbDim)
	last := net.Ch[len(net.Ch)-1]
	for j := 0; j < net.EmbDim; j++ {
		var v float64
		row := net.EmbW[j*last : (j+1)*last]
		for i := 0; i < last; i++ {
			v += row[i] * gap[i]
		}
		emb[j] = v + net.EmbB[j]
	}
	// L2 normalize
	var nrm float64
	for _, v := range emb {
		nrm += v * v
	}
	if nrm > 0 {
		nrm = math.Sqrt(nrm)
		for j := range emb {
			emb[j] /= nrm
		}
	}
	return emb
}

// convReLUBN applies a 3x3 stride-1 pad-1 convolution followed by batch norm
// and ReLU. Layout is NCHW (channel-major), spatial size h. cin is the input
// channel count.
func (b *block) convReLUBN(in []float64, h, cin int, W, B, G, BB, M, V []float64) []float64 {
	out := make([]float64, b.COut*h*h)
	co := b.COut
	for fo := 0; fo < co; fo++ {
		g, bb, m, v := G[fo], BB[fo], M[fo], V[fo]
		scale := g / math.Sqrt(v+bnEps)
		shift := bb - m*scale
		for y := 0; y < h; y++ {
			for x := 0; x < h; x++ {
				var acc float64
				for fi := 0; fi < cin; fi++ {
					row := W[(fo*cin+fi)*9:]
					inCh := in[fi*h*h:]
					for dy := 0; dy < 3; dy++ {
						iy := y - 1 + dy
						if iy < 0 || iy >= h {
							continue
						}
						for dx := 0; dx < 3; dx++ {
							ix := x - 1 + dx
							if ix < 0 || ix >= h {
								continue
							}
							acc += row[dy*3+dx] * inCh[iy*h+ix]
						}
					}
				}
				v := acc + B[fo]
				v = v*scale + shift
				if v < 0 {
					v = 0
				}
				out[fo*h*h+y*h+x] = v
			}
		}
	}
	return out
}

// se applies squeeze-and-excitation channel attention (channel-major, h x h).
func (b *block) se(in []float64, h int) []float64 {
	c := b.COut
	per := h * h
	// global average over spatial
	gap := make([]float64, c)
	for k := 0; k < c; k++ {
		var s float64
		ch := in[k*per : (k+1)*per]
		for i := 0; i < per; i++ {
			s += ch[i]
		}
		gap[k] = s / float64(per)
	}
	// fc1: r x c, ReLU
	r := b.SER
	z := make([]float64, r)
	for j := 0; j < r; j++ {
		var v float64
		row := b.SE1W[j*c : (j+1)*c]
		for i := 0; i < c; i++ {
			v += row[i] * gap[i]
		}
		v += b.SE1B[j]
		if v < 0 {
			v = 0
		}
		z[j] = v
	}
	// fc2: c x r, sigmoid
	w := make([]float64, c)
	for j := 0; j < c; j++ {
		var v float64
		row := b.SE2W[j*r : (j+1)*r]
		for i := 0; i < r; i++ {
			v += row[i] * z[i]
		}
		v += b.SE2B[j]
		w[j] = 1.0 / (1.0 + math.Exp(-v))
	}
	// scale channels
	out := make([]float64, len(in))
	for k := 0; k < c; k++ {
		wk := w[k]
		ch := in[k*per : (k+1)*per]
		och := out[k*per : (k+1)*per]
		for i := 0; i < per; i++ {
			och[i] = ch[i] * wk
		}
	}
	return out
}

// maxPool2x2 downsamples a channel-major feature map of size h x h.
func maxPool2x2(in []float64, h, ch int) []float64 {
	nh := h / 2
	out := make([]float64, ch*nh*nh)
	for c := 0; c < ch; c++ {
		for y := 0; y < nh; y++ {
			for x := 0; x < nh; x++ {
				a := in[c*h*h+(2*y)*h+2*x]
				b := in[c*h*h+(2*y)*h+2*x+1]
				d := in[c*h*h+(2*y+1)*h+2*x]
				e := in[c*h*h+(2*y+1)*h+2*x+1]
				m := a
				if b > m {
					m = b
				}
				if d > m {
					m = d
				}
				if e > m {
					m = e
				}
				out[c*nh*nh+y*nh+x] = m
			}
		}
	}
	return out
}
