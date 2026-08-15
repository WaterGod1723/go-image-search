# iconnet

A dependency-free (pure Go stdlib) icon embedding + retrieval module. It runs
a small CNN (272k params, 64x64 RGBA input → 64-dim L2-normalized embedding)
trained in Python and exported to a flat binary.

The module is fully self-contained: copy the `iconnet/` directory anywhere and
`go build` — no external packages beyond the Go standard library.

## Layout

```
iconnet/
  go.mod            module iconnet (pure stdlib)
  weights.bin       exported weights (1.07 MB) from pynet/out/ctr3k_se.pt
  net.go            binary weight loader
  forward.go        CNN forward pass (Conv3x3+BN+ReLU ×2, SE, MaxPool, GAP,
                    Linear, L2 norm) — byte-identical to PyTorch
  sprite.go         preprocessing: query segmentation + trim + letterbox,
                    matching pynet/preprocess.py byte-for-byte
  index.go          gallery vectorization + cosine retrieval
  cmd/iconretrieve  CLI demo: build gallery, retrieve query
```

## Usage

```go
net, _ := iconnet.LoadFile("weights.bin")          // 4 blocks, emb 64

// gallery: reduce every reference icon to an embedding once
idx := iconnet.NewIndex(net, refNames, func(k string) (*iconnet.Sprite, bool) {
    return iconnet.RefSprite(filepath.Join(refDir, k), net.InputSize)
})

// query: segment + embed, then cosine top-k
sp, ok := iconnet.QuerySprite(queryPath, net.InputSize)
q := idx.EmbedSprite(sp)
keys, sims := idx.Query(q, 5)
```

CLI:
```
go run ./cmd/iconretrieve -weights weights.bin \
  -gallery /path/to/icons -query /path/to/query.png
```

## How weights are produced (Python side)

`pynet/export_go.py` converts `pynet/out/ctr3k_se.pt` into `weights.bin`:

```
python pynet/export_go.py --ckpt pynet/out/ctr3k_se.pt --out iconnet/weights.bin
```

Binary format: magic `ICN1` + nBlocks(uint8) + per-block channel counts
(uint32) + tensors as little-endian float32. See `net.go` for the exact order.

## Verification

`go test ./...` runs:

- `TestLoadWeights` — loads the network, forward-pass smoke test
- `TestAlignment` — Go embedding == Python embedding (max abs diff ~4e-7) on
  reference sprites saved by the Python pipeline
- `TestFullQueryPipeline` — full Go pipeline (segment→trim→letterbox→embed)
  matches Python's `query_sprite` embedding (cos ≈ 1.0)
- `TestRecallIndomain` — reproduces Python recall@1 on the in-domain test set
  (Go 78.3% vs Python 77.5%, statistically equal)

## Fidelity notes

To be byte-identical with the Python pipeline, the Go port replicates:

- Pillow's exact `Image.resize(BILINEAR)` two-pass convolution (support-scaled
  triangle kernel, PRECISION_BITS=22 fixed-point rounding) — a naive
  `(dst+0.5)*scale-0.5` bilinear does NOT match.
- `scipy.ndimage.label` default = **4-connectivity** (not 8).
- `scipy.ndimage.binary_closing` with a `(2*close_k+1)` square element.
- Background estimated from the 8px border-ring **median**.
- Refs letterbox with their **actual alpha values** (0..255), not binarized.
- RGBA read as **non-premultiplied** (NRGBA).
