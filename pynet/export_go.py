"""Export a trained EmbedCNN checkpoint to the iconnet Go binary format.

The binary is a flat stream of little-endian float32 tensors with a small
header. Order matches the model construction exactly:

  header : magic "ICN1" (4 bytes) + nBlocks uint8
  block i (outCh ci, c_in = 4 for i==0 else prev ci):
    conv1.W (ci x c_in x 3 x 3), conv1.b (ci)
    bn1.gamma (ci), bn1.beta (ci), bn1.mean (ci), bn1.var (ci)   # RUNNING stats
    conv2.W (ci x ci x 3 x 3), conv2.b (ci)
    bn2.gamma (ci), bn2.beta (ci), bn2.mean (ci), bn2.var (ci)
    se.fc1.W (r x ci), se.fc1.b (r), se.fc2.W (ci x r), se.fc2.b (ci)   # r = ci/8
  embed.W (emb x outChLast), embed.b (emb)

PyTorch sequential slot offsets: block i -> body.{8i} conv1, {8i+1} bn1,
{8i+3} conv2, {8i+4} bn2, {8i+6} SE. MaxPool (8i+7) has no params.
"""
import argparse
import struct

import torch


def flatten(v):
    return v.detach().to("cpu").float().contiguous().numpy().ravel().astype("<f4")


def write_tensor(f, v):
    f.write(flatten(v).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    st = ck["state"]
    ch = list(ck["ch"])
    n = len(ch)
    assert ck["attn"] == "se", "export only supports SE attention"
    assert ck["res"] == 64, "export assumes 64x64 input (res=%d)" % ck["res"]
    emb = ck["emb"]

    with open(args.out, "wb") as f:
        f.write(b"ICN1")
        f.write(struct.pack("<B", n))
        for ci in ch:
            f.write(struct.pack("<I", ci))
        c_in = 4
        for i, ci in enumerate(ch):
            base = 8 * i
            write_tensor(f, st["body.%d.weight" % base])          # conv1.W
            write_tensor(f, st["body.%d.bias" % base])            # conv1.b
            write_tensor(f, st["body.%d.weight" % (base + 1)])    # bn1.gamma
            write_tensor(f, st["body.%d.bias" % (base + 1)])      # bn1.beta
            write_tensor(f, st["body.%d.running_mean" % (base + 1)])
            write_tensor(f, st["body.%d.running_var" % (base + 1)])
            write_tensor(f, st["body.%d.weight" % (base + 3)])    # conv2.W
            write_tensor(f, st["body.%d.bias" % (base + 3)])      # conv2.b
            write_tensor(f, st["body.%d.weight" % (base + 4)])    # bn2.gamma
            write_tensor(f, st["body.%d.bias" % (base + 4)])      # bn2.beta
            write_tensor(f, st["body.%d.running_mean" % (base + 4)])
            write_tensor(f, st["body.%d.running_var" % (base + 4)])
            r = ci // 8
            write_tensor(f, st["body.%d.fc.0.weight" % (base + 6)])  # se.fc1.W (r x ci)
            write_tensor(f, st["body.%d.fc.0.bias" % (base + 6)])    # se.fc1.b
            write_tensor(f, st["body.%d.fc.2.weight" % (base + 6)])  # se.fc2.W (ci x r)
            write_tensor(f, st["body.%d.fc.2.bias" % (base + 6)])    # se.fc2.b
            c_in = ci
        write_tensor(f, st["embed.weight"])  # emb x outChLast
        write_tensor(f, st["embed.bias"])

    size = __import__("os").path.getsize(args.out)
    print("exported %s: %d bytes (%.1f KB)" % (args.out, size, size / 1024))


if __name__ == "__main__":
    main()
