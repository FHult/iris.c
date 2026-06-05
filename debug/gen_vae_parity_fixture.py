#!/usr/bin/env python3
"""
debug/gen_vae_parity_fixture.py — extract a (image, teacher-latent) pair for the
C-1 VAE parity harness (debug/vae_parity.c). No GPU, no teacher re-run: the teacher
latents already exist in the cold precompute cache.

Writes raw little-endian float32 bins:
  <out>/vae_in.bin      : [3, S, S]  CHW in [-1,1]   (replicates precompute _preprocess_vae)
  <out>/vae_teacher.bin : [32, S/8, S/8]              (mflux encode() VAE-latent, stored)

Usage:
  debug/gen_vae_parity_fixture.py --rec 000004_3398 \
    --shards /Volumes/16TBCold/shards \
    --vae-cache /Volumes/16TBCold/precomputed/vae/current \
    --size 512 --out /tmp
Then:
  /tmp/vae_parity flux-klein-model/vae/diffusion_pytorch_model.safetensors \
    /tmp/vae_in.bin /tmp/vae_teacher.bin 512 512
"""
import argparse, io, os, tarfile
import numpy as np
from PIL import Image as PilImage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True, help="rec_id, e.g. 000004_3398")
    ap.add_argument("--shards", default="/Volumes/16TBCold/shards")
    ap.add_argument("--vae-cache", default="/Volumes/16TBCold/precomputed/vae/current")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default="/tmp")
    a = ap.parse_args()

    shard = a.rec.split("_")[0]
    tar_path = os.path.join(a.shards, f"{shard}.tar")
    with tarfile.open(tar_path) as t:                       # seek-bound on cold HDD
        member = next((m for m in t
                       if m.isfile()
                       and m.name.split("/")[-1].startswith(a.rec)
                       and not m.name.endswith(".txt")), None)
        if member is None:
            print(f"no image member for {a.rec} in {tar_path}")
            return 1
        jpg = t.extractfile(member).read()
    img = np.array(PilImage.open(io.BytesIO(jpg)).convert("RGB"), dtype=np.uint8)
    img = np.array(PilImage.fromarray(img).resize((a.size, a.size), PilImage.LANCZOS),
                   dtype=np.uint8)
    img_f = (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)   # [3,S,S] CHW
    img_f.tofile(os.path.join(a.out, "vae_in.bin"))

    lat = np.load(os.path.join(a.vae_cache, f"{a.rec}.npz"))["latent"].astype(np.float32)
    lat.tofile(os.path.join(a.out, "vae_teacher.bin"))

    print(f"image {member.name}  img_f {img_f.shape} -> {a.out}/vae_in.bin")
    print(f"teacher {lat.shape} mean={lat.mean():.4f} std={lat.std():.4f} "
          f"-> {a.out}/vae_teacher.bin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
