#!/usr/bin/env python3
"""
Chạy dóng hàng CroCoAlign, có vá lỗi tương thích với stack hiện đại:

- PL 2.6 bỏ `LightningModule._load_model_state` mà `nn_core.load_model` gọi -> tự nạp checkpoint.
- torch 2.6+ mặc định `weights_only=True` -> nạp với weights_only=False (checkpoint tin cậy).
- nn_core lưu checkpoint dạng .ckpt.zip (bên trong có `checkpoint.ckpt`).

Tái sử dụng nguyên thuật toán trong CroCoAlign/src/.../crocoalign.py, chỉ thay bước load.

Dùng:
  python run_align.py <ckpt.(zip)> <source.jsonl> <target.jsonl> -o tsv -b 32
"""
import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1] / "CroCoAlign"
sys.path.insert(0, str(REPO / "src"))


def custom_load_model(module_class, checkpoint_path, map_location=None, **_):
    """Nạp checkpoint nn_core (.ckpt.zip) mà không cần _load_model_state của PL cũ."""
    p = Path(checkpoint_path)
    # tìm file zip thực sự
    candidates = [p, p.with_suffix(".ckpt.zip"), Path(str(p) + ".zip")]
    zip_path = next((c for c in candidates if c.exists() and zipfile.is_zipfile(c)), None)
    if zip_path is None:
        if p.exists() and zipfile.is_zipfile(p):
            zip_path = p
        else:
            raise FileNotFoundError(f"Không thấy checkpoint zip hợp lệ từ: {p}")

    with tempfile.TemporaryDirectory() as d:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(d)
        inner = next(Path(d).rglob("checkpoint.ckpt"), None)
        if inner is None:
            raise FileNotFoundError("Không thấy 'checkpoint.ckpt' trong zip")
        # luôn nạp lên CPU để tránh chiếm gấp đôi VRAM (8GB); .to(device) sẽ chuyển sau
        ckpt = torch.load(inner, map_location="cpu", weights_only=False)

    hparams = dict(ckpt.get("hyper_parameters", ckpt.get("hparams", {})) or {})
    print(f"[loader] freeze_encoder={hparams.get('freeze_encoder')} "
          f"precomputed_embeddings={hparams.get('precomputed_embeddings')} "
          f"transformer_name={hparams.get('transformer_name')}")
    model = module_class(metadata=None, **hparams)

    # sentence-transformers đổi tên submodule: cũ `.auto_model.` -> mới `.model.`
    sd = ckpt["state_dict"]
    model_keys = set(model.state_dict().keys())
    remapped = {}
    for k, v in sd.items():
        nk = k
        if nk not in model_keys and ".auto_model." in nk:
            cand = nk.replace(".auto_model.", ".model.")
            if cand in model_keys:
                nk = cand
        remapped[nk] = v
    res = model.load_state_dict(remapped, strict=False)
    miss, unexp = list(res.missing_keys), list(res.unexpected_keys)
    print(f"[loader] load_state_dict: missing={len(miss)} unexpected={len(unexp)}")
    if miss[:5]:
        print("  e.g. missing:", miss[:5])
    if unexp[:5]:
        print("  e.g. unexpected:", unexp[:5])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("source")
    ap.add_argument("target")
    ap.add_argument("-o", "--out", default="tsv")
    ap.add_argument("-b", "--batch_size", type=int, default=32)
    ap.add_argument("-r", "--recovery", default="labse-batch")
    ap.add_argument("-p", "--precomputed_embeddings", action="store_true")
    args = ap.parse_args()

    # vá loader TRƯỚC khi CroCoAlign dùng
    import sentence_aligner.crocoalign as ca

    ca.load_model = custom_load_model

    aligner = ca.CroCoAlign(
        args.ckpt,
        args.source,
        args.target,
        args.precomputed_embeddings,
        args.batch_size,
        args.recovery,
        args.out,
    )
    print(f"[run_align] device = {aligner.device}")
    aligner.main()


if __name__ == "__main__":
    main()
