"""
Run learned-memory diagnostics for DPD checkpoints.

Modes:
  membank    learned memory grids and stats
  attention  Stage-1 RBF attention maps
  retrieval  Stage-2 same-position memory retrieval outputs and top-k panels
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

import data as data_utils
from model_analytic import patchify, unpatchify
from model_dpd import DPDConfig, DeepPatchwiseDenoiser


SELECTIONS = ["first", "random", "highest_norm", "lowest_norm", "highest_std", "lowest_std", "brightest", "darkest"]


def cfg_from_checkpoint(cfg_dict: dict) -> DPDConfig:
    valid = {f.name for f in fields(DPDConfig)}
    return DPDConfig(**{k: v for k, v in dict(cfg_dict).items() if k in valid})


def run_name_from_ckpt(ckpt: str) -> str:
    path = Path(ckpt)
    try:
        return path.parents[1].name
    except IndexError as exc:
        raise ValueError(f"Could not infer run name from checkpoint path: {ckpt}") from exc


def default_outdir(mode: str, ckpt: str, sigma: float) -> Path:
    run_name = run_name_from_ckpt(ckpt)
    if mode == "membank":
        return Path("diagnostics") / f"{mode}_{run_name}"
    return Path("diagnostics") / f"{mode}_{run_name}_sig{sigma:g}"


def load_model(ckpt_path: str, device: torch.device, use_ema: bool) -> Tuple[DeepPatchwiseDenoiser, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ckpt["model_cfg"])
    cfg.print_params = True
    model = DeepPatchwiseDenoiser(cfg).to(device)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=False)
    if use_ema and ckpt.get("ema") is not None:
        msd = model.state_dict()
        copied = 0
        for k, v in ckpt["ema"].items():
            k = k.removeprefix("_orig_mod.")
            if k in msd and torch.is_floating_point(msd[k]):
                msd[k].copy_(v.to(device=msd[k].device, dtype=msd[k].dtype))
                copied += 1
        print(f"[ckpt] using EMA weights; copied={copied}")
    model.eval()
    return model, ckpt


def build_folder_dataset(root: str, resolution: int, channels: int) -> Dataset:
    tfm = data_utils.build_transform(
        resolution=int(resolution),
        hflip=False,
        channels=int(channels),
        normalize_to="neg_one_to_one",
    )
    return data_utils.ImageFolderFlat(root=root, transform=tfm)


def get_images(ds: Dataset, indices: Sequence[int]) -> torch.Tensor:
    return torch.stack([(ds[int(idx)][0] if isinstance(ds[int(idx)], (tuple, list)) else ds[int(idx)]) for idx in indices], dim=0)


def parse_indices(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_query_patches(raw: str, grid_h: int, grid_w: int) -> List[Tuple[int, int]]:
    aliases = {
        "center": (grid_h // 2, grid_w // 2),
        "top_left": (0, 0),
        "top_right": (0, grid_w - 1),
        "bottom_left": (grid_h - 1, 0),
        "bottom_right": (grid_h - 1, grid_w - 1),
        "upper_center": (grid_h // 3, grid_w // 2),
        "lower_center": (2 * grid_h // 3, grid_w // 2),
    }
    out: List[Tuple[int, int]] = []
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        if item in aliases:
            out.append(aliases[item])
        elif ":" in item:
            r, c = item.split(":", 1)
            out.append((int(r), int(c)))
        else:
            idx = int(item)
            out.append((idx // grid_w, idx % grid_w))
    deduped = []
    seen = set()
    for r, c in out:
        if not (0 <= r < grid_h and 0 <= c < grid_w):
            raise ValueError(f"query patch {(r, c)} is outside grid {grid_h}x{grid_w}")
        if (r, c) not in seen:
            deduped.append((r, c))
            seen.add((r, c))
    return deduped


def parse_layers(raw: str, num_layers: int) -> List[int]:
    if raw.strip().lower() in {"all", "*"}:
        return list(range(num_layers))
    layers = sorted({int(x) for x in raw.split(",") if x.strip()})
    for layer in layers:
        if not (0 <= layer < num_layers):
            raise ValueError(f"layer {layer} is outside [0,{num_layers})")
    return layers


def to_image01(x: torch.Tensor) -> np.ndarray:
    x = ((x.detach().cpu().float() + 1.0) / 2.0).clamp(0.0, 1.0)
    arr = x.permute(1, 2, 0).numpy()
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def image_tensor_to_pil(
    img: torch.Tensor,
    *,
    mode: str,
    size: int,
    marker_rc: Optional[Tuple[int, int]] = None,
    patch_grid: Optional[int] = None,
) -> Image.Image:
    x = img.detach().cpu().float()
    if mode == "clipped":
        x = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    elif mode == "minmax":
        lo = x.amin(dim=(1, 2), keepdim=True)
        hi = x.amax(dim=(1, 2), keepdim=True)
        x = (x - lo) / (hi - lo).clamp_min(1e-8)
    else:
        raise ValueError(f"unknown image mode: {mode}")
    arr = x.permute(1, 2, 0).numpy()
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    pil = Image.fromarray((arr * 255.0).round().astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST)
    if marker_rc is not None and patch_grid is not None:
        draw = ImageDraw.Draw(pil)
        r, c = marker_rc
        cell = size / float(patch_grid)
        draw.rectangle(
            (int(round(c * cell)), int(round(r * cell)), int(round((c + 1) * cell)), int(round((r + 1) * cell))),
            outline=(255, 0, 0),
            width=max(1, size // 64),
        )
    return pil


def labeled_tile(img: Image.Image, label: str, *, pad: int = 16) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + pad), "white")
    out.paste(img.convert("RGB"), (0, pad))
    ImageDraw.Draw(out).text((2, 2), label, fill=(0, 0, 0))
    return out


def save_grid(path: Path, tiles: List[Image.Image], cols: int) -> None:
    if not tiles:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = math.ceil(len(tiles) / cols)
    w = max(tile.width for tile in tiles)
    h = max(tile.height for tile in tiles)
    canvas = Image.new("RGB", (cols * w, rows * h), "white")
    for i, tile in enumerate(tiles):
        canvas.paste(tile.convert("RGB"), ((i % cols) * w, (i // cols) * h))
    canvas.save(path)


def colorize(values: np.ndarray, *, gamma: float = 1.0) -> np.ndarray:
    x = values.astype(np.float64)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    y = np.zeros_like(x) if hi <= lo + 1e-20 else np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    if gamma != 1.0:
        y = np.power(y, float(gamma))
    stops = np.array([[0, 0, 4], [72, 18, 110], [177, 54, 122], [251, 136, 97], [252, 253, 191]], dtype=np.float64)
    pos = y * (len(stops) - 1)
    lo_i = np.floor(pos).astype(np.int64)
    hi_i = np.clip(lo_i + 1, 0, len(stops) - 1)
    frac = pos[..., None] - lo_i[..., None]
    return np.clip(stops[lo_i] * (1.0 - frac) + stops[hi_i] * frac, 0, 255).astype(np.uint8)


def heatmap_image(heat: np.ndarray, *, size: int, gamma: float, marker_rc: Optional[Tuple[int, int]] = None) -> Image.Image:
    img = Image.fromarray(colorize(heat, gamma=gamma)).resize((size, size), Image.Resampling.NEAREST)
    if marker_rc is not None:
        draw = ImageDraw.Draw(img)
        grid_h, grid_w = heat.shape
        r, c = marker_rc
        cell_w, cell_h = size / float(grid_w), size / float(grid_h)
        draw.rectangle(
            (int(round(c * cell_w)), int(round(r * cell_h)), int(round((c + 1) * cell_w)), int(round((r + 1) * cell_h))),
            outline=(0, 255, 255),
            width=max(1, size // 64),
        )
    return img


def select_memory_indices(memory: torch.Tensor, *, selection: str, num_images: int, seed: int) -> List[int]:
    n = int(memory.shape[0])
    if selection == "first":
        return list(range(min(num_images, n)))
    if selection == "random":
        rng = random.Random(seed)
        out = list(range(n))
        rng.shuffle(out)
        return out[: min(num_images, n)]
    flat = memory.float().reshape(n, -1)
    scores = {
        "highest_norm": flat.norm(dim=1),
        "lowest_norm": -flat.norm(dim=1),
        "highest_std": flat.std(dim=1),
        "lowest_std": -flat.std(dim=1),
        "brightest": flat.mean(dim=1),
        "darkest": -flat.mean(dim=1),
    }
    return torch.argsort(scores[selection], descending=True)[: min(num_images, n)].tolist()


def write_membank(args, model: DeepPatchwiseDenoiser, outdir: Path) -> None:
    memory = model.learned_memory.detach().cpu().float()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[membank] memory={tuple(memory.shape)} outdir={outdir}")
    selections = SELECTIONS if args.all_selections else [args.selection]
    for selection in selections:
        indices = select_memory_indices(memory, selection=selection, num_images=args.num_memory_images, seed=args.seed)
        imgs = unpatchify(
            memory[indices],
            img_channels=model.cfg.img_channels,
            img_resolution=model.cfg.img_resolution,
            patch_size=model.cfg.patch_size,
        )
        sel_dir = outdir / selection if args.all_selections else outdir
        sel_dir.mkdir(parents=True, exist_ok=True)
        for mode in ["clipped", "minmax"]:
            tiles = [labeled_tile(image_tensor_to_pil(img, mode=mode, size=args.mem_tile_size), str(idx)) for img, idx in zip(imgs, indices)]
            save_grid(sel_dir / f"learned_memory_{selection}_{mode}.png", tiles, cols=args.mem_cols)
        flat = imgs.float().reshape(imgs.shape[0], -1)
        with (sel_dir / f"learned_memory_{selection}_stats.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "min", "max", "mean", "std", "l2_norm", "frac_lt_minus1", "frac_gt_plus1"])
            writer.writeheader()
            for row_idx, idx in enumerate(indices):
                x = flat[row_idx]
                writer.writerow({
                    "index": idx,
                    "min": float(x.min()),
                    "max": float(x.max()),
                    "mean": float(x.mean()),
                    "std": float(x.std()),
                    "l2_norm": float(x.norm()),
                    "frac_lt_minus1": float((x < -1.0).float().mean()),
                    "frac_gt_plus1": float((x > 1.0).float().mean()),
                })
        torch.save({"indices": indices, "images": imgs, "tokens": memory[indices]}, sel_dir / f"learned_memory_{selection}_selected.pt")


def stage1_attention(layer, tokens: torch.Tensor, pos_embed: torch.Tensor, tau, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    b, m, _d = tokens.shape
    if cond.shape[0] == 1 and b != 1:
        cond = cond.expand(b, -1)
    x = tokens + pos_embed.to(device=tokens.device, dtype=tokens.dtype)
    x = layer.window_encoder._norm(x, cond)
    q = torch.einsum("bmd,hdk->bhmk", x, layer.window_encoder.W_q)
    k = torch.einsum("bmd,hdk->bhmk", x, layer.window_encoder.W_k)
    v = torch.einsum("bmd,hde->bhme", x, layer.window_encoder.W_vo)
    tau_t = torch.as_tensor(tau, device=tokens.device, dtype=torch.float32).clamp_min(1e-8)
    qf, kf = q.float(), k.float()
    dist2 = ((qf[:, :, :, None, :] - kf[:, :, None, :, :]) ** 2).sum(dim=-1).clamp_min(0.0)
    attn = torch.softmax(-dist2 / (2.0 * (tau_t ** 2)), dim=-1)
    value_features = torch.einsum("bhqm,bhme->bhqe", attn.to(dtype=v.dtype), v)
    return attn, value_features


def entropy_stats(weights: np.ndarray, grid_h: int, grid_w: int, query_rc: Tuple[int, int]) -> Tuple[float, float, float]:
    p = weights.astype(np.float64)
    p = p / max(float(p.sum()), 1e-20)
    entropy = float(-(p * np.log(p + 1e-20)).sum())
    eff = float(np.exp(entropy))
    rr, cc = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    qr, qc = query_rc
    dist = np.sqrt((rr.reshape(-1) - qr) ** 2 + (cc.reshape(-1) - qc) ** 2)
    return entropy, eff, float((p.reshape(-1) * dist).sum())


def write_attention(args, model: DeepPatchwiseDenoiser, sigma: float, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    ds = build_folder_dataset(args.data_root, args.resolution, args.channels)
    indices = parse_indices(args.indices)
    clean = get_images(ds, indices).to(device)
    noisy = clean + sigma * torch.randn_like(clean)
    outputs = model(noisy, sigma=sigma, return_all=True)
    states = outputs["states"]
    cond = model.time_condition(sigma, device=device)
    queries = parse_query_patches(args.query_patches, model.grid_size, model.grid_size)
    rows: List[Dict[str, float]] = []
    for b, data_idx in enumerate(indices):
        sample_dir = outdir / f"sample_{data_idx:06d}"
        tiles = [
            labeled_tile(image_tensor_to_pil(clean[b], mode="clipped", size=args.tile_size), "clean"),
            labeled_tile(image_tensor_to_pil(noisy[b], mode="clipped", size=args.tile_size), "noisy"),
        ]
        tiles.extend(labeled_tile(image_tensor_to_pil(state[b], mode="clipped", size=args.tile_size), f"h{i}") for i, state in enumerate(states))
        save_grid(sample_dir / "states.png", tiles, cols=6)
        for layer_idx, layer in enumerate(model.layers):
            tau1, _tau2 = layer.resolve_taus(sigma, cond, device=device)
            tokens = patchify(states[layer_idx][b : b + 1].to(device=device, dtype=layer.window_encoder.W_q.dtype), model.cfg.patch_size)
            attn_t, value_features = stage1_attention(layer, tokens, model.pos_embed, tau1, cond)
            attn = attn_t[0].detach().float().cpu().numpy()
            layer_dir = sample_dir / f"layer_{layer_idx:02d}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            avg_matrix = Image.fromarray(colorize(attn.mean(axis=0), gamma=args.heatmap_gamma)).resize((args.full_matrix_size, args.full_matrix_size), Image.Resampling.NEAREST)
            labeled_tile(avg_matrix, "head avg full attention").save(layer_dir / "full_attention_avg.png")
            for qr, qc in queries:
                qidx = qr * model.grid_size + qc
                head_tiles = [
                    labeled_tile(heatmap_image(attn[h, qidx].reshape(model.grid_size, model.grid_size), size=args.attention_map_size, gamma=args.heatmap_gamma, marker_rc=(qr, qc)), f"head {h}")
                    for h in range(attn.shape[0])
                ]
                save_grid(layer_dir / f"q{qr:02d}_{qc:02d}_heads.png", head_tiles, cols=4)
                avg = attn[:, qidx].mean(axis=0).reshape(model.grid_size, model.grid_size)
                save_grid(
                    layer_dir / f"q{qr:02d}_{qc:02d}_avg.png",
                    [labeled_tile(heatmap_image(avg, size=args.attention_map_size, gamma=args.heatmap_gamma, marker_rc=(qr, qc)), "head avg")],
                    cols=1,
                )
                for head_idx in range(attn.shape[0]):
                    ent, eff, disp = entropy_stats(attn[head_idx, qidx], model.grid_size, model.grid_size, (qr, qc))
                    rows.append({"sample_index": data_idx, "layer": layer_idx, "head": head_idx, "query_row": qr, "query_col": qc, "entropy": ent, "effective_size": eff, "expected_distance": disp})
                ent, eff, disp = entropy_stats(attn[:, qidx].mean(axis=0), model.grid_size, model.grid_size, (qr, qc))
                rows.append({"sample_index": data_idx, "layer": layer_idx, "head": -1, "query_row": qr, "query_col": qc, "entropy": ent, "effective_size": eff, "expected_distance": disp})
    with (outdir / "attention_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_index", "layer", "head", "query_row", "query_col", "entropy", "effective_size", "expected_distance"])
        writer.writeheader()
        writer.writerows(rows)


def layer_retrieval(model: DeepPatchwiseDenoiser, layer_idx: int, x_state: torch.Tensor, mem_tokens: torch.Tensor, sigma: float, cond: torch.Tensor):
    layer = model.layers[layer_idx]
    tau1, tau2 = layer.resolve_taus(sigma, cond, device=x_state.device)
    x_tokens = patchify(x_state.to(dtype=layer.window_encoder.W_q.dtype), model.cfg.patch_size)
    q_tokens = layer.window_encoder(x_tokens, model.pos_embed, tau=tau1, cond=cond.expand(x_state.shape[0], -1))
    q_cond = cond.expand(x_state.shape[0], -1)
    mem_cond = cond.expand(mem_tokens.shape[0], -1)
    k_tokens = layer.window_encoder(mem_tokens.to(dtype=layer.window_encoder.W_q.dtype), model.pos_embed, tau=tau1, cond=mem_cond)
    q_feat = layer.stage2_project_q(layer.stage2_normalize_qk(q_tokens, q_cond).float()).float()
    k_feat = layer.stage2_project_k(layer.stage2_normalize_qk(k_tokens, mem_cond).float()).float()
    v = layer.stage2_project_v(mem_tokens.float()).float()
    tau2_t = torch.as_tensor(tau2, device=x_state.device, dtype=torch.float32).clamp_min(float(model.cfg.min_tau))
    q_norm2 = (q_feat ** 2).sum(dim=-1, keepdim=True)
    k_norm2 = (k_feat ** 2).sum(dim=-1).transpose(0, 1).unsqueeze(0)
    dot = torch.einsum("bmd,rmd->bmr", q_feat, k_feat)
    weights = torch.softmax(-(q_norm2 + k_norm2 - 2.0 * dot).clamp_min(0.0) / (2.0 * (tau2_t ** 2)), dim=-1)
    pred_tokens = torch.einsum("bmr,rmd->bmd", weights, v)
    denoised = unpatchify(pred_tokens, img_channels=model.cfg.img_channels, img_resolution=model.cfg.img_resolution, patch_size=model.cfg.patch_size).float()
    alpha = layer.update_alpha(cond, device=x_state.device, dtype=torch.float32)
    alpha_img = alpha if alpha.ndim == 0 else unpatchify(alpha.expand(x_state.shape[0], -1, model.token_dim), img_channels=model.cfg.img_channels, img_resolution=model.cfg.img_resolution, patch_size=model.cfg.patch_size)
    updated = (1.0 - alpha_img) * x_state.float() + alpha_img * denoised
    if model.cfg.clamp_output:
        updated = updated.clamp(-1.0, 1.0)
    return denoised, updated, weights, v


def write_retrieval(args, model: DeepPatchwiseDenoiser, sigma: float, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    ds = build_folder_dataset(args.data_root, args.resolution, args.channels)
    sample_indices = parse_indices(args.indices)
    clean = get_images(ds, sample_indices).to(device=device, dtype=torch.float32)
    noisy = clean + sigma * torch.randn_like(clean)
    mem_tokens = model.learned_memory.detach().to(device=device, dtype=next(model.parameters()).dtype)
    cond = model.time_condition(sigma, device=device)
    query_patches = parse_query_patches(args.query_patches, model.grid_size, model.grid_size)
    topk_layers = parse_layers(args.topk_layers, model.cfg.num_layers)
    memory_indices = select_memory_indices(mem_tokens.detach().cpu(), selection="random", num_images=args.num_memory_images, seed=args.seed)
    raw_imgs = unpatchify(mem_tokens.detach().cpu().float(), img_channels=model.cfg.img_channels, img_resolution=model.cfg.img_resolution, patch_size=model.cfg.patch_size)
    save_grid(outdir / "value_projected" / "raw_memory_clipped.png", [labeled_tile(image_tensor_to_pil(raw_imgs[i], mode="clipped", size=args.tile_size), f"mem {i}") for i in memory_indices], cols=args.cols)
    with (outdir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"sigma": sigma, "indices": sample_indices, "query_patches": query_patches, "topk_layers": topk_layers}, f, indent=2)

    states = [noisy.float()]
    denoised_layers: List[torch.Tensor] = []
    weights_by_layer: List[torch.Tensor] = []
    value_tokens_by_layer: List[torch.Tensor] = []
    x_state = noisy.float()
    for layer_idx in range(model.cfg.num_layers):
        denoised, x_state, weights, value_tokens = layer_retrieval(model, layer_idx, x_state, mem_tokens, sigma, cond)
        denoised_layers.append(denoised.detach().cpu())
        states.append(x_state.detach().cpu())
        weights_by_layer.append(weights.detach().cpu())
        value_tokens_by_layer.append(value_tokens.detach().cpu())

    for b, data_idx in enumerate(sample_indices):
        tiles = [
            labeled_tile(image_tensor_to_pil(clean[b], mode="clipped", size=args.tile_size), "clean"),
            labeled_tile(image_tensor_to_pil(noisy[b], mode="clipped", size=args.tile_size), "noisy"),
        ]
        for layer_idx, d in enumerate(denoised_layers):
            tiles.append(labeled_tile(image_tensor_to_pil(states[layer_idx][b], mode="clipped", size=args.tile_size), f"h{layer_idx}"))
            tiles.append(labeled_tile(image_tensor_to_pil(d[b], mode="clipped", size=args.tile_size), f"D{layer_idx}"))
        tiles.append(labeled_tile(image_tensor_to_pil(states[-1][b], mode="clipped", size=args.tile_size), "final"))
        save_grid(outdir / "retrieval_outputs" / f"sample_{data_idx:06d}_states_D_layers.png", tiles, cols=6)

    rows = []
    for name, x in [("clean", clean.detach().cpu()), ("noisy", noisy.detach().cpu())]:
        rows.append({"name": name, "mse_to_clean": float(torch.mean((x.float() - clean.detach().cpu().float()) ** 2)), "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()), "std": float(x.std(unbiased=False))})
    for layer_idx, d in enumerate(denoised_layers):
        for name, x in [(f"D{layer_idx}", d), (f"h{layer_idx + 1}", states[layer_idx + 1])]:
            rows.append({"name": name, "mse_to_clean": float(torch.mean((x.float() - clean.detach().cpu().float()) ** 2)), "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()), "std": float(x.std(unbiased=False))})
    path = outdir / "retrieval_outputs" / "retrieval_stats.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "min", "max", "mean", "std", "mse_to_clean"])
        writer.writeheader()
        writer.writerows(rows)

    topk_rows = []
    for b, data_idx in enumerate(sample_indices):
        for layer_idx in topk_layers:
            weights = weights_by_layer[layer_idx][b]
            value_imgs = unpatchify(value_tokens_by_layer[layer_idx].float(), img_channels=model.cfg.img_channels, img_resolution=model.cfg.img_resolution, patch_size=model.cfg.patch_size)
            for qr, qc in query_patches:
                qidx = qr * model.grid_size + qc
                vals, idxs = torch.topk(weights[qidx], k=min(args.topk, weights.shape[-1]))
                tiles = [
                    labeled_tile(image_tensor_to_pil(clean[b], mode="clipped", size=args.tile_size, marker_rc=(qr, qc), patch_grid=model.grid_size), "clean"),
                    labeled_tile(image_tensor_to_pil(noisy[b], mode="clipped", size=args.tile_size, marker_rc=(qr, qc), patch_grid=model.grid_size), "noisy"),
                    labeled_tile(image_tensor_to_pil(states[layer_idx][b], mode="clipped", size=args.tile_size, marker_rc=(qr, qc), patch_grid=model.grid_size), f"h{layer_idx}"),
                ]
                for rank, (val, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
                    tiles.append(labeled_tile(image_tensor_to_pil(raw_imgs[idx], mode="clipped", size=args.tile_size, marker_rc=(qr, qc), patch_grid=model.grid_size), f"raw {rank}: {idx}"))
                    tiles.append(labeled_tile(image_tensor_to_pil(value_imgs[idx], mode="clipped", size=args.tile_size, marker_rc=(qr, qc), patch_grid=model.grid_size), f"Wv {rank}: {val:.3f}"))
                    topk_rows.append({"sample_index": data_idx, "layer": layer_idx, "query_row": qr, "query_col": qc, "rank": rank, "memory_index": idx, "weight": float(val)})
                save_grid(outdir / "topk" / f"sample_{data_idx:06d}" / f"layer_{layer_idx:02d}_q{qr:02d}_{qc:02d}_top{args.topk}.png", tiles, cols=3)
    with (outdir / "topk" / "topk_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_index", "layer", "query_row", "query_col", "rank", "memory_index", "weight"])
        writer.writeheader()
        writer.writerows(topk_rows)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--mode", choices=["membank", "attention", "retrieval"], required=True)
    p.add_argument("--sigma", type=float, default=1.0, help="Used by attention/retrieval; ignored for membank output naming.")
    p.add_argument("--outdir", default=None)
    p.add_argument("--data_root", default="./datasets/celeba32/test")
    p.add_argument("--resolution", type=int, default=32)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--indices", default="0,25,100")
    p.add_argument("--query_patches", default="center,6:7,8:8,10:8,0:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num_memory_images", type=int, default=24)
    p.add_argument("--selection", choices=SELECTIONS, default="random")
    p.add_argument("--all_selections", action="store_true")
    p.add_argument("--mem_cols", type=int, default=6)
    p.add_argument("--mem_tile_size", type=int, default=72)
    p.add_argument("--tile_size", type=int, default=80)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--topk_layers", default="0,3,7")
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--heatmap_gamma", type=float, default=0.35)
    p.add_argument("--attention_map_size", type=int, default=120)
    p.add_argument("--full_matrix_size", type=int, default=512)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, ckpt = load_model(args.ckpt, device=device, use_ema=args.use_ema)
    sigma = float(args.sigma)
    outdir = Path(args.outdir) if args.outdir is not None else default_outdir(args.mode, args.ckpt, sigma)
    print(f"[run] mode={args.mode} sigma={sigma:g} ckpt={args.ckpt} outdir={outdir}")
    if args.mode == "membank":
        write_membank(args, model, outdir)
    elif args.mode == "attention":
        write_attention(args, model, sigma, outdir)
    elif args.mode == "retrieval":
        write_retrieval(args, model, sigma, outdir)
    print(f"[done] wrote {outdir}")


if __name__ == "__main__":
    main()
