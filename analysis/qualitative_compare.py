#!/usr/bin/env python3
"""Reproducible multi-model qualitative comparison on a fixed split/index set.

This script intentionally does NOT select examples using model scores. Supply
predetermined dataset indices before inspecting the corresponding predictions.
It uses the project's RadarWindows + load_model APIs and overlays GT contours,
model probability maps, calibrated decision contours, radar support/age, and
sparse LOS velocity arrows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from doppler_jepa.data import RadarWindows
from doppler_jepa.evaluate import load_model


def parse_model(spec: str) -> Tuple[str, Path, Path]:
    # NAME=checkpoint.pt,threshold.json
    if "=" not in spec or "," not in spec:
        raise argparse.ArgumentTypeError(
            "--model must be NAME=checkpoint.pt,threshold.json"
        )
    name, rhs = spec.split("=", 1)
    ckpt, cal = rhs.split(",", 1)
    return name.strip(), Path(ckpt), Path(cal)


def read_threshold(path: Path, checkpoint: Path) -> float:
    obj = json.loads(path.read_text())
    if obj.get("split") != "dev":
        raise ValueError(f"Calibration must come from dev: {path}")
    expected = str(checkpoint.resolve())
    if obj.get("checkpoint") != expected:
        raise ValueError(
            f"Calibration/checkpoint mismatch:\n  {path}\n  expected {expected}\n"
            f"  got      {obj.get('checkpoint')}"
        )
    return float(obj["threshold"])


def recent_prior_from_raster(x: np.ndarray):
    """Mimic recent-per-cell overwrite at raster resolution for visualization."""
    _, _, h, w = x.shape
    support = np.zeros((h, w), dtype=bool)
    age = np.full((h, w), np.nan, dtype=np.float32)
    vx = np.zeros((h, w), dtype=np.float32)
    vy = np.zeros((h, w), dtype=np.float32)
    for frame in x:
        obs = frame[5] > 0
        support[obs] = True
        age[obs] = frame[4][obs]
        # channels 2:4 are normalized by /20 in points_to_grid
        vx[obs] = 20.0 * frame[2][obs]
        vy[obs] = 20.0 * frame[3][obs]
    return support, age, vx, vy


def plot_input(ax, item, grid):
    x = item["x"].numpy()
    support, age, vx, vy = recent_prior_from_raster(x)
    xx, yy = grid.centers()
    sel = support & np.isfinite(age)
    if sel.any():
        sc = ax.scatter(
            yy[sel], xx[sel], c=-age[sel], s=8,
            vmin=0.0, vmax=2.0, cmap="viridis", linewidths=0,
        )
        # Deterministic thinning for legible arrows.
        ii, jj = np.where(sel)
        keep = np.arange(len(ii)) % max(1, len(ii) // 80 + 1) == 0
        if keep.any():
            ax.quiver(
                yy[ii[keep], jj[keep]], xx[ii[keep], jj[keep]],
                vy[ii[keep], jj[keep]], vx[ii[keep], jj[keep]],
                angles="xy", scale_units="xy", scale=3.0, width=0.003,
                alpha=0.7,
            )
        return sc
    return None


def configure_bev(ax, title: str):
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("left y (m)", fontsize=7)
    ax.set_ylabel("forward x (m)", fontsize=7)
    ax.set_aspect("equal")
    ax.invert_xaxis()
    ax.tick_params(labelsize=6)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--indices",
        default="15,250,500,900,1400,2000",
        help="Predetermined dataset indices; choose BEFORE viewing predictions",
    )
    ap.add_argument(
        "--model", action="append", required=True,
        help="Repeat: NAME=checkpoint.pt,threshold.json",
    )
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    if args.split == "test" and not bool(int(__import__("os").environ.get("ALLOW_FROZEN_TEST_QUALITATIVE", "0"))):
        raise SystemExit(
            "Refusing test qualitative inspection. After protocol freeze, run with "
            "ALLOW_FROZEN_TEST_QUALITATIVE=1 and keep the predeclared indices."
        )

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable")

    parsed = [parse_model(s) for s in args.model]
    models: Dict[str, torch.nn.Module] = {}
    data: Dict[str, RadarWindows] = {}
    thresholds: Dict[str, float] = {}
    grid = None

    for name, ckpt_path, cal_path in parsed:
        model, ckpt, this_grid = load_model(str(ckpt_path), device)
        if grid is None:
            grid = this_grid
        elif this_grid.asdict() != grid.asdict():
            raise ValueError("All compared checkpoints must use the same grid")
        thresholds[name] = read_threshold(cal_path, ckpt_path)
        models[name] = model
        data[name] = RadarWindows(
            args.cache,
            args.split,
            grid=this_grid,
            no_doppler=ckpt["args"]["no_doppler"],
            latest_only=ckpt["args"].get("latest_only", False),
        )

    names = [x[0] for x in parsed]
    base_name = names[0]
    indices = [int(v) for v in args.indices.split(",") if v.strip()]
    if not indices:
        raise ValueError("No indices supplied")
    if max(indices) >= len(data[base_name]):
        raise IndexError(f"Index {max(indices)} >= dataset length {len(data[base_name])}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest: List[dict] = []

    xx, yy = grid.centers()

    for idx in indices:
        items = {name: data[name][idx] for name in names}
        token = items[base_name]["token"]
        scene = items[base_name]["scene"]
        for name in names[1:]:
            if items[name]["token"] != token:
                raise ValueError(f"Dataset row mismatch at {idx}: {name}")

        preds = {}
        for name in names:
            item = items[name]
            logits = models[name](
                item["x"][None].to(device), item["dt"][None].to(device)
            )["logits"]
            preds[name] = logits.sigmoid()[0].cpu().numpy()

        horizons = items[base_name]["dt"].numpy()
        gt = items[base_name]["y"].numpy()
        nrow = len(horizons)
        ncol = 2 + len(names)
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.55*ncol, 2.45*nrow), squeeze=False)
        last_image = None

        for k, horizon in enumerate(horizons):
            # Input/history panel.
            sc = plot_input(axes[k, 0], items[base_name], grid)
            configure_bev(axes[k, 0], f"Radar history\nrow for +{horizon:.2f}s")
            if sc is not None and k == 0:
                cbar = fig.colorbar(sc, ax=axes[k, 0], fraction=0.046, pad=0.03)
                cbar.set_label("observation age (s)", fontsize=6)
                cbar.ax.tick_params(labelsize=6)

            # GT panel.
            axes[k, 1].pcolormesh(yy, xx, gt[k], vmin=0, vmax=1, shading="nearest")
            configure_bev(axes[k, 1], f"Ground truth\n+{horizon:.2f}s")

            # Model panels.
            for j, name in enumerate(names, start=2):
                p = preds[name][k]
                last_image = axes[k, j].pcolormesh(
                    yy, xx,
                    np.ma.masked_where(items[name]["valid"].numpy() == 0, p),
                    vmin=0, vmax=1, shading="nearest",
                )
                if gt[k].any():
                    axes[k, j].contour(yy, xx, gt[k], levels=[0.5], linewidths=1.0)
                thr = thresholds[name]
                if (p.min() < thr) and (p.max() > thr):
                    axes[k, j].contour(
                        yy, xx, p, levels=[thr], linewidths=0.8, linestyles="--"
                    )
                configure_bev(axes[k, j], f"{name}\n+{horizon:.2f}s")

        if last_image is not None:
            cbar = fig.colorbar(last_image, ax=axes[:, 2:].ravel().tolist(), fraction=0.012, pad=0.01)
            cbar.set_label("predicted footprint score", fontsize=7)
            cbar.ax.tick_params(labelsize=6)
        fig.suptitle(f"idx={idx} | scene={scene} | token={token}", fontsize=9)
        fig.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.92, wspace=0.22, hspace=0.32)
        fig.savefig(out / f"qual_{idx:04d}.pdf", bbox_inches="tight")
        fig.savefig(out / f"qual_{idx:04d}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"index": idx, "scene": scene, "token": token, "horizons_s": horizons.tolist()})

    (out / "selection_manifest.json").write_text(json.dumps({
        "split": args.split,
        "selection_rule": "predetermined dataset indices; no method-score selection",
        "indices": indices,
        "models": names,
        "samples": manifest,
    }, indent=2))
    print(json.dumps({"out": str(out), "count": len(indices), "models": names}, indent=2))


if __name__ == "__main__":
    main()
