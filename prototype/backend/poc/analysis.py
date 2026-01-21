from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy import stats
except Exception:
    stats = None


def _load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _pearson_and_p(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return float("nan"), float("nan")
    r = cov / math.sqrt(var_x * var_y)
    if stats is None:
        return r, float("nan")
    return r, stats.pearsonr(xs, ys).pvalue


def analyze(path: str) -> None:
    rows = _load_rows(Path(path))
    pairs = {"baseline": defaultdict(list), "cos": defaultdict(list)}

    for r in rows:
        dim = r.get("dimension")
        dim_names = r.get("dim_names") or []
        dim_values = r.get("dim_values") or []
        if dim not in dim_names:
            continue
        idx = dim_names.index(dim)
        target = dim_values[idx]

        for method, key in [("baseline", "judge_baseline"), ("cos", "judge_cos")]:
            judge = r.get(key, {})
            parsed = judge.get("parsed") if isinstance(judge, dict) else None
            if not parsed:
                continue
            scores = parsed.get("scores")
            if not scores or not isinstance(scores, list):
                continue
            score = scores[0].get("score")
            if score is None:
                continue
            pairs[method][dim].append((target, score))

    print("Method\tDimension\tN\tPearson\tp_value")
    for method in ["baseline", "cos"]:
        for dim, ts in pairs[method].items():
            xs = [t for t, _ in ts]
            ys = [s for _, s in ts]
            r, p = _pearson_and_p(xs, ys)
            p_str = f"{p:.3g}" if not math.isnan(p) else "nan"
            print(f"{method}\t{dim}\t{len(ts)}\t{r:.3f}\t{p_str}")

    for method in ["baseline", "cos"]:
        all_ts = []
        all_scores = []
        for ts in pairs[method].values():
            for t, s in ts:
                all_ts.append(t)
                all_scores.append(s)
        r, p = _pearson_and_p(all_ts, all_scores)
        p_str = f"{p:.3g}" if not math.isnan(p) else "nan"
        print(f"{method}\tALL\t{len(all_ts)}\t{r:.3f}\t{p_str}")

    if stats is None:
        print("Note: scipy not available; p-values not computed.")


def _fit_linear(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan"), float("nan")
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return float("nan"), float("nan")
    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _r2(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan")
    slope, intercept = _fit_linear(x, y)
    if math.isnan(slope):
        return float("nan")
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1 - ss_res / ss_tot


def _partial_corr(x, y, controls):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    controls = np.asarray(controls, dtype=float)
    if x.size < 3:
        return float("nan")
    if controls.ndim == 1:
        controls = controls[:, None]
    if controls.size == 0:
        return _pearson_and_p(x.tolist(), y.tolist())[0]
    x_res = x - controls @ np.linalg.lstsq(controls, x, rcond=None)[0]
    y_res = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
    return _pearson_and_p(x_res.tolist(), y_res.tolist())[0]


def analyze_decoupling(
    path: str,
    heatmap_path: str | None = None,
    heatmap_metric: str = "slope",
) -> None:
    if np is None:
        raise RuntimeError("numpy is required for decoupling analysis.")
    rows = _load_rows(Path(path))
    records = []
    for r in rows:
        dim = r.get("dimension")
        dim_names = r.get("dim_names") or []
        dim_values = r.get("dim_values") or []
        if dim not in dim_names:
            continue
        for method, key in [("baseline", "judge_baseline"), ("cos", "judge_cos")]:
            judge = r.get(key, {})
            parsed = judge.get("parsed") if isinstance(judge, dict) else None
            if not parsed:
                continue
            scores = parsed.get("scores")
            if not scores or not isinstance(scores, list):
                continue
            score = scores[0].get("score")
            if score is None:
                continue
            records.append({
                "prompt": r.get("prompt", ""),
                "method": method,
                "dimension": dim,
                "dim_names": dim_names,
                "dim_values": dim_values,
                "score": float(score),
            })

    dims = sorted({name for rec in records for name in rec["dim_names"]})
    print("Method\tJudgedDim\tSliderDim\tN\tSlope\tR2\tPartialCorr")
    summaries = {"baseline": {"self": [], "cross": []}, "cos": {"self": [], "cross": []}}
    heatmap_data = {"baseline": {}, "cos": {}}

    for method in ["baseline", "cos"]:
        for judged_dim in dims:
            for slider_dim in dims:
                xs, ys, controls = [], [], []
                for rec in records:
                    if rec["method"] != method:
                        continue
                    if rec["dimension"] != judged_dim:
                        continue
                    dim_names = rec["dim_names"]
                    dim_values = rec["dim_values"]
                    if slider_dim not in dim_names:
                        continue
                    slider_idx = dim_names.index(slider_dim)
                    xs.append(dim_values[slider_idx])
                    ys.append(rec["score"])
                    other_vals = [v for i, v in enumerate(dim_values) if i != slider_idx]
                    controls.append(other_vals)
                if len(xs) < 2:
                    continue
                slope, _ = _fit_linear(xs, ys)
                r2 = _r2(xs, ys)
                pcorr = _partial_corr(xs, ys, controls)
                label = "self" if judged_dim == slider_dim else "cross"
                summaries[method][label].append(abs(slope))
                print(f"{method}\t{judged_dim}\t{slider_dim}\t{len(xs)}\t{slope:.3f}\t{r2:.3f}\t{pcorr:.3f}")
                heatmap_data[method][(judged_dim, slider_dim)] = {
                    "slope": slope,
                    "r2": r2,
                    "partial": pcorr,
                    "n": len(xs),
                }

    for method in ["baseline", "cos"]:
        self_mean = float(np.mean(summaries[method]["self"])) if summaries[method]["self"] else float("nan")
        cross_mean = float(np.mean(summaries[method]["cross"])) if summaries[method]["cross"] else float("nan")
        ratio = self_mean / (cross_mean + 1e-9) if not math.isnan(self_mean) and not math.isnan(cross_mean) else float("nan")
        print(f"{method}\tSELF_SLOPE_MEAN\t{self_mean:.3f}")
        print(f"{method}\tCROSS_SLOPE_MEAN\t{cross_mean:.3f}")
        print(f"{method}\tDECOUPLING_RATIO\t{ratio:.3f}")

    if heatmap_path:
        if plt is None:
            raise RuntimeError("matplotlib is required for heatmap output.")
        metric = heatmap_metric.lower()
        if metric not in {"slope", "r2", "partial"}:
            raise ValueError("heatmap_metric must be one of: slope, r2, partial")
        for method in ["baseline", "cos"]:
            mat = np.zeros((len(dims), len(dims)), dtype=float)
            for i, judged_dim in enumerate(dims):
                for j, slider_dim in enumerate(dims):
                    entry = heatmap_data[method].get((judged_dim, slider_dim))
                    mat[i, j] = entry[metric] if entry else float("nan")
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(dims)), dims, rotation=45, ha="right")
            ax.set_yticks(range(len(dims)), dims)
            ax.set_title(f"{method} {metric} heatmap")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            out_path = Path(heatmap_path)
            out_path = out_path.with_name(f"{out_path.stem}_{method}_{metric}{out_path.suffix}")
            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            print(f"Wrote heatmap: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analysis utilities for grid judge JSONL.")
    parser.add_argument(
        "--input",
        default="/scratch/yirenl2/projects/context-steering-writing/outputs/grid_judge.jsonl",
        help="Path to grid_judge.jsonl",
    )
    parser.add_argument(
        "--decoupling",
        action="store_true",
        help="Run decoupling analysis (slopes, R2, partial correlations).",
    )
    parser.add_argument(
        "--heatmap",
        default="",
        help="If set, writes heatmaps to this path (suffixes per method/metric).",
    )
    parser.add_argument(
        "--heatmap_metric",
        default="slope",
        help="Heatmap metric: slope, r2, or partial.",
    )
    args = parser.parse_args()

    analyze(args.input)
    if args.decoupling:
        heatmap_path = args.heatmap or None
        analyze_decoupling(args.input, heatmap_path=heatmap_path, heatmap_metric=args.heatmap_metric)


if __name__ == "__main__":
    main()
