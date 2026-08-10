#!/usr/bin/env python3
"""Audit, aggregate and render the GraphMem/Mem0 concurrency Pareto matrix."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
FANDOL = Path(
    "/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/fandol/"
    "FandolHei-Regular.otf")

# Keep the two paper-facing comparison figures visually interchangeable.
PAPER_FIGSIZE = (14.4, 4.8)
PARETO_FIGSIZE = (14.4, 10.5)
SYSTEM_COLORS = {"GraphMem": "#2563EB", "Mem0 OSS": "#F97316"}
CORE_STYLES = {
    1: {"marker": "^", "linestyle": ":"},
    4: {"marker": "s", "linestyle": "--"},
    8: {"marker": "o", "linestyle": "-"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_11/mem0_pareto_20260809")
    parser.add_argument("--report", type=Path,
                        default=WORKSPACE / "GraphMem_report")
    return parser.parse_args()


def metric(cell: dict[str, Any], section: str, key: str) -> float:
    value = cell["aggregate"][section][key]
    if isinstance(value, dict):
        value = value["mean"]
    return float(value)


def load_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    sources = []
    for system_dir, label in (("graphmem", "GraphMem"), ("mem0", "Mem0 OSS")):
        for workers in (1, 4, 8):
            path = root / system_dir / f"w{workers}" / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            sources.append({
                "system": label,
                "workers": workers,
                "path": str(path.resolve()),
                "workload_sha256": summary["workload_sha256"],
                "cpu_ids": summary["worker_cpu_ids"],
                "query_count": summary["query_count"],
                "memory_count": summary["memory_count"],
            })
            for cell in summary["cells"]:
                trial = cell["trials"][0]
                rows.append({
                    "system": label,
                    "workers": workers,
                    "clients": int(cell["clients"]),
                    "qps": metric(cell, "qps", "mean"),
                    "latency_mean_ms": metric(cell, "latency_ms", "mean"),
                    "latency_p50_ms": metric(cell, "latency_ms", "p50"),
                    "latency_p95_ms": metric(cell, "latency_ms", "p95"),
                    "latency_p99_ms": metric(cell, "latency_ms", "p99"),
                    "latency_max_ms": metric(cell, "latency_ms", "max"),
                    "service_mean_ms": metric(cell, "service_ms", "mean"),
                    "service_p95_ms": metric(cell, "service_ms", "p95"),
                    "queue_mean_ms": metric(cell, "queue_ms", "mean"),
                    "queue_p95_ms": metric(cell, "queue_ms", "p95"),
                    "completed": int(trial["completed"]),
                    "failed": int(trial["failed"]),
                    "timed_out": int(trial["timed_out"]),
                    "rejected": int(trial["rejected"]),
                    "wrong_partition": int(trial.get(
                        "wrong_memory", trial.get("wrong_user", 0))),
                    "worker_rss_mib": float(cell["total_worker_rss_mib"]),
                    "worker_pss_mib": float(cell["total_worker_pss_mib"]),
                    "duration_measured_sec": float(
                        trial["duration_measured_sec"]),
                })
    return rows, sources


def mark_pareto(rows: list[dict[str, Any]]) -> None:
    for worker in (1, 4, 8):
        group = [row for row in rows if row["workers"] == worker]
        for row in group:
            row["pareto"] = not any(
                other["qps"] >= row["qps"]
                and other["latency_p95_ms"] <= row["latency_p95_ms"]
                and (other["qps"] > row["qps"]
                     or other["latency_p95_ms"] < row["latency_p95_ms"])
                for other in group
            )


def pairwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["system"], row["workers"], row["clients"]): row
               for row in rows}
    result = []
    for workers in (1, 4, 8):
        for clients in (1, 4, 16, 64, 128, 256):
            graph = indexed[("GraphMem", workers, clients)]
            mem0 = indexed[("Mem0 OSS", workers, clients)]
            result.append({
                "workers": workers,
                "clients": clients,
                "graphmem_qps": graph["qps"],
                "mem0_qps": mem0["qps"],
                "qps_speedup": graph["qps"] / mem0["qps"],
                "graphmem_p95_ms": graph["latency_p95_ms"],
                "mem0_p95_ms": mem0["latency_p95_ms"],
                "p95_reduction": 1.0 - (
                    graph["latency_p95_ms"] / mem0["latency_p95_ms"]),
                "graphmem_rss_mib": graph["worker_rss_mib"],
                "mem0_rss_mib": mem0["worker_rss_mib"],
                "rss_ratio": mem0["worker_rss_mib"] / graph["worker_rss_mib"],
                "graphmem_pss_mib": graph["worker_pss_mib"],
                "mem0_pss_mib": mem0["worker_pss_mib"],
                "pss_ratio": mem0["worker_pss_mib"] / graph["worker_pss_mib"],
                "strictly_dominates": (
                    graph["qps"] > mem0["qps"]
                    and graph["latency_p95_ms"] < mem0["latency_p95_ms"]),
            })
    return result


def configure_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if FANDOL.is_file():
        font_manager.fontManager.addfont(str(FANDOL))
        family = font_manager.FontProperties(fname=str(FANDOL)).get_name()
        plt.rcParams["font.family"] = family
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "axes.edgecolor": "#94A3B8",
        "axes.labelcolor": "#334155",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "grid.color": "#CBD5E1",
        "grid.alpha": 0.55,
        "figure.facecolor": "white",
        "axes.facecolor": "#F8FAFC",
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 15,
        "lines.linewidth": 2.6,
    })
    return plt


def save_figure(fig, base: Path, *, fixed_canvas: bool = False) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        output = base.with_suffix(f".{suffix}")
        fig.savefig(output, dpi=220,
                    bbox_inches=None if fixed_canvas else "tight",
                    facecolor="white")
        if suffix == "svg":
            output.write_text(
                "\n".join(line.rstrip() for line in
                          output.read_text(encoding="utf-8").splitlines())
                + "\n",
                encoding="utf-8",
            )


def place_concurrency_labels(fig, axis, points: list[dict[str, Any]]) -> None:
    """Greedily place concurrency labels without covering peers or markers."""
    from matplotlib import patheffects
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import Bbox

    renderer = fig.canvas.get_renderer()
    font = FontProperties(size=13)
    pixels_per_point = fig.dpi / 72.0
    axis_box = axis.get_window_extent(renderer)
    axis_box = Bbox.from_extents(
        axis_box.x0 + 2, axis_box.y0 + 2,
        axis_box.x1 - 2, axis_box.y1 - 2,
    )
    display_points = [axis.transData.transform((row["x"], row["y"]))
                      for row in points]
    marker_boxes = [Bbox.from_extents(x - 8, y - 8, x + 8, y + 8)
                    for x, y in display_points]
    densities = []
    for index, (x, y) in enumerate(display_points):
        density = sum(
            int(other != index and abs(x - ox) < 72 and abs(y - oy) < 40)
            for other, (ox, oy) in enumerate(display_points)
        )
        densities.append(density)
    order = sorted(range(len(points)),
                   key=lambda index: (-densities[index],
                                      -points[index]["y"],
                                      points[index]["x"]))
    placed: list[Bbox] = []
    fallback_offsets = [
        (0, 14), (0, -14), (13, 11), (-13, 11), (13, -11), (-13, -11),
        (19, 0), (-19, 0), (0, 22), (0, -22),
        (21, 14), (-21, 14), (21, -14), (-21, -14),
        (28, 0), (-28, 0), (0, 31), (0, -31),
    ]

    for index in order:
        point = points[index]
        px, py = display_points[index]
        width, height, _ = renderer.get_text_width_height_descent(
            point["label"], font, ismath=False)
        candidates = [point["preferred"], *fallback_offsets]
        # Preserve candidate order while dropping duplicates.
        candidates = list(dict.fromkeys(candidates))
        selected = candidates[0]
        selected_alignment = ("center", "center")
        selected_box = None
        for dx, dy in candidates:
            ha = "left" if dx > 1 else "right" if dx < -1 else "center"
            va = "bottom" if dy > 1 else "top" if dy < -1 else "center"
            anchor_x = px + dx * pixels_per_point
            anchor_y = py + dy * pixels_per_point
            x0 = (anchor_x if ha == "left" else
                  anchor_x - width if ha == "right" else
                  anchor_x - width / 2)
            y0 = (anchor_y if va == "bottom" else
                  anchor_y - height if va == "top" else
                  anchor_y - height / 2)
            candidate_box = Bbox.from_extents(
                x0 - 2.2, y0 - 2.2,
                x0 + width + 2.2, y0 + height + 2.2,
            )
            if not (axis_box.contains(candidate_box.x0, candidate_box.y0)
                    and axis_box.contains(candidate_box.x1, candidate_box.y1)):
                continue
            if any(candidate_box.overlaps(box) for box in placed):
                continue
            if any(candidate_box.overlaps(box) for other, box in enumerate(marker_boxes)
                   if other != index):
                continue
            selected = (dx, dy)
            selected_alignment = (ha, va)
            selected_box = candidate_box
            break
        if selected_box is None:
            dx, dy = selected
            selected_alignment = (
                "left" if dx > 1 else "right" if dx < -1 else "center",
                "bottom" if dy > 1 else "top" if dy < -1 else "center",
            )
        else:
            placed.append(selected_box)
        annotation = axis.annotate(
            point["label"], (point["x"], point["y"]),
            xytext=selected, textcoords="offset points",
            fontsize=13, color=point["color"],
            ha=selected_alignment[0], va=selected_alignment[1],
            clip_on=True, zorder=5,
        )
        annotation.set_path_effects([
            patheffects.withStroke(linewidth=4.0, foreground="white"),
            patheffects.Normal(),
        ])


def plot_pareto(
    rows: list[dict[str, Any]],
    base: Path,
    *,
    sampling_note: str = "完整在线检索路径；不含回答模型生成",
) -> None:
    plt = configure_plotting()
    from matplotlib.lines import Line2D

    metrics = (
        ("latency_p50_ms", "(a) p50"),
        ("latency_p95_ms", "(b) p95"),
        ("latency_p99_ms", "(c) p99"),
        ("latency_max_ms", "(d) max"),
    )
    fig, axes_grid = plt.subplots(2, 2, figsize=PARETO_FIGSIZE, sharey=True)
    axes = list(axes_grid.flat)
    labels_by_axis: list[list[dict[str, Any]]] = []
    for axis, (metric_key, panel_title) in zip(axes, metrics):
        point_labels: list[dict[str, Any]] = []
        for system in ("Mem0 OSS", "GraphMem"):
            for workers in (1, 4, 8):
                group = sorted((row for row in rows
                                if row["workers"] == workers
                                and row["system"] == system),
                               key=lambda row: row["clients"])
                style = CORE_STYLES[workers]
                axis.plot(
                    [row[metric_key] for row in group],
                    [row["qps"] for row in group],
                    color=SYSTEM_COLORS[system],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    markersize=8.2,
                    markeredgecolor="white",
                    markeredgewidth=0.75,
                    alpha=0.90,
                    zorder=2,
                )
                for row in group:
                    preferred = {
                        "GraphMem": {1: (8, -17), 4: (8, 12), 8: (8, 11)},
                        "Mem0 OSS": {1: (-8, 18), 4: (-8, -19), 8: (-8, 13)},
                    }[system][workers]
                    point_labels.append({
                        "x": row[metric_key], "y": row["qps"],
                        "label": str(row["clients"]),
                        "color": SYSTEM_COLORS[system],
                        "preferred": preferred,
                    })
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylim(8.0, 205.0)
        axis.set_yticks([10, 20, 50, 100, 200])
        axis.set_yticklabels(["10", "20", "50", "100", "200"])
        axis.grid(True, which="both", linestyle="--", linewidth=0.65)
        axis.set_title(panel_title, fontweight="bold")
        axis.set_xlabel("端到端延迟（ms，对数轴）")
        labels_by_axis.append(point_labels)
    axes[0].set_ylabel("吞吐（QPS，对数轴）")
    axes[2].set_ylabel("吞吐（QPS，对数轴）")
    method_handles = [
        Line2D([0], [0], color=SYSTEM_COLORS[system], linewidth=3.4,
               label=system)
        for system in ("GraphMem", "Mem0 OSS")
    ]
    core_handles = [
        Line2D([0], [0], color="#64748B", marker=CORE_STYLES[workers]["marker"],
               linestyle=CORE_STYLES[workers]["linestyle"], markersize=8.4,
               markerfacecolor="#64748B", markeredgecolor="white",
               label=f"{workers} core")
        for workers in (1, 4, 8)
    ]
    fig.legend(handles=method_handles + core_handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 0.925), frameon=False,
               handlelength=2.8, columnspacing=1.7)
    fig.suptitle(
        "GraphMem vs. Mem0：吞吐–延迟并发对比（左上更优）",
        y=0.985, fontsize=21, fontweight="bold",
    )
    fig.text(0.995, 0.018, "点旁数字 = 并发用户数", ha="right",
             fontsize=13.5, color="#64748B")
    fig.text(0.005, 0.018, sampling_note,
             ha="left", fontsize=13.5, color="#64748B")
    fig.subplots_adjust(left=0.065, right=0.992, bottom=0.095, top=0.835,
                        wspace=0.16, hspace=0.34)
    fig.canvas.draw()
    for axis, point_labels in zip(axes, labels_by_axis):
        place_concurrency_labels(fig, axis, point_labels)
    save_figure(fig, base, fixed_canvas=True)
    plt.close(fig)


def plot_memory(rows: list[dict[str, Any]], base: Path) -> None:
    """Connect worker-scaling points within each system and concurrency."""
    plt = configure_plotting()
    from matplotlib.lines import Line2D

    workers = (1, 4, 8)
    clients = (1, 4, 16, 64, 128, 256)
    client_markers = {
        1: "o", 4: "s", 16: "^", 64: "D", 128: "P", 256: "X",
    }
    metrics = (
        ("worker_rss_mib", "(a) RSS"),
        ("worker_pss_mib", "(b) PSS"),
    )
    fig, axes = plt.subplots(1, 2, figsize=PAPER_FIGSIZE, sharey=True)
    for axis, (metric_key, panel_title) in zip(axes, metrics):
        for system in ("GraphMem", "Mem0 OSS"):
            for client in clients:
                group = sorted(
                    (row for row in rows
                     if row["system"] == system
                     and row["clients"] == client),
                    key=lambda row: row["workers"],
                )
                axis.plot(
                    [row[metric_key] / 1024.0 for row in group],
                    [row["qps"] for row in group],
                    color=SYSTEM_COLORS[system], linewidth=1.9,
                    linestyle="-", marker=client_markers[client],
                    markersize=6.5, alpha=0.9,
                    markeredgecolor="white", markeredgewidth=0.9,
                    zorder=3,
                )
        max_memory = max(row[metric_key] for row in rows) / 1024.0
        axis.set_xlim(0, max_memory * 1.1)
        axis.set_ylim(0, 184)
        axis.set_xlabel("检索 worker 聚合内存（GiB）")
        axis.set_ylabel("QPS")
        axis.set_title(panel_title, fontweight="bold")
        axis.grid(True, linestyle="--", linewidth=0.65, zorder=0)
    method_handles = [
        Line2D([0], [0], color=SYSTEM_COLORS["GraphMem"],
               linewidth=2.5, label="GraphMem"),
        Line2D([0], [0], color=SYSTEM_COLORS["Mem0 OSS"],
               linewidth=2.5, label="Mem0 OSS"),
    ]
    client_handles = [
        Line2D([0], [0], color="#64748B", marker=client_markers[client],
               linestyle="none", markersize=7, label=f"C={client}")
        for client in clients
    ]
    fig.legend(method_handles + client_handles,
               [handle.get_label() for handle in method_handles + client_handles],
               loc="upper center", ncol=8,
               bbox_to_anchor=(0.5, 0.91), frameon=False,
               handlelength=2.2, columnspacing=1.2)
    fig.suptitle("GraphMem 与 Mem0：按并发连接 worker 扩展点的 QPS–Memory 曲线",
                 y=0.985, fontsize=21, fontweight="bold")
    fig.text(0.005, 0.018,
             "每条线固定方法与并发度，并依次连接 1、4、8 worker；颜色区分方法，点型区分并发",
             ha="left", fontsize=13.5, color="#64748B")
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.17, top=0.77,
                        wspace=0.22)
    save_figure(fig, base, fixed_canvas=True)
    plt.close(fig)


def plot_scaling(rows: list[dict[str, Any]], base: Path) -> None:
    plt = configure_plotting()
    colors = {"GraphMem": "#2563EB", "Mem0 OSS": "#F97316"}
    markers = {"GraphMem": "o", "Mem0 OSS": "s"}
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.6), sharex=True)
    for column, workers in enumerate((1, 4, 8)):
        for system in ("GraphMem", "Mem0 OSS"):
            group = sorted((row for row in rows
                            if row["workers"] == workers
                            and row["system"] == system),
                           key=lambda row: row["clients"])
            xs = [row["clients"] for row in group]
            axes[0, column].plot(
                xs, [row["qps"] for row in group], marker=markers[system],
                color=colors[system], linewidth=2, markersize=5, label=system,
            )
            axes[1, column].plot(
                xs, [row["latency_p95_ms"] for row in group],
                marker=markers[system], color=colors[system],
                linewidth=2, markersize=5, label=system,
            )
        axes[0, column].set_title(f"{workers} worker / core", fontweight="bold")
        axes[0, column].set_ylabel("QPS")
        axes[1, column].set_ylabel("p95 延迟（ms，对数轴）")
        axes[1, column].set_yscale("log")
        axes[1, column].set_xlabel("并发用户数")
        for row_axis in (axes[0, column], axes[1, column]):
            row_axis.set_xscale("log", base=2)
            row_axis.set_xticks([1, 4, 16, 64, 128, 256])
            row_axis.set_xticklabels(["1", "4", "16", "64", "128", "256"])
            row_axis.grid(True, which="both", linestyle="--", linewidth=0.7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.suptitle("并发扩展曲线：吞吐与 p95 尾延迟", y=1.04,
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, base)
    plt.close(fig)


def plot_decomposition(rows: list[dict[str, Any]], base: Path) -> None:
    plt = configure_plotting()
    colors = {"GraphMem": "#2563EB", "Mem0 OSS": "#F97316"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for system in ("GraphMem", "Mem0 OSS"):
        group = sorted((row for row in rows
                        if row["workers"] == 8 and row["system"] == system),
                       key=lambda row: row["clients"])
        xs = [row["clients"] for row in group]
        axes[0].plot(xs, [row["service_mean_ms"] for row in group],
                     marker="o", linewidth=2, color=colors[system], label=system)
        axes[1].plot(xs, [row["queue_mean_ms"] for row in group],
                     marker="o", linewidth=2, color=colors[system], label=system)
    for axis, title, ylabel in zip(
        axes,
        ("Worker 内服务时间", "调度与排队时间"),
        ("平均 service latency（ms）", "平均 queue latency（ms）"),
    ):
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks([1, 4, 16, 64, 128, 256])
        axis.set_xticklabels(["1", "4", "16", "64", "128", "256"])
        axis.set_xlabel("并发用户数")
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontweight="bold")
        axis.grid(True, which="both", linestyle="--", linewidth=0.7)
    axes[0].legend(frameon=False)
    fig.suptitle("8-worker 延迟分解：服务路径与并发排队",
                 y=1.03, fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, base)
    plt.close(fig)


def write_tex_table(path: Path, pairs: list[dict[str, Any]]) -> None:
    lines = [
        "% Auto-generated by render_v5_11_mem0_pareto.py",
        "\\begin{tabular}{rr|rrr|rrr}",
        "\\toprule",
        "Workers & 并发 & GraphMem QPS & Mem0 QPS & 加速比 & "
        "GraphMem p95 & Mem0 p95 & p95 降幅 \\\\",
        "\\midrule",
    ]
    for index, row in enumerate(pairs):
        lines.append(
            f"{row['workers']} & {row['clients']} & "
            f"{row['graphmem_qps']:.2f} & {row['mem0_qps']:.2f} & "
            f"{row['qps_speedup']:.2f}$\\times$ & "
            f"{row['graphmem_p95_ms']:.0f} ms & {row['mem0_p95_ms']:.0f} ms & "
            f"{row['p95_reduction'] * 100:.1f}\\% \\\\")
        if index in (5, 11):
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows, sources = load_rows(args.input)
    mark_pareto(rows)
    pairs = pairwise(rows)
    if len(rows) != 36 or len(pairs) != 18:
        raise RuntimeError("incomplete 3x6x2 Pareto matrix")
    failures = sum(row["failed"] + row["timed_out"] + row["rejected"]
                   + row["wrong_partition"] for row in rows)
    graphmem_frontier = [row for row in rows
                         if row["system"] == "GraphMem" and row["pareto"]]
    mem0_frontier = [row for row in rows
                     if row["system"] == "Mem0 OSS" and row["pareto"]]
    payload = {
        "schema_version": "graphmem-mem0-pareto-aggregate-v1",
        "protocol": json.loads((args.input / "workload.json").read_text())["protocol"],
        "sources": sources,
        "audit": {
            "cells_expected": 36,
            "cells_actual": len(rows),
            "failed_timeout_rejected_or_wrong_partition": failures,
            "paired_strict_dominance": sum(
                int(row["strictly_dominates"]) for row in pairs),
            "paired_total": len(pairs),
            "all_mem0_points_dominated_within_worker": all(
                not row["pareto"] for row in rows if row["system"] == "Mem0 OSS"),
        },
        "headline": {
            "graphmem_peak_qps": max(row["qps"] for row in rows
                                     if row["system"] == "GraphMem"),
            "mem0_peak_qps": max(row["qps"] for row in rows
                                 if row["system"] == "Mem0 OSS"),
            "graphmem_frontier_points": len(graphmem_frontier),
            "mem0_frontier_points": len(mem0_frontier),
        },
        "rows": rows,
        "pairwise": pairs,
    }
    args.input.joinpath("aggregate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with args.input.joinpath("results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures = args.report / "figures"
    generated = args.report / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    plot_pareto(rows, figures / "eval_mem0_pareto")
    plot_memory(rows, figures / "eval_mem0_memory")
    plot_scaling(rows, figures / "eval_mem0_scaling")
    plot_decomposition(rows, figures / "eval_mem0_latency_decomposition")
    write_tex_table(generated / "v5_11_mem0_pareto_table.tex", pairs)
    (generated / "v5_11_mem0_pareto_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "aggregate": str(args.input / "aggregate.json"),
        "rows": len(rows),
        "audit": payload["audit"],
        "headline": payload["headline"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
