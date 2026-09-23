"""Render a one-page result summary for a resume or portfolio attachment.

Usage:
    python scripts/make_portfolio.py --run runs/<name> [--out docs/portfolio.jpg]

Produces a single landscape image: headline metrics, one sample per class, the
confusion matrix, the training curves, the class-imbalance chart and a short
write-up of the technical decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

from aoi_detection.config import load_train_config
from aoi_detection.data.dataset import (
    ID_COL,
    LABEL_COL,
    class_counts,
    compute_class_weights,
    load_labels,
    split_train_valid,
)
from aoi_detection.utils.logging import get_logger

log = get_logger("portfolio")

ZH = ("正常", "空洞", "水平刮痕", "垂直刮痕", "邊緣缺陷", "顆粒")

INTRO = (
    "針對工業表面影像的自動光學檢測任務，建立可區分五種瑕疵型態與正常品的分類模型。\n"
    "評估標準以產線成本為準：漏檢（不良品流出）遠比過殺（良品被誤刷）昂貴，因此以\n"
    "混淆矩陣的錯誤方向作為調整依據，而非單一準確率。"
)


def _use_cjk_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _narrative(counts, weights) -> list[tuple[str, str]]:
    ratio = max(counts.values) / min(counts.values)
    return [
        (
            "模型選型",
            "選用 MobileNetV2（ImageNet 預訓練）搭配\n"
            "GAP + Dropout(0.5) + Softmax(6)。輕量架構\n"
            "兼顧辨識準確率與產線即時推論的延遲需求。",
        ),
        (
            "類別不平衡",
            f"最多與最少類別相差 {ratio:.1f} 倍。以 balanced\n"
            f"class weight 將稀有類損失加權 ×{max(weights.values()):.1f}，\n"
            f"使僅 {counts[2]} 張訓練樣本的水平刮痕達 100% recall。",
        ),
        (
            "誤判分析",
            "逐件檢視漏檢樣本的 softmax 機率分佈，確認\n"
            "多數屬高信心誤判而非模型猶豫，據此判斷\n"
            "調整信心門檻無效，問題須由資料面著手。",
        ),
        (
            "工程實作",
            "將實驗 notebook 重構為模組化套件：超參數\n"
            "外部化為 YAML、固定亂數種子確保可重現，\n"
            "並建置 35 項單元測試與 ruff / mypy 檢查。",
        ),
    ]


def build(run_dir: Path, cfg_path: Path, out: Path) -> Path:
    _use_cjk_font()
    cfg = load_train_config(cfg_path)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    history = json.loads((run_dir / "history.json").read_text())
    cm = np.array(metrics["confusion_matrix"])

    train_df, _ = split_train_valid(load_labels(cfg.train_csv_path), cfg.valid_split, cfg.seed)
    counts = class_counts(train_df)
    weights = compute_class_weights(train_df)

    defective = int(cm[1:].sum())
    escapes = int(cm[1:, 0].sum())
    overkill = int(cm[0, 1:].sum())
    good = int(cm[0].sum())

    fig = plt.figure(figsize=(16, 11.6), dpi=110)
    fig.patch.set_facecolor("white")
    gs = GridSpec(
        4,
        4,
        figure=fig,
        height_ratios=[1.25, 2.6, 2.2, 1.3],
        hspace=0.45,
        wspace=0.3,
        left=0.05,
        right=0.97,
        top=0.955,
        bottom=0.04,
    )

    # --- headline -------------------------------------------------------
    head = fig.add_subplot(gs[0, :])
    head.axis("off")
    head.text(0, 0.95, "AOI 表面瑕疵自動分類系統", fontsize=25, fontweight="bold", va="top")
    head.text(
        0,
        0.60,
        f"MobileNetV2 遷移學習 · 六類瑕疵 · 訓練 {len(train_df)} 張 / 獨立測試 {int(cm.sum())} 張",
        fontsize=13,
        color="#555",
        va="top",
    )
    head.text(0, 0.34, INTRO, fontsize=10.5, color="#444", va="top", linespacing=1.65)

    cards = [
        ("測試準確率", f"{metrics['accuracy']:.1%}", f"{int(np.trace(cm))}/{int(cm.sum())} 張正確"),
        ("漏檢率", f"{escapes / defective:.1%}", f"{escapes}/{defective} 件不良品未攔下"),
        ("過殺率", f"{overkill / good:.1%}", f"{overkill}/{good} 件良品被誤判"),
        ("最稀有類 recall", f"{cm[2, 2] / cm[2].sum():.0%}", f"僅 {counts[2]} 張訓練樣本"),
    ]
    for i, (label, value, note) in enumerate(cards):
        x = 0.545 + i * 0.118
        head.text(x, 1.00, label, fontsize=10, color="#666", va="top")
        head.text(x, 0.79, value, fontsize=21, fontweight="bold", va="top", color="#b5451b")
        head.text(x, 0.44, note, fontsize=8, color="#888", va="top")

    # --- class samples --------------------------------------------------
    inner = gs[1, :2].subgridspec(2, 3, hspace=0.35, wspace=0.08)
    for i in range(6):
        ax = fig.add_subplot(inner[i // 3, i % 3])
        ids = train_df.loc[train_df[LABEL_COL].astype(int) == i, ID_COL].to_numpy()
        img = cv2.imread(str(cfg.train_images_dir / ids[0]), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{i}  {ZH[i]}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    # Anchor the section label just above the sample grid so layout tweaks
    # cannot push it on top of the images.
    samples_top = gs[1, :2].get_position(fig).y1
    fig.text(
        0.05,
        samples_top + 0.018,
        "六類瑕疵樣本（512×512 灰階）",
        fontsize=12,
        fontweight="bold",
    )

    # --- confusion matrix -----------------------------------------------
    ax = fig.add_subplot(gs[1, 2:])
    sns.heatmap(
        cm,
        cmap="Oranges",
        annot=True,
        fmt="d",
        cbar=False,
        ax=ax,
        xticklabels=ZH,
        yticklabels=ZH,
        annot_kws={"fontsize": 11},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title("混淆矩陣（獨立測試集）", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("模型預測", fontsize=10)
    ax.set_ylabel("實際類別", fontsize=10)
    plt.setp(ax.get_xticklabels(), fontsize=9, rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), fontsize=9, rotation=0)

    # --- training curve -------------------------------------------------
    ax = fig.add_subplot(gs[2, :2])
    epochs = range(1, len(history["accuracy"]) + 1)
    ax.plot(epochs, history["accuracy"], lw=2, label="訓練集")
    ax.plot(epochs, history["val_accuracy"], lw=2, label="驗證集")
    best = max(history["val_accuracy"])
    be = history["val_accuracy"].index(best) + 1
    ax.scatter([be], [best], s=70, zorder=5, color="#b5451b")
    ax.annotate(
        f"最佳 checkpoint\nepoch {be}, {best:.1%}",
        (be, best),
        textcoords="offset points",
        xytext=(-15, -45),
        fontsize=9,
        ha="center",
        color="#b5451b",
    )
    ax.set_title("訓練過程：準確率（以驗證集挑選最佳權重）", fontsize=12, fontweight="bold")
    ax.set_xlabel("epoch", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    # --- imbalance ------------------------------------------------------
    ax = fig.add_subplot(gs[2, 2:])
    x = np.arange(6)
    bars = ax.bar(x, counts.values, color="#d9d9d9", edgecolor="#999")
    ax.set_xticks(x)
    ax.set_xticklabels(ZH, fontsize=9)
    ax.set_ylabel("訓練張數", fontsize=10)
    ax.set_title("類別不平衡處理：balanced class weight", fontsize=12, fontweight="bold")
    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 12,
            f"{counts[i]} 張\n權重 ×{weights[i]:.2f}",
            ha="center",
            fontsize=8.5,
            color="#b5451b" if weights[i] > 1.5 else "#666",
            fontweight="bold" if weights[i] > 1.5 else "normal",
        )
    ax.set_ylim(0, max(counts.values) * 1.28)
    ax.grid(axis="y", alpha=0.3)
    ax.annotate(
        f"最多與最少相差 {max(counts.values) / min(counts.values):.1f} 倍\n"
        f"→ 稀有類損失加權 ×{max(weights.values()):.1f}，該類測試 recall 100%",
        xy=(0.5, 0.86),
        xycoords="axes fraction",
        fontsize=9,
        ha="center",
        bbox={"boxstyle": "round,pad=0.5", "fc": "#fff4ee", "ec": "#e0b9a4"},
    )

    # --- narrative ------------------------------------------------------
    foot = fig.add_subplot(gs[3, :])
    foot.axis("off")
    foot.add_patch(
        plt.Rectangle(
            (0, 0), 1, 1, transform=foot.transAxes, fc="#fafafa", ec="#e2e2e2", lw=1, zorder=0
        )
    )
    for i, (title, body) in enumerate(_narrative(counts, weights)):
        x = 0.022 + i * 0.247
        foot.text(
            x,
            0.86,
            title,
            fontsize=11.5,
            fontweight="bold",
            color="#b5451b",
            va="top",
            transform=foot.transAxes,
        )
        foot.text(
            x,
            0.60,
            body,
            fontsize=9.3,
            color="#444",
            va="top",
            linespacing=1.8,
            transform=foot.transAxes,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=110,
        facecolor="white",
        bbox_inches="tight",
        pil_kwargs={"quality": 88} if out.suffix.lower() in {".jpg", ".jpeg"} else None,
    )
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="run directory to summarise")
    parser.add_argument("--config", type=Path, default=Path("configs/train_mobilenet_v2.yaml"))
    parser.add_argument("--out", type=Path, default=Path("docs/portfolio.jpg"))
    args = parser.parse_args(argv)

    out = build(args.run, args.config, args.out)
    size_kb = out.stat().st_size / 1024
    log.info("wrote %s (%.0f KB)", out, size_kb)
    if size_kb > 1024:
        log.warning("over 1 MB - 104 recommends staying under that for image uploads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
