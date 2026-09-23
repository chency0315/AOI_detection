"""Render a one-page result summary for a resume or portfolio attachment.

Usage:
    python scripts/make_portfolio.py --run runs/<name> [--lang zh|en] [--out FILE]

Produces a single landscape image: headline metrics, one sample per class, the
confusion matrix, the training curves, the class-imbalance chart and a short
write-up of the technical decisions. Every label lives in STRINGS, so adding a
language is a matter of adding one entry rather than touching the layout.
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

STRINGS: dict[str, dict[str, object]] = {
    "zh": {
        "classes": ("正常", "空洞", "水平刮痕", "垂直刮痕", "邊緣缺陷", "顆粒"),
        "title": "AOI 表面瑕疵自動分類系統",
        "subtitle": "MobileNetV2 遷移學習 · 六類瑕疵 · 訓練 {n_train} 張 / 獨立測試 {n_test} 張",
        "intro": (
            "針對工業表面影像的自動光學檢測任務，建立可區分五種瑕疵型態與正常品的分類模型。\n"
            "評估標準以產線成本為準：漏檢（不良品流出）遠比過殺（良品被誤刷）昂貴，因此以\n"
            "混淆矩陣的錯誤方向作為調整依據，而非單一準確率。"
        ),
        "cards": (
            ("測試準確率", "{hit}/{n_test} 張正確"),
            ("漏檢率", "{escapes}/{defective} 件不良品未攔下"),
            ("過殺率", "{overkill}/{good} 件良品被誤判"),
            ("最稀有類 recall", "僅 {rarest} 張訓練樣本"),
        ),
        "samples": "六類瑕疵樣本（512×512 灰階）",
        "cm_title": "混淆矩陣（獨立測試集）",
        "cm_x": "模型預測",
        "cm_y": "實際類別",
        "curve_title": "訓練過程：準確率（以驗證集挑選最佳權重）",
        "curve_train": "訓練集",
        "curve_valid": "驗證集",
        "curve_best": "最佳 checkpoint\nepoch {epoch}, {acc:.1%}",
        "bar_title": "類別不平衡處理：balanced class weight",
        "bar_y": "訓練張數",
        "bar_label": "{n} 張\n權重 ×{w:.2f}",
        "bar_note": (
            "最多與最少相差 {ratio:.1f} 倍\n→ 稀有類損失加權 ×{maxw:.1f}，該類測試 recall 100%"
        ),
        "blocks": (
            (
                "模型選型",
                "選用 MobileNetV2（ImageNet 預訓練）搭配\n"
                "GAP + Dropout(0.5) + Softmax(6)。輕量架構\n"
                "兼顧辨識準確率與產線即時推論的延遲需求。",
            ),
            (
                "類別不平衡",
                "最多與最少類別相差 {ratio:.1f} 倍。以 balanced\n"
                "class weight 將稀有類損失加權 ×{maxw:.1f}，\n"
                "使僅 {rarest} 張訓練樣本的水平刮痕達 100% recall。",
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
        ),
    },
    "en": {
        "classes": ("Normal", "Void", "Horizontal", "Vertical", "Edge", "Particle"),
        "title": "AOI Surface Defect Classification",
        "subtitle": (
            "MobileNetV2 transfer learning · 6 classes · "
            "{n_train} training / {n_test} held-out test images"
        ),
        "intro": (
            "A six-class classifier for automated optical inspection of industrial surface\n"
            "images. Evaluated by line cost rather than accuracy alone: an escape (a defective\n"
            "part passed as good) costs far more than overkill (a good part rejected), so the\n"
            "direction of each error in the confusion matrix drives the tuning."
        ),
        "cards": (
            ("Test accuracy", "{hit}/{n_test} correct"),
            ("Escape rate", "{escapes}/{defective} defects passed"),
            ("Overkill rate", "{overkill}/{good} good parts rejected"),
            ("Rarest-class recall", "from only {rarest} training images"),
        ),
        "samples": "One sample per class (512×512 grayscale)",
        "cm_title": "Confusion matrix (held-out test set)",
        "cm_x": "Predicted",
        "cm_y": "Actual",
        "curve_title": "Training accuracy (best weights chosen on validation)",
        "curve_train": "train",
        "curve_valid": "validation",
        "curve_best": "best checkpoint\nepoch {epoch}, {acc:.1%}",
        "bar_title": "Class imbalance handling: balanced class weights",
        "bar_y": "training images",
        "bar_label": "{n} imgs\nweight ×{w:.2f}",
        "bar_note": (
            "{ratio:.1f}× gap between largest and smallest class\n"
            "→ rare-class loss weighted ×{maxw:.1f}, giving that class 100% test recall"
        ),
        "blocks": (
            (
                "Model choice",
                "MobileNetV2 (ImageNet pretrained) with GAP +\n"
                "Dropout(0.5) + Softmax(6). A lightweight backbone\n"
                "balances accuracy against inline inference latency.",
            ),
            (
                "Class imbalance",
                "A {ratio:.1f}× gap between the largest and smallest\n"
                "class. Balanced class weights scale the rare-class\n"
                "loss ×{maxw:.1f}, taking {rarest} images to 100% recall.",
            ),
            (
                "Error analysis",
                "Inspected the softmax distribution of every escape.\n"
                "Most were confident errors rather than hesitation, so\n"
                "a confidence threshold could not fix them — the gap\n"
                "is in the data, not the decision rule.",
            ),
            (
                "Engineering",
                "Refactored the experiment notebook into a package:\n"
                "hyper-parameters in YAML, a fixed seed for\n"
                "reproducibility, 35 unit tests and ruff / mypy.",
            ),
        ),
    },
}


def _use_font(lang: str) -> None:
    cjk = ["Microsoft JhengHei", "Microsoft YaHei"]
    plt.rcParams["font.sans-serif"] = (cjk if lang == "zh" else []) + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def build(run_dir: Path, cfg_path: Path, out: Path, lang: str = "zh") -> Path:
    _use_font(lang)
    t = STRINGS[lang]
    names: tuple[str, ...] = t["classes"]  # type: ignore[assignment]
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
    head.text(0, 0.95, t["title"], fontsize=25, fontweight="bold", va="top")
    head.text(
        0,
        0.60,
        str(t["subtitle"]).format(n_train=len(train_df), n_test=int(cm.sum())),
        fontsize=13,
        color="#555",
        va="top",
    )
    head.text(0, 0.34, t["intro"], fontsize=10.5, color="#444", va="top", linespacing=1.65)

    fmt = {
        "hit": int(np.trace(cm)),
        "n_test": int(cm.sum()),
        "escapes": escapes,
        "defective": defective,
        "overkill": overkill,
        "good": good,
        "rarest": counts[2],
    }
    values = [
        f"{metrics['accuracy']:.1%}",
        f"{escapes / defective:.1%}",
        f"{overkill / good:.1%}",
        f"{cm[2, 2] / cm[2].sum():.0%}",
    ]
    cards = [
        (label, value, note.format(**fmt))
        for (label, note), value in zip(t["cards"], values)  # type: ignore[call-overload]
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
        ax.set_title(f"{i}  {names[i]}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    # Anchor the section label just above the sample grid so layout tweaks
    # cannot push it on top of the images.
    samples_top = gs[1, :2].get_position(fig).y1
    fig.text(0.05, samples_top + 0.018, t["samples"], fontsize=12, fontweight="bold")

    # --- confusion matrix -----------------------------------------------
    ax = fig.add_subplot(gs[1, 2:])
    sns.heatmap(
        cm,
        cmap="Oranges",
        annot=True,
        fmt="d",
        cbar=False,
        ax=ax,
        xticklabels=names,
        yticklabels=names,
        annot_kws={"fontsize": 11},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(t["cm_title"], fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(t["cm_x"], fontsize=10)
    ax.set_ylabel(t["cm_y"], fontsize=10)
    plt.setp(ax.get_xticklabels(), fontsize=9, rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), fontsize=9, rotation=0)

    # --- training curve -------------------------------------------------
    ax = fig.add_subplot(gs[2, :2])
    epochs = range(1, len(history["accuracy"]) + 1)
    ax.plot(epochs, history["accuracy"], lw=2, label=t["curve_train"])
    ax.plot(epochs, history["val_accuracy"], lw=2, label=t["curve_valid"])
    best = max(history["val_accuracy"])
    be = history["val_accuracy"].index(best) + 1
    ax.scatter([be], [best], s=70, zorder=5, color="#b5451b")
    ax.annotate(
        str(t["curve_best"]).format(epoch=be, acc=best),
        (be, best),
        textcoords="offset points",
        xytext=(-15, -45),
        fontsize=9,
        ha="center",
        color="#b5451b",
    )
    ax.set_title(t["curve_title"], fontsize=12, fontweight="bold")
    ax.set_xlabel("epoch", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    # --- imbalance ------------------------------------------------------
    ax = fig.add_subplot(gs[2, 2:])
    x = np.arange(6)
    bars = ax.bar(x, counts.values, color="#d9d9d9", edgecolor="#999")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(t["bar_y"], fontsize=10)
    ax.set_title(t["bar_title"], fontsize=12, fontweight="bold")
    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 12,
            str(t["bar_label"]).format(n=counts[i], w=weights[i]),
            ha="center",
            fontsize=8.5,
            color="#b5451b" if weights[i] > 1.5 else "#666",
            fontweight="bold" if weights[i] > 1.5 else "normal",
        )
    ax.set_ylim(0, max(counts.values) * 1.28)
    ax.grid(axis="y", alpha=0.3)
    ratio = max(counts.values) / min(counts.values)
    maxw = max(weights.values())
    ax.annotate(
        str(t["bar_note"]).format(ratio=ratio, maxw=maxw),
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
    for i, (title, body) in enumerate(t["blocks"]):  # type: ignore[call-overload]
        body = body.format(ratio=ratio, maxw=maxw, rarest=counts[2])
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
    parser.add_argument(
        "--lang", choices=sorted(STRINGS), default="zh", help="label language (default: zh)"
    )
    parser.add_argument(
        "--out", type=Path, help="output image (default: docs/portfolio[_<lang>].jpg)"
    )
    args = parser.parse_args(argv)

    suffix = "" if args.lang == "zh" else f"_{args.lang}"
    out = args.out or Path(f"docs/portfolio{suffix}.jpg")
    out = build(args.run, args.config, out, args.lang)
    size_kb = out.stat().st_size / 1024
    log.info("wrote %s (%.0f KB)", out, size_kb)
    if size_kb > 1024:
        log.warning("over 1 MB - 104 recommends staying under that for image uploads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
