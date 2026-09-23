# AOI瑕疵檢測分類

[English](README.md) | 繁體中文

使用 CNN 深度學習模型進行 AOI 瑕疵檢測。

自動光學檢測（Automated Optical Inspection, AOI）— 從產品影像中找出表面與組裝瑕疵。
目前內建一個傳統的閾值（threshold）基準偵測器；CNN 偵測器透過同一個 `Detector`
介面與模型註冊表接入。

## 專案結構

```
src/aoi_detection/
  config.py            # Config（檢測流程）+ TrainConfig（分類器訓練）
  data/
    loader.py          # 檢測 CLI 用的影像搜尋
    dataset.py         # 讀取 train.csv、8:2 切分、類別權重、CLASS_NAMES
    generators.py      # Keras ImageDataGenerator（資料增強、preprocess_input）
  models/
    threshold.py       # 傳統閾值基準偵測器
    mobilenet_v2.py    # MobileNetV2 (ImageNet) + GAP + Dropout + softmax(6)
  training/
    trainer.py         # 訓練迴圈、最佳 checkpoint、history.json、accuracy/loss 圖
    evaluate.py        # 預測 test.csv、submission.csv、混淆矩陣、指標
  pipeline/            # 前處理 -> 推論 -> 後處理 -> 報告
  utils/
    visualize.py       # EDA 圖表、訓練曲線、混淆矩陣
configs/
  default.yaml         # 檢測流程設定
  train_mobilenet_v2.yaml  # 訓練超參數（與 Colab notebook 一致）
data/raw/              # train.csv、train_images/、test.csv、test_images/（git 忽略）
notebooks/             # train_mobilenet_v2_walkthrough.ipynb（呼叫上述模組）
scripts/               # explore_data.py、train.py、evaluate.py、run_inspection.sh
runs/                  # 訓練輸出，每次執行一個資料夾（git 忽略）
tests/                 # pytest 測試
```

## 快速開始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

對整個資料夾的影像執行推論：

```bash
python -m aoi_detection.cli --config configs/default.yaml --input data/raw --output outputs/
```

這會輸出 `report.json`，並為每張未通過檢測的影像產生一張 `*_overlay.png`；
只要有任何影像未通過，程式就會以非零狀態碼結束 — 可直接當作 CI 的關卡。

## 訓練 MobileNetV2 瑕疵分類器

Colab notebook `AOI_detection_Colaboratory_(mobilenetV2).ipynb` 已拆分成上述模組。
共六個類別：`0 normal（正常）、1 void（空洞）、2 horizontal_defect（水平瑕疵）、
3 vertical_defect（垂直瑕疵）、4 edge_defect（邊緣瑕疵）、5 particle（顆粒）`。

### 1. 資料擺放

```
data/raw/
  train.csv            # ID,Label   （2528 筆）
  train_images/        # train_00000.png ...
  test.csv             # ID[,Label] （Label 可省略；有的話會計算分數）
  test_images/
```

### 2. 安裝訓練環境

```bash
pip install -e ".[train]"
```

TensorFlow 2.15（Windows 上為 CPU 版）。Python 3.9–3.11。

### 3. 資料探索

```bash
python scripts/explore_data.py
```

輸出 `runs/eda/image_info.png`、`class_samples.png`、`class_distribution.png`，
並在 log 中列出各類別的影像數量。加上 `--show` 會直接開啟視窗顯示。

### 4. 訓練

```bash
python scripts/train.py --config configs/train_mobilenet_v2.yaml
```

建議先跑一次快速測試：`python scripts/train.py --epochs 2 --run-name smoke`。

每次訓練會寫入 `runs/<時間戳記>/`：

| 檔案 | 內容 |
|---|---|
| `best_model.keras` | `val_accuracy` 最高那個 epoch 的 checkpoint |
| `history.json`、`training_log.csv` | 每個 epoch 的 `accuracy / loss / val_accuracy / val_loss` |
| `accuracy.png`、`loss.png` | 訓練 vs 驗證曲線圖 |
| `train_config.json` | 本次實際使用的超參數 |

超參數（影像尺寸、batch、epochs、學習率、dropout、資料增強、類別權重、
early stopping）都在 `configs/train_mobilenet_v2.yaml`。

### 5. 評估

```bash
python scripts/evaluate.py
```

預設使用最新一次的訓練結果（或用 `--run runs/<名稱>` 指定），對 `test.csv`
每一列做預測，輸出 `submission.csv`、`probabilities.csv`；若 `test.csv` 含有
`Label` 欄位，另外輸出 `confusion_matrix.png`、`metrics.json`，並在 log 中印出
各類別的 precision / recall 報告。`--show` 會開啟混淆矩陣視窗。

### 6. Notebook

`notebooks/train_mobilenet_v2_walkthrough.ipynb` 以同樣的步驟執行並直接在
notebook 內顯示圖表。原始的 Colab notebook 保持不變。

### 用過殺換漏檢

產線上兩種錯誤成本不同：**漏檢**（不良品被判成良品）通常遠比**過殺**（良品被
誤判成不良品）昂貴。單純的 `argmax` 把兩者一視同仁。

`normal_confidence_threshold` 讓判定變成不對稱 — 要放行需要信心，要攔下不用：

```yaml
normal_confidence_threshold: 0.8   # 0 表示停用
```

```bash
python scripts/evaluate.py --normal-threshold 0.8
```

當 `normal` 勝出但機率低於門檻時，該影像會改判成機率最高的瑕疵類。在 224px
的訓練結果上，這讓準確率 95.3% → 96.0%、漏檢 5 → 4、過殺 0 → 0。

**門檻要用 validation set 挑，不要用 test set。** 在 test 上掃門檻再回報那個
test 分數，會讓成績偏樂觀。這條規則也救不了「有信心的錯誤」：這個模型 5 件漏檢
裡有 4 件的 P(normal) > 0.97，落在真正良品的範圍內，沒有任何門檻分得開 —
那種情況需要補該瑕疵子型態的資料。

### 與 Colab notebook 的差異

- Checkpoint 改用原生 `.keras` 格式，而非舊版 `.h5`。
- 測試準確率是拿預測結果與 `test.csv` 的 `Label` 欄位比較（原 notebook 誤用了
  `ID` 欄位）。
- 混淆矩陣為 6x6 並標上類別名稱（原 notebook 寫死只有 2 個刻度）。
- 驗證集 generator 不做 shuffle，因此驗證指標是可重現的。

## 加入 CNN 偵測器

`models/base.py` 定義了介面：輸入影像，輸出二值瑕疵遮罩。
實作它、在 `models/registry.py` 註冊名稱，再從設定檔選用即可 —
流程、CLI 與報告都不需要修改。

```yaml
detector:
  name: cnn
  weights: checkpoints/best.pt
```

## 開發

```bash
ruff check src tests      # lint
ruff format src tests     # 排版
mypy src                  # 型別檢查
pytest --cov=aoi_detection
```

## 目前狀態

閾值基準偵測器可完整運作並有測試覆蓋。MobileNetV2 分類器可透過
`scripts/train.py` / `scripts/evaluate.py` 訓練與評估；將它接入檢測 CLI 的
`Detector` 介面仍待完成。

## 專案成果

<img width="1673" height="998" alt="aoi_detection_final" src="https://github.com/user-attachments/assets/6f90ed2c-1378-460b-a223-cef7c55cc4b9" />

