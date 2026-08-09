# vanilla-rnn-fpga-quantization

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)

A from-scratch **vanilla RNN** trained for a *last-character completion* task, then
exported and **quantized to Q1.15 fixed-point** for deployment on an FPGA.

The RNN is implemented with raw `nn.Parameter` tensors (i.e. **not** `torch.nn.RNN`),
so the exact recurrence and its weights are fully transparent and portable to hardware.

> Note: source comments are written in Korean and are preserved intentionally.

## Overview

**Task — last-character completion.** Given a lowercase word with its final letter
removed (e.g. `"hell"`), the model predicts the missing last character (`"o"`). The
vocabulary is the 26 lowercase letters `a`–`z`. This is a **sequence-to-one**
classification problem: the model runs over the prefix and the logits at the **last
timestep** are used for the prediction.

**Recurrence.**

```
h_t = tanh(x_t @ Wx + h_{t-1} @ Wh + b)
y_t = h_t @ Wo + bo
```

- `x` shape: `(B, T, D)` — B: number of words, T: prefix length, D: vocab size (26, one-hot)
- `logits` shape: `(B, T, O)`, O = vocab size (26)

**Model config:** input = 26, hidden = 128, output = 26.
**Training:** AdamW, lr = 3e-3, gradient clipping (norm) = 1.0, batch size = 256,
20 epochs, xavier-uniform init for `Wx`/`Wo` and orthogonal init for `Wh`.

**Stack:** PyTorch (raw `nn.Parameter`), NumPy, pandas.

## Pipeline

```
train  ──▶  export CSV  ──▶  quantize to Q1.15  ──▶  C header (.h)
(.pth)      (per-tensor)     (int16)                (const int16_t arrays)
```

1. **Train** the RNN and save a PyTorch `state_dict` (`.pth`).
2. **Export** the 5 parameters (`Wx`, `Wh`, `b`, `Wo`, `bo`) to per-tensor `.csv`.
3. **Quantize** each float to **Q1.15 signed int16** via `round(x * 2^15)` then clip to
   the int16 range `[-32768, 32767]`. An integer-only reference RNN step
   (`fixed_mul` / `fixed_matvec` / `fixed_tanh` / `rnn_step_fixed`) is provided to
   validate the fixed-point math against the float model.
4. **Emit** a C header of `const int16_t` arrays for FPGA / HLS use.

## Results

| dataset | train / test words | final test accuracy |
| --- | --- | --- |
| real English word list | 263,739 / 13,881 | **0.66** (released weights: **0.675**) |
| synthetic successor-rule corpus | 4,000 / 800 | **1.00** |

On the real word list the last character is inherently ambiguous (`hel` → `hell`?
`help`?), so accuracy saturates in the high 0.6s — that is the ceiling of the task,
not a training problem. The synthetic corpus has a deterministic answer, so reaching
1.00 verifies the pipeline end-to-end. The trained model completes real words
plausibly: `appl` → `e`, `tabl` → `e`.

Full logs, metrics, and the quantized artifacts are under
[`results/`](results/) — see [RESULTS.md](RESULTS.md).

## Dataset

The model reads an English **word list** from local `.txt` files (one file for
training, one for testing). Each file is lowercased and only alphabetic runs of length
`>= 2` are kept (`load_words_from_txt`).

Two datasets are used in this project:

- **Real word list** (263,739 train / 13,881 test words) — the dataset the released
  weights were trained on. Download `train_original.txt` / `test_original.txt` from
  [Releases](https://github.com/donghyeoni/RNN-HW-accelerator/releases) and place them
  at `data/train.txt` / `data/test.txt` (or point at them via `--train` / `--test`).
  The resulting metrics live under [`results/real_data/`](results/real_data/).
- **Synthetic corpus** — **no download is required to reproduce the pipeline.**
  `run_all.py` synthesizes a deterministic, *learnable* corpus (each word is
  `prefix + successor(prefix[-1])`, e.g. `ab` → `c`) into `data/train.txt` /
  `data/test.txt` and runs the whole flow with fixed seeds — the committed results
  directly under `results/` come from this.

Any other plain-text English corpus at `data/train.txt` / `data/test.txt` works too.

## Repository structure

```
vanilla-rnn-fpga-quantization/
├── src/
│   ├── config.py         # hidden_size, frac_bits(=15), paths, hyperparameters
│   ├── model.py          # VanillaRNN + LastCharDataset, collate_lastchar, load_words_from_txt
│   ├── train.py          # train_epoch, evaluate, CLI entrypoint (build loaders, train, save .pth)
│   ├── evaluate.py       # evaluate a saved .pth on the test set (no training)
│   ├── predict.py        # predict_last_char inference / demo
│   ├── weight_export.py  # load .pth, inspect shapes, export tensors -> CSV
│   └── quantize.py        # quantize_to_int16 (Q1.15), fixed-point reference RNN, dump_c_array -> .h
├── run_all.py            # synthesize a learnable corpus + run train->export->quantize->predict
├── results/              # committed artifacts (synthetic corpus): logs, metrics.json, sample C header + CSV
│   └── real_data/        # committed artifacts from the real word-list dataset (see Releases)
├── requirements.txt
├── RESULTS.md
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Reproduce the full pipeline end-to-end on a synthetic corpus (no data needed):

```bash
python run_all.py        # writes results/ (logs, metrics.json, C header), see RESULTS.md
```

Or run the modules individually from the repository root (they use
package-relative imports, so use `python -m`):

```bash
# 1. Train (saves weights/nextword_weights.pth by default)
python -m src.train --train data/train.txt --test data/test.txt --epochs 20 --seed 0

# 2. Evaluate a saved .pth on the test set (e.g. weights downloaded from Releases)
python -m src.evaluate --weights weights/nextword_weights.pth --test data/test.txt

# 3. Inference demo
python -m src.predict --weights weights/nextword_weights.pth --prefixes hell worl appl

# 4. Export weights to per-tensor CSV
python -m src.weight_export --weights weights/nextword_weights.pth --out weights_csv

# 5. Quantize to Q1.15 and emit the C header
python -m src.quantize --csv weights_csv --header weights_for_FPGA/rnn_weights_q15.h
```

## Notes

### Generated weight artifacts (git-ignored)

The following directories are **produced by running the pipeline** and are intentionally
excluded from version control (see `.gitignore`):

- **`weights/`** — the PyTorch `.pth` state_dict of the trained model.
- **`weights_csv/`** — per-tensor CSV exports:
  `Wx` (26×128), `Wh` (128×128), `Wo` (128×26), `b` (128), `bo` (26).
- **`weights_for_FPGA/`** — a C header `rnn_weights_q15.h` containing
  `const int16_t` Q1.15 arrays for the five parameters.

A committed **sample** of these (the generated `rnn_weights_q15.h` header and
`Wx.csv`) plus the run logs and `metrics.json` live under `results/` — see
[RESULTS.md](RESULTS.md).

### Releases

Trained weights and the real word-list dataset are distributed via
[Releases](https://github.com/donghyeoni/RNN-HW-accelerator/releases):

- `nextword_weights_original.pth` — weights trained on the real word list
  (test accuracy ≈ 0.68, see [RESULTS.md](RESULTS.md)).
- `nextword_weights_synthetic.pth` — weights trained on the synthetic corpus
  (paired with the artifacts directly under `results/`).
- `train_original.txt` / `test_original.txt` — the real word-list dataset.
- `train_synthetic.txt` / `test_synthetic.txt` — one instance of the synthetic
  corpus (regenerable with `run_all.py`).

### FPGA target

The generated `rnn_weights_q15.h` exposes the five parameters as `const int16_t` arrays
in **Q1.15** format (1 sign bit, 15 fractional bits). It is intended to be `#include`d
into an FPGA/HLS project (e.g. Xilinx Vitis HLS or a Verilog design) where the recurrence
is implemented with integer arithmetic. The `rnn_step_fixed` reference in
`src/quantize.py` mirrors the intended hardware datapath: int32 accumulation for
matrix-vector products followed by an arithmetic shift right by `FRAC_BITS` back to
Q1.15, with `tanh` shown as a float reference to be replaced by a hardware-friendly
approximation (e.g. LUT) on device.
