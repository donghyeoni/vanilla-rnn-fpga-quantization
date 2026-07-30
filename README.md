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

## Dataset

The model reads a generic English **word list** from local `.txt` files (one file for
training, one for testing). Each file is lowercased and only alphabetic runs of length
`>= 2` are kept (`load_words_from_txt`).

**No dataset is required to reproduce the pipeline.** `run_all.py` synthesizes a
deterministic, *learnable* corpus (each word is `prefix + successor(prefix[-1])`,
e.g. `ab` → `c`) into `data/train.txt` / `data/test.txt` and runs the whole flow
with fixed seeds — the committed results under `results/` come from this.

You can still supply your own English word list at `data/train.txt` /
`data/test.txt` (or via `--train` / `--test`); any plain-text corpus works.

> The original notebooks hardcoded Linux paths
> (`/home/dh/dataset/words/Train/train.txt` and `.../Test/test.txt`); these are now
> configurable via `src/config.py` and CLI arguments.

## Repository structure

```
vanilla-rnn-fpga-quantization/
├── src/
│   ├── config.py         # hidden_size, frac_bits(=15), paths, hyperparameters
│   ├── model.py          # VanillaRNN + LastCharDataset, collate_lastchar, load_words_from_txt
│   ├── train.py          # train_epoch, evaluate, CLI entrypoint (build loaders, train, save .pth)
│   ├── predict.py        # predict_last_char inference / demo
│   ├── weight_export.py  # load .pth, inspect shapes, export tensors -> CSV
│   └── quantize.py        # quantize_to_int16 (Q1.15), fixed-point reference RNN, dump_c_array -> .h
├── run_all.py            # synthesize a learnable corpus + run train->export->quantize->predict
├── results/              # committed artifacts: logs, metrics.json, sample C header + CSV
│   └── notebook_reference/  # logs preserved from the original notebooks
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

# 2. Inference demo
python -m src.predict --weights weights/nextword_weights.pth --prefixes hell worl appl

# 3. Export weights to per-tensor CSV
python -m src.weight_export --weights weights/nextword_weights.pth --out weights_csv

# 4. Quantize to Q1.15 and emit the C header
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
[RESULTS.md](RESULTS.md). The original notebooks have been removed; their logs
are preserved under `results/notebook_reference/`.

### Weight filename

The original notebooks referenced `nextword_weights2.pth`, but the file actually present
on disk was `nextword_weights.pth`. This project **standardizes on
`nextword_weights.pth`** (see `WEIGHTS_FILENAME` in `src/config.py`).

### FPGA target

The generated `rnn_weights_q15.h` exposes the five parameters as `const int16_t` arrays
in **Q1.15** format (1 sign bit, 15 fractional bits). It is intended to be `#include`d
into an FPGA/HLS project (e.g. Xilinx Vitis HLS or a Verilog design) where the recurrence
is implemented with integer arithmetic. The `rnn_step_fixed` reference in
`src/quantize.py` mirrors the intended hardware datapath: int32 accumulation for
matrix-vector products followed by an arithmetic shift right by `FRAC_BITS` back to
Q1.15, with `tanh` shown as a float reference to be replaced by a hardware-friendly
approximation (e.g. LUT) on device.
