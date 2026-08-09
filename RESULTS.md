# Results

Two result sets are committed:

- [`results/synthetic/`](results/synthetic/) — **synthetic corpus**, produced by a single reproducible
  command with no external data: `python run_all.py`
- [`results/real_data/`](results/real_data/) — **real English word list** (the
  dataset and the resulting weights are in
  [Releases](https://github.com/donghyeoni/vanilla-rnn-fpga-quantization/releases))

| | synthetic corpus | real word list |
| --- | --- | --- |
| train / test words | 4,000 / 800 | 263,739 / 13,881 |
| task difficulty | deterministic rule → 1.00 reachable | ambiguous (e.g. `hel` → `l`? `p`?) |
| final test accuracy | **1.00** | **0.66** |
| float vs Q1.15 prediction agreement | **1.0000** | **0.7792** |
| purpose | pipeline verification | the actual model |

## Synthetic corpus

### Task setup

Each synthesized word is `prefix + successor(prefix[-1])` (e.g. `ab` → last char
`c`), so the last character is a deterministic function of the prefix and the
vanilla RNN can learn it. 4000 train / 800 test words, 15 epochs.

### Training

Final metrics ([`results/synthetic/metrics.json`](results/synthetic/metrics.json), full log
[`results/synthetic/train.log`](results/synthetic/train.log)):

| metric | value |
| --- | --- |
| Final test accuracy | **1.00** |
| Epochs | 15 |
| Hidden size | 128 |

The RNN reaches 100% test accuracy — it correctly learns the successor rule. The
inference demo ([`results/synthetic/predict.log`](results/synthetic/predict.log)) confirms it:
`hell` → `m` (l→m), `kore` → `f` (e→f), `knoc` → `d` (c→d).

### Quantization → FPGA (Q1.15)

The trained weights are exported to per-tensor CSV
([`results/synthetic/export.log`](results/synthetic/export.log)) and quantized to **Q1.15 signed
int16** (`round(x * 2^15)`, clipped to `[-32768, 32767]`), then emitted as a C
header ([`results/synthetic/quantize.log`](results/synthetic/quantize.log)):

| tensor | shape | dtype |
| --- | --- | --- |
| Wx | 26×128 | int16 |
| Wh | 128×128 | int16 |
| b | 128 | int16 |
| Wo | 128×26 | int16 |
| bo | 26 | int16 |

Committed samples of the generated hardware artifacts:
[`results/synthetic/rnn_weights_q15.h`](results/synthetic/rnn_weights_q15.h) (the FPGA C header) and
[`results/synthetic/Wx.csv`](results/synthetic/Wx.csv).

### Fixed-point validation

`src/validate_quantization.py` runs the float model and the integer-only reference
(`rnn_step_fixed` — int16 weights, wide accumulate, arithmetic shift, saturate
before tanh; the same datapath the FPGA implements) over the full test set and
compares predictions ([`results/synthetic/validate.log`](results/synthetic/validate.log)):

| | float | Q1.15 fixed |
| --- | --- | --- |
| test accuracy | 1.0000 | **1.0000** |
| prediction agreement | — | **1.0000 (800/800)** |

All synthetic-corpus weights fit inside the Q1.15 range `[-1, 1)`, so quantization
is essentially lossless and the integer datapath reproduces the float model exactly.

## Real word list

The same pipeline run on the real English word list from
[Releases](https://github.com/donghyeoni/vanilla-rnn-fpga-quantization/releases)
(`train_original.txt` / `test_original.txt` → `data/train.txt` / `data/test.txt`),
20 epochs, seed 0. Artifacts under [`results/real_data/`](results/real_data/).

### Training

Full log: [`results/real_data/train.log`](results/real_data/train.log),
metrics: [`results/real_data/metrics.json`](results/real_data/metrics.json).

| metric | value |
| --- | --- |
| Final test accuracy | **0.66** |
| Epochs | 20 |
| Hidden size | 128 |

Unlike the synthetic rule, real spelling is ambiguous (`hel` → `hell`? `help`?),
so accuracy saturates well below 1.0 — this is the expected ceiling of the task,
not a pipeline defect (the synthetic run above shows the pipeline itself is sound).

The released `nextword_weights_original.pth` scores **test_acc 0.6750** on the same
test set ([`results/real_data/eval_original.log`](results/real_data/eval_original.log)).

The inference demo ([`results/real_data/predict.log`](results/real_data/predict.log))
behaves like English: `appl` → `e`, `tabl` → `e`.

### Quantization → FPGA (Q1.15)

Export / quantization logs and the generated hardware artifacts from the real-data
weights: [`results/real_data/export.log`](results/real_data/export.log),
[`results/real_data/quantize.log`](results/real_data/quantize.log),
[`results/real_data/rnn_weights_q15.h`](results/real_data/rnn_weights_q15.h),
[`results/real_data/Wx.csv`](results/real_data/Wx.csv).

### Fixed-point validation

The same float-vs-integer comparison on the released original weights over all
13,881 test words ([`results/real_data/validate.log`](results/real_data/validate.log)):

| | float | Q1.15 fixed |
| --- | --- | --- |
| test accuracy | 0.6750 | **0.6272** |
| prediction agreement | — | **0.7792** |

Unlike the synthetic weights, the real-data weights do **not** all fit in Q1.15:
38.8% of `Wx` entries lie outside `[-1, 1)` and get clipped to ±1 at quantization
time, which costs ≈4.8%p of accuracy. `tanh` saturation keeps the recurrence
usable despite the clipping. A known improvement would be a wider integer format
for `Wx` (e.g. Q4.12) or a per-tensor scale factor — the header format and this
validation harness are where that change would land.
