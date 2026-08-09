# Results

Two result sets are committed:

- [`results/`](results/) — **synthetic corpus**, produced by a single reproducible
  command with no external data: `python run_all.py`
- [`results/real_data/`](results/real_data/) — **real English word list** (the
  dataset and the resulting weights are in
  [Releases](https://github.com/donghyeoni/RNN-HW-accelerator/releases))

| | synthetic corpus | real word list |
| --- | --- | --- |
| train / test words | 4,000 / 800 | 263,739 / 13,881 |
| task difficulty | deterministic rule → 1.00 reachable | ambiguous (e.g. `hel` → `l`? `p`?) |
| final test accuracy | **1.00** | **0.66** |
| purpose | pipeline verification | the actual model |

## Synthetic corpus

### Task setup

Each synthesized word is `prefix + successor(prefix[-1])` (e.g. `ab` → last char
`c`), so the last character is a deterministic function of the prefix and the
vanilla RNN can learn it. 4000 train / 800 test words, 15 epochs.

### Training

Final metrics ([`results/metrics.json`](results/metrics.json), full log
[`results/train.log`](results/train.log)):

| metric | value |
| --- | --- |
| Final test accuracy | **1.00** |
| Epochs | 15 |
| Hidden size | 128 |

The RNN reaches 100% test accuracy — it correctly learns the successor rule. The
inference demo ([`results/predict.log`](results/predict.log)) confirms it:
`hell` → `m` (l→m), `kore` → `f` (e→f), `knoc` → `d` (c→d).

### Quantization → FPGA (Q1.15)

The trained weights are exported to per-tensor CSV
([`results/export.log`](results/export.log)) and quantized to **Q1.15 signed
int16** (`round(x * 2^15)`, clipped to `[-32768, 32767]`), then emitted as a C
header ([`results/quantize.log`](results/quantize.log)):

| tensor | shape | dtype |
| --- | --- | --- |
| Wx | 26×128 | int16 |
| Wh | 128×128 | int16 |
| b | 128 | int16 |
| Wo | 128×26 | int16 |
| bo | 26 | int16 |

Committed samples of the generated hardware artifacts:
[`results/rnn_weights_q15.h`](results/rnn_weights_q15.h) (the FPGA C header) and
[`results/Wx.csv`](results/Wx.csv).

## Real word list

The same pipeline run on the real English word list from
[Releases](https://github.com/donghyeoni/RNN-HW-accelerator/releases)
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
