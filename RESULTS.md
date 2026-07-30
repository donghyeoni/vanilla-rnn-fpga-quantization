# Results

Produced by a single reproducible command (no external data — a deterministic,
learnable word corpus is synthesized with fixed seeds and torch is seeded):

```bash
python run_all.py
```

Artifacts under [`results/`](results/).

## Task setup

Each synthesized word is `prefix + successor(prefix[-1])` (e.g. `ab` → last char
`c`), so the last character is a deterministic function of the prefix and the
vanilla RNN can learn it. 4000 train / 800 test words, 15 epochs.

## Training

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

## Quantization → FPGA (Q1.15)

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

## Original notebooks

The original notebooks' logs are preserved under
[`results/notebook_reference/`](results/notebook_reference/) for provenance.
