"""Regenerate every committed artifact under ``results/`` in one command.

The original notebooks trained on a private word list (``/home/dh/dataset``).
To make the full pipeline reproducible with **no external data**, this script
synthesizes a deterministic, *learnable* word corpus and runs the whole flow:

    train  ->  export weights to CSV  ->  quantize to Q1.15 + emit C header

Each generated word is ``prefix + successor(prefix[-1])`` (e.g. ``ab`` -> last
char ``c``), so the last character is a deterministic function of the prefix and
the RNN can actually learn it — giving a meaningful (high) test accuracy.

Artifacts written to ``results/``:

* ``train.log`` / ``export.log`` / ``quantize.log`` / ``predict.log``
* ``metrics.json``            -- final test accuracy + quantization scale
* ``rnn_weights_q15.h``       -- generated FPGA C header (Q1.15 int16 weights)
* ``Wx.csv``                  -- one exported weight matrix (sample)

Usage
-----
    python run_all.py
"""

import json
import os
import random
import re
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO_ROOT, "results")
DATA_DIR = os.path.join(REPO_ROOT, "data")
# subprocess args stay relative to REPO_ROOT (run() sets cwd) so logs don't
# embed machine-specific absolute paths
WEIGHTS = "weights/nextword_weights.pth"
CSV_DIR = "weights_csv"
HEADER = "weights_for_FPGA/rnn_weights_q15.h"


def make_corpus(path, n, seed):
    rng = random.Random(seed)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n):
            length = rng.randint(2, 6)               # prefix length
            prefix = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz")
                             for _ in range(length))
            last = chr((ord(prefix[-1]) - 97 + 1) % 26 + 97)  # successor
            f.write(prefix + last + "\n")


def run(name, args):
    log_path = os.path.join(OUT_DIR, f"{name}.log")
    print(f"  {name} ...")
    proc = subprocess.run([sys.executable] + args, cwd=REPO_ROOT,
                          capture_output=True, text=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n[stderr]\n" + proc.stderr)
    if proc.returncode != 0:
        print(f"    WARNING: {name} exited with {proc.returncode} (see log)")
    return proc.stdout


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    make_corpus(os.path.join(DATA_DIR, "train.txt"), 4000, seed=0)
    make_corpus(os.path.join(DATA_DIR, "test.txt"), 800, seed=1)
    print("Synthetic corpus written to data/train.txt, data/test.txt")

    train_out = run("train", [
        "-m", "src.train", "--epochs", "15", "--batch-size", "256",
        "--seed", "0", "--out", WEIGHTS])
    run("export", ["-m", "src.weight_export", "--weights", WEIGHTS,
                   "--out", CSV_DIR, "--inspect"])
    run("quantize", ["-m", "src.quantize", "--csv", CSV_DIR, "--header", HEADER])
    run("predict", ["-m", "src.predict", "--weights", WEIGHTS])

    # copy the headline artifacts into results/
    header_abs = os.path.join(REPO_ROOT, HEADER)
    wx_abs = os.path.join(REPO_ROOT, CSV_DIR, "Wx.csv")
    if os.path.exists(header_abs):
        shutil.copy(header_abs, os.path.join(OUT_DIR, "rnn_weights_q15.h"))
    if os.path.exists(wx_abs):
        shutil.copy(wx_abs, os.path.join(OUT_DIR, "Wx.csv"))

    # parse final test accuracy from the training log
    accs = re.findall(r"test_acc\s+([\d.]+)", train_out)
    metrics = {
        "final_test_acc": float(accs[-1]) if accs else None,
        "epochs": 15,
        "quant_format": "Q1.15 int16 (scale=32768)",
        "vocab_size": 26,
        "hidden_size": 128,
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. final_test_acc={metrics['final_test_acc']}. Artifacts under results/.")


if __name__ == "__main__":
    main()
