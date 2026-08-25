"""
포맷 실험 공통 유틸 (shared helpers for the Q-format experiments).

``format_sweep.py``(균일 포맷 스윕)와 ``ablation.py``(오차 원인 분리)가 함께 쓰는
로딩·측정·표 출력 코드를 모아 둔다.
"""

import csv
import json
import os
import string

import numpy as np
import torch

from . import config
from . import fixedpoint as fp
from .model import load_words_from_txt

STOI = {ch: i for i, ch in enumerate(string.ascii_lowercase)}


def load_params(weights_path):
    """.pth state_dict -> {텐서 이름: float32 numpy 배열}."""
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    return {name: state[name].numpy().astype(np.float32) for name in config.PARAM_NAMES}


def load_eval_set(test_path):
    """테스트 단어 파일 -> (prefix 인덱스 시퀀스 목록, 정답 인덱스 목록)."""
    words = load_words_from_txt(test_path)
    id_seqs = [[STOI[ch] for ch in w[:-1]] for w in words]
    targets = [STOI[w[-1]] for w in words]
    return id_seqs, targets


def float_predictions(params, id_seqs):
    """float 기준 예측을 한 번만 계산해 두고 모든 포맷에서 재사용한다."""
    return [fp.predict_float(params, ids) for ids in id_seqs]


def weight_stats(params):
    """텐서별 absmax와, 그 값을 담기 위해 필요한 최소 정수부 포맷."""
    rows = []
    for name in config.PARAM_NAMES:
        absmax = float(np.abs(params[name]).max())
        bits = fp.fit_frac_bits(absmax)
        rows.append({
            "tensor": name,
            "shape": "x".join(str(d) for d in params[name].shape),
            "absmax": absmax,
            "outside_pm1_pct": fp.clip_fraction(params[name], 15) * 100.0,
            "required_format": fp.qname(bits),
            "required_frac_bits": bits,
        })
    return rows


def print_weight_stats(rows):
    print(f"{'tensor':<7}{'shape':<10}{'absmax':>10}{'|w|>=1':>9}{'min format':>12}")
    print("-" * 48)
    for r in rows:
        print(f"{r['tensor']:<7}{r['shape']:<10}{r['absmax']:>10.4f}"
              f"{r['outside_pm1_pct']:>8.2f}%{r['required_format']:>12}")
    print()


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"csv written -> {path}")


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"json written -> {path}")
