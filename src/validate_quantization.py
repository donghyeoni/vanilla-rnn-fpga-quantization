"""
6. 양자화 검증 (fixed-point validation).

FPGA 보드 없이, float 모델과 Q1.15 정수 모델(rnn_step_fixed — 하드웨어
데이터패스와 동일한 int32 누산 + 시프트)을 같은 테스트셋에 대해 돌려
last-char 예측이 얼마나 일치하는지 측정한다. 일치율이 높으면 양자화된
가중치를 하드웨어에 그대로 실어도 모델이 유효하다는 근거가 된다.

사용 예:
    python -m src.validate_quantization --weights weights/nextword_weights.pth --test data/test.txt
"""

import argparse
import string

import numpy as np
import torch

from . import config
from .model import load_words_from_txt
from .quantize import quantize_to_int16, rnn_step_fixed


def load_float_params(weights_path):
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    return {name: state[name].numpy().astype(np.float32) for name in config.PARAM_NAMES}


def predict_float(params, ids):
    # 원본 모델과 동일한 순환: h = tanh(x@Wx + h@Wh + b), y = h@Wo + bo
    h = np.zeros(config.HIDDEN_SIZE, dtype=np.float32)
    for i in ids:
        x = np.zeros(config.VOCAB_SIZE, dtype=np.float32)
        x[i] = 1.0
        h = np.tanh(x @ params["Wx"] + h @ params["Wh"] + params["b"])
    y = h @ params["Wo"] + params["bo"]
    return int(np.argmax(y))


def predict_fixed(q, ids):
    # rnn_step_fixed는 W.T @ x 형태이므로 위 float 버전과 같은 곱 방향이다
    h = np.zeros(config.HIDDEN_SIZE, dtype=np.int16)
    for i in ids:
        x = np.zeros(config.VOCAB_SIZE, dtype=np.float32)
        x[i] = 1.0
        h, y = rnn_step_fixed(quantize_to_int16(x), h, q)
    return int(np.argmax(y.astype(np.int32)))


def main():
    parser = argparse.ArgumentParser(
        description="Compare float vs Q1.15 fixed-point predictions on the test set.")
    parser.add_argument("--weights", type=str, default=str(config.WEIGHTS_PATH),
                        help="path to the .pth state_dict to validate")
    parser.add_argument("--test", type=str, default=str(config.TEST_PATH),
                        help="path to test.txt")
    args = parser.parse_args()

    params = load_float_params(args.weights)
    q = {name: quantize_to_int16(arr) for name, arr in params.items()}

    stoi = {ch: i for i, ch in enumerate(string.ascii_lowercase)}
    words = load_words_from_txt(args.test)
    print(f"weights: {args.weights}")
    print(f"test words: {len(words)}")

    n = agree = float_ok = fixed_ok = 0
    for w in words:
        ids = [stoi[ch] for ch in w[:-1]]
        target = stoi[w[-1]]
        pf = predict_float(params, ids)
        pq = predict_fixed(q, ids)
        n += 1
        agree += pf == pq
        float_ok += pf == target
        fixed_ok += pq == target

    print(f"float    test_acc {float_ok / n:.4f}")
    print(f"fixed    test_acc {fixed_ok / n:.4f}  (Q1.15 int16 weights, wide accumulate >> {config.FRAC_BITS}, saturate before tanh)")
    print(f"float vs fixed prediction agreement: {agree / n:.4f}")


if __name__ == "__main__":
    main()
