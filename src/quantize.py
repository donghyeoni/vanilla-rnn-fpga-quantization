"""
고정소수점 포멧(16비트 고정소수점: Q1.15)
가중치를 정수로 양자화

파이프라인:
  1. per-tensor CSV 로드
  2. float -> Q1.15 signed int16 양자화  (round(x * 2^15), int16 clip)
  3. 정수 전용 레퍼런스 RNN step 구현 (검증용)
  4. FPGA/HLS용 C 헤더(const int16_t 배열) 생성

사용 예:
    python -m src.quantize --csv weights_csv --header weights_for_FPGA/rnn_weights_q15.h
"""

import argparse
import os

import numpy as np

from . import config

FRAC_BITS = config.FRAC_BITS
SCALE = config.SCALE   # 2 ** FRAC_BITS


# -----------------------------------------------------------------------------
# 0. 가중치 양자화
# -----------------------------------------------------------------------------
def quantize_to_int16(arr):
    x_scaled = np.round(arr * SCALE)
    x_clipped = np.clip(x_scaled, config.INT16_MIN, config.INT16_MAX)
    return x_clipped.astype(np.int16)


def load_and_quantize(csv_dir):
    Wx_f = np.loadtxt(os.path.join(csv_dir, "Wx.csv"), delimiter=",")
    Wh_f = np.loadtxt(os.path.join(csv_dir, "Wh.csv"), delimiter=",")
    b_f = np.loadtxt(os.path.join(csv_dir, "b.csv"), delimiter=",")
    Wo_f = np.loadtxt(os.path.join(csv_dir, "Wo.csv"), delimiter=",")
    bo_f = np.loadtxt(os.path.join(csv_dir, "bo.csv"), delimiter=",")

    return {
        "Wx": quantize_to_int16(Wx_f),
        "Wh": quantize_to_int16(Wh_f),
        "b": quantize_to_int16(b_f),
        "Wo": quantize_to_int16(Wo_f),
        "bo": quantize_to_int16(bo_f),
    }


# -----------------------------------------------------------------------------
# 1. 정수 버전 RNN 설계 및 출력 비교
# -----------------------------------------------------------------------------
def fixed_mul(a, b):
    # Q1.15 × Q1.15 → Q1.15 (중간 곱은 int32로, 끝에 시프트)
    tmp = (a.astype(np.int32) * b.astype(np.int32))
    return (tmp >> FRAC_BITS).astype(np.int16)


def saturate_int16(a):
    # 랩어라운드 대신 포화(saturation) — 하드웨어 고정소수점의 표준 동작
    return np.clip(a, config.INT16_MIN, config.INT16_MAX).astype(np.int16)


def fixed_matvec(W, x):
    # W: (out, in), x: (in,)
    # 누산은 넓은 정수로 유지한다 (int16 곱 128개의 합은 int32도 넘칠 수 있다 —
    # 하드웨어에서는 DSP의 넓은 누산기(예: 48비트)에 해당). 시프트 후에도
    # int16으로 줄이지 않고 넓은 타입으로 반환하고, 필요한 지점에서만 포화시킨다.
    tmp = W.astype(np.int64) @ x.astype(np.int64)
    return tmp >> FRAC_BITS


def fixed_tanh(x):
    # 간단하게: float로 잠깐 바꿔서 tanh 후 다시 양자화 (레퍼런스 용)
    x_f = x.astype(np.float32) / SCALE
    y_f = np.tanh(x_f)
    return quantize_to_int16(y_f)


def rnn_step_fixed(x_t_q, h_prev_q, q):
    # x_t_q, h_prev_q : Q1.15 정수 벡터
    # q : 양자화된 가중치 딕셔너리 {Wx, Wh, b, Wo, bo}
    # pre-activation은 tanh에 넣기 직전에만 int16으로 포화시킨다
    # (tanh는 |z|>~3에서 어차피 ±1로 포화하므로 정보 손실이 거의 없다).
    z = fixed_matvec(q["Wx"].T, x_t_q) + fixed_matvec(q["Wh"].T, h_prev_q) + q["b"].astype(np.int64)
    h_t = fixed_tanh(saturate_int16(z))
    # 출력 로짓은 argmax 비교용이므로 포화 없이 넓은 정수 그대로 반환한다.
    y = fixed_matvec(q["Wo"].T, h_t) + q["bo"].astype(np.int64)
    return h_t, y


# -----------------------------------------------------------------------------
# 2. FPGA에 사용할 형태로 저장
# -----------------------------------------------------------------------------
def dump_c_array(name, arr, file):
    if arr.ndim == 2:
        h, w = arr.shape
        file.write(f"const int16_t {name}[{h}][{w}] = {{\n")
        for i in range(h):
            row = ", ".join(str(int(v)) for v in arr[i])
            file.write(f"    {{{row}}},\n")
        file.write("};\n\n")
    elif arr.ndim == 1:
        n = arr.shape[0]
        row = ", ".join(str(int(v)) for v in arr)
        file.write(f"const int16_t {name}[{n}] = {{{row}}};\n\n")


def write_header(q, header_path):
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    with open(header_path, "w") as f:
        f.write("#include <stdint.h>\n\n")
        dump_c_array("Wx", q["Wx"], f)
        dump_c_array("Wh", q["Wh"], f)
        dump_c_array("b", q["b"], f)
        dump_c_array("Wo", q["Wo"], f)
        dump_c_array("bo", q["bo"], f)
    # include "rnn_weights_q15.h"로 사용
    # HLS, Verilog Vitis 어디에 쓰는 지 확인
    print(f"C header written -> {header_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize CSV weights to Q1.15 int16 and emit a C header.")
    parser.add_argument("--csv", type=str, default=str(config.WEIGHTS_CSV_DIR),
                        help="directory containing per-tensor CSVs")
    parser.add_argument("--header", type=str, default=str(config.C_HEADER_PATH),
                        help="output C header path")
    args = parser.parse_args()

    q = load_and_quantize(args.csv)
    for name, arr in q.items():
        print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    write_header(q, args.header)


if __name__ == "__main__":
    main()
