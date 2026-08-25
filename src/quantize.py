"""
CSV 가중치 -> int16 고정소수점 -> FPGA용 C 헤더.

포맷은 균일(모든 텐서 동일)일 수도, 텐서별로 다를 수도 있다:

    --format 15      모든 텐서 Q1.15 (기존 동작, 기본값)
    --format 12      모든 텐서 Q4.12
    --format auto    텐서별 absmax로 최적 포맷 선택
    --format mixed   auto + 테스트셋으로 z/y 활성값 범위 캘리브레이션
                     (--calib 로 데이터 경로 지정)

정수 데이터패스 자체(시프트 정렬, 포화, tanh)는 ``fixedpoint.py``에 있다.
이 모듈은 그 포맷 명세를 받아 하드웨어가 쓸 산출물을 뽑는 역할만 한다.

생성되는 헤더에는 가중치 배열뿐 아니라 **누산기 시프트량**이 함께 들어간다.
텐서마다 소수부 비트 수가 다르면 곱셈 결과의 정렬 시프트도 달라지므로,
HLS/RTL 쪽에서 이 상수를 그대로 쓰면 파이썬 모델과 비트 단위로 일치한다.

사용 예:
    python -m src.quantize --csv weights_csv --header weights_for_FPGA/rnn_weights_q15.h
    python -m src.quantize --csv weights_csv --format auto \
        --header weights_for_FPGA/rnn_weights_mixed.h
"""

import argparse
import os

import numpy as np

from . import config
from . import fixedpoint as fp

FRAC_BITS = config.FRAC_BITS
SCALE = config.SCALE   # 2 ** FRAC_BITS


# -----------------------------------------------------------------------------
# 0. 가중치 양자화
# -----------------------------------------------------------------------------
def quantize_to_int16(arr, frac_bits=FRAC_BITS):
    """실수 배열 -> Q(16-f).f int16. 기본값은 기존과 같은 Q1.15."""
    return fp.quantize(arr, frac_bits)


def load_csv_params(csv_dir):
    """텐서별 CSV를 읽어 float 배열 딕셔너리로 돌려준다."""
    params = {}
    for name in config.PARAM_NAMES:
        arr = np.loadtxt(os.path.join(csv_dir, f"{name}.csv"), delimiter=",")
        params[name] = np.atleast_1d(arr)
    return params


def quantize_params(params, fmt):
    """포맷 명세에 따라 텐서별로 양자화한다."""
    return {name: fp.quantize(params[name], getattr(fmt, name))
            for name in config.PARAM_NAMES}


def load_and_quantize(csv_dir, fmt=None):
    """CSV 로드 + 양자화 (fmt 생략 시 기존과 동일한 균일 Q1.15)."""
    params = load_csv_params(csv_dir)
    fmt = fmt or fp.FixedFormat.uniform(FRAC_BITS)
    return quantize_params(params, fmt)


# -----------------------------------------------------------------------------
# 1. 포맷 선택
# -----------------------------------------------------------------------------
def resolve_format(spec, params, calib_path=None):
    """CLI의 --format 문자열을 FixedFormat으로 해석한다."""
    if spec == "auto":
        return fp.FixedFormat.from_weights(params)
    if spec == "mixed":
        if not calib_path:
            raise SystemExit("--format mixed requires --calib <word list>")
        from .experiment import load_eval_set
        id_seqs, _ = load_eval_set(calib_path)
        return fp.FixedFormat.calibrate(params, id_seqs[:config.CALIB_SAMPLES])
    try:
        frac_bits = int(spec)
    except ValueError:
        raise SystemExit(f"unknown format: {spec} (expected an integer, 'auto', or 'mixed')")
    if not 0 <= frac_bits <= 15:
        raise SystemExit(f"fractional bits must be in 0..15, got {frac_bits}")
    return fp.FixedFormat.uniform(frac_bits)


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


def write_format_defines(fmt, file):
    """포맷과 누산기 시프트량을 #define으로 내보낸다."""
    file.write("/*\n")
    file.write(" * Fixed-point format (Q(16-f).f signed int16, value = raw / 2^f)\n")
    for line in fmt.describe().splitlines():
        file.write(f" *{line}\n")
    file.write(" *\n")
    file.write(" * z_t = (Wx^T x_t >> SHIFT_X) + (Wh^T h >> SHIFT_H) + (b >> SHIFT_B)\n")
    file.write(" *       -> saturate to int16 -> tanh -> h_t\n")
    file.write(" * y_t = (Wo^T h_t >> SHIFT_O) + (bo >> SHIFT_BO)\n")
    file.write(" */\n\n")

    for name in fp.ALL_FIELDS:
        file.write(f"#define FRAC_BITS_{name.upper():<3s} {getattr(fmt, name)}\n")
    file.write("\n")
    file.write(f"#define SHIFT_X   {fmt.Wx + fmt.x - fmt.z}\n")
    file.write(f"#define SHIFT_H   {fmt.Wh + fmt.h - fmt.z}\n")
    file.write(f"#define SHIFT_B   {fmt.b - fmt.z}\n")
    file.write(f"#define SHIFT_O   {fmt.Wo + fmt.h - fmt.y}\n")
    file.write(f"#define SHIFT_BO  {fmt.bo - fmt.y}\n\n")


def write_header(q, header_path, fmt=None):
    fmt = fmt or fp.FixedFormat.uniform(FRAC_BITS)
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    with open(header_path, "w") as f:
        f.write("#include <stdint.h>\n\n")
        write_format_defines(fmt, f)
        for name in config.PARAM_NAMES:
            dump_c_array(name, q[name], f)
    print(f"C header written -> {header_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize CSV weights to int16 fixed-point and emit a C header.")
    parser.add_argument("--csv", type=str, default=str(config.WEIGHTS_CSV_DIR),
                        help="directory containing per-tensor CSVs")
    parser.add_argument("--header", type=str, default=str(config.C_HEADER_PATH),
                        help="output C header path")
    parser.add_argument("--format", type=str, default=str(FRAC_BITS),
                        help="fractional bits (e.g. 15, 12), 'auto', or 'mixed'")
    parser.add_argument("--calib", type=str, default=None,
                        help="calibration word list, required by --format mixed")
    args = parser.parse_args()

    params = load_csv_params(args.csv)
    fmt = resolve_format(args.format, params, args.calib)
    print(f"format: {fmt.label}")
    print(fmt.describe())
    print()

    q = quantize_params(params, fmt)
    for name in config.PARAM_NAMES:
        clipped = fp.clip_fraction(params[name], getattr(fmt, name)) * 100
        print(f"{name}: shape={q[name].shape}, dtype={q[name].dtype}, "
              f"format={fp.qname(getattr(fmt, name))}, clipped={clipped:.2f}%")
    write_header(q, args.header, fmt)


if __name__ == "__main__":
    main()
