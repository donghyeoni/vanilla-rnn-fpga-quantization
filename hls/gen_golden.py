"""
골든 벡터 생성 (golden vectors for HLS C/RTL simulation).

``src/fixedpoint.py``의 ``FixedRNN``을 돌려 ``<len> <ids...> <pred>`` 형식의
텍스트를 뽑는다. ``hls/rnn_kernel_tb.cpp``가 이 파일을 읽어 커널 출력과 전건
비교하므로, 여기서 만든 예측이 HLS 커널의 정답 기준이 된다.

**이 스크립트의 핵심은 벡터 생성이 아니라 검증이다.** 커널이 컴파일 시점에
``#include``하는 C 헤더와, 골든 벡터를 만든 파이썬 모델이 같은 데이터·같은 포맷을
쓰고 있는지 먼저 확인한다 (포맷 상수, 시프트량, tanh 표, 양자화된 가중치 전체).
둘이 어긋난 상태로 co-simulation을 돌리면 원인 불명의 불일치만 남는다.

사용 예:
    python hls/gen_golden.py \
        --weights weights/nextword_weights_original.pth \
        --test    data/test_original.txt \
        --header  results/tanh_lut_study/rnn_weights_mixed_lut.h \
        --n 512 --out hls/golden.txt
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# Windows 콘솔(cp949)에서 한글 출력이 깨지는 것을 막는다
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config                      # noqa: E402
from src import fixedpoint as fp            # noqa: E402
from src.tanh_lut import TanhLUT            # noqa: E402

# src.experiment은 torch를 끌어온다. 헤더 파서만 쓰는 hls/selftest.py가
# torch 없이 이 모듈을 import할 수 있도록 main 안에서 늦게 불러온다.

MAX_SEQ_LEN = 32        # hls/rnn_kernel.h의 MAX_SEQ_LEN과 일치해야 한다


# -----------------------------------------------------------------------------
# C 헤더 파서
# -----------------------------------------------------------------------------
def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def parse_header(path):
    """생성된 C 헤더 -> ({매크로 이름: int}, {배열 이름: np.int16 배열})."""
    src = strip_comments(Path(path).read_text(encoding="utf-8"))

    defines = {}
    for name, value in re.findall(r"#define\s+(\w+)\s+(-?\d+)", src):
        defines[name] = int(value)

    arrays = {}
    pattern = r"const\s+int16_t\s+(\w+)\s*((?:\[\s*\d+\s*\])+)\s*=\s*\{(.*?)\}\s*;"
    for name, dims_txt, body in re.findall(pattern, src, flags=re.S):
        dims = [int(d) for d in re.findall(r"\d+", dims_txt)]
        values = [int(v) for v in re.findall(r"-?\d+", body)]
        expected = int(np.prod(dims))
        if len(values) != expected:
            raise ValueError(f"{name}: 원소 수 불일치 "
                             f"(헤더 {len(values)}개, 선언 {dims} = {expected}개)")
        arrays[name] = np.array(values, dtype=np.int64).reshape(dims).astype(np.int16)

    return defines, arrays


# -----------------------------------------------------------------------------
# 헤더 대 파이썬 모델 정합성 검사
# -----------------------------------------------------------------------------
def check_consistency(defines, arrays, net, lut):
    """헤더의 상수·배열이 FixedRNN/TanhLUT와 완전히 같은지 확인한다."""
    problems = []

    def cmp(label, got, want):
        if got != want:
            problems.append(f"{label}: 헤더 {got} != 파이썬 {want}")

    fmt = net.fmt
    for name in ("Wx", "Wh", "b", "Wo", "bo", "x", "h", "z", "y"):
        key = f"FRAC_BITS_{name.upper()}"
        if key in defines:
            cmp(key, defines[key], getattr(fmt, name))

    for key, want in (("SHIFT_X", net.shift_x), ("SHIFT_H", net.shift_h),
                      ("SHIFT_B", net.shift_b), ("SHIFT_O", net.shift_o),
                      ("SHIFT_BO", net.shift_bo)):
        if key in defines:
            cmp(key, defines[key], want)

    if lut is not None:
        cmp("TANH_LUT_ENTRIES",   defines.get("TANH_LUT_ENTRIES"),   lut.entries)
        cmp("TANH_LUT_SAT_LIMIT", defines.get("TANH_LUT_SAT_LIMIT"), lut.sat_limit)
        cmp("TANH_LUT_STEP",      defines.get("TANH_LUT_STEP"),      lut.step)
        cmp("TANH_LUT_STEP_LOG2", defines.get("TANH_LUT_STEP_LOG2"), lut.step_log2)
        cmp("TANH_LUT_Z_FRAC",    defines.get("TANH_LUT_Z_FRAC"),    lut.z_frac_bits)
        cmp("TANH_LUT_H_FRAC",    defines.get("TANH_LUT_H_FRAC"),    lut.h_frac_bits)
        if "TANH_LUT" in arrays:
            if not np.array_equal(arrays["TANH_LUT"], lut.table.astype(np.int16)):
                problems.append("TANH_LUT: 표 값이 다르다")

    # 양자화된 가중치 전체 비교 (헤더가 진짜 이 포맷으로 만들어졌는지)
    for name in config.PARAM_NAMES:
        if name not in arrays:
            problems.append(f"{name}: 헤더에 배열이 없다")
            continue
        want = net.q[name]
        got = arrays[name]
        if got.shape != want.shape:
            problems.append(f"{name}: shape {got.shape} != {want.shape}")
        elif not np.array_equal(got, want):
            diff = int((got != want).sum())
            problems.append(f"{name}: {diff}개 원소가 다르다 "
                            f"(헤더가 다른 포맷/가중치로 생성됐을 가능성)")

    return problems


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="HLS 테스트벤치용 골든 벡터 생성")
    ap.add_argument("--weights", default=str(config.WEIGHTS_PATH),
                    help="학습된 .pth (기본: weights/nextword_weights.pth)")
    ap.add_argument("--test", default=str(config.TEST_PATH),
                    help="테스트 단어 파일")
    ap.add_argument("--header", required=True,
                    help="커널이 include하는 생성된 C 헤더")
    ap.add_argument("--out", default=str(Path(__file__).parent / "golden.txt"))
    ap.add_argument("--n", type=int, default=512,
                    help="벡터 개수 (0이면 전체). co-sim은 느리므로 기본은 512")
    ap.add_argument("--no-lut", action="store_true",
                    help="tanh를 표 대신 float 참조로 계산 (헤더에 표가 없을 때)")
    args = ap.parse_args()

    from src import experiment as ex

    params = ex.load_params(args.weights)
    id_seqs, targets = ex.load_eval_set(args.test)

    # 헤더는 --format auto (= FixedFormat.from_weights)로 생성된 것을 전제한다.
    fmt = fp.FixedFormat.from_weights(params)

    defines, arrays = parse_header(args.header)

    lut = None
    if not args.no_lut and "TANH_LUT" in arrays:
        z_frac = defines["TANH_LUT_Z_FRAC"]
        lut = TanhLUT(z_frac_bits=z_frac,
                      h_frac_bits=defines["TANH_LUT_H_FRAC"],
                      sat_bound=defines["TANH_LUT_SAT_LIMIT"] / (2.0 ** z_frac),
                      step=defines["TANH_LUT_STEP"])

    net = fp.FixedRNN(params, fmt, tanh_lut=lut)

    print(f"header  : {args.header}")
    print(f"format  : {fmt.label}")
    print(fmt.describe())
    print(f"shifts  : x={net.shift_x} h={net.shift_h} b={net.shift_b} "
          f"o={net.shift_o} bo={net.shift_bo}")
    print(f"tanh    : {lut.describe() if lut else 'float reference (no LUT)'}")

    problems = check_consistency(defines, arrays, net, lut)
    if problems:
        print("\n[FAIL] 헤더와 파이썬 모델이 일치하지 않는다:")
        for p in problems:
            print(f"  - {p}")
        print("\n헤더를 다시 생성하거나 --header 경로를 확인할 것:")
        print("  python -m src.quantize --csv weights_csv --format auto "
              "--tanh-lut --header weights_for_FPGA/rnn_weights_mixed_lut.h")
        return 1
    print("\n[OK] 포맷 상수 / 시프트 / tanh 표 / 양자화 가중치 전부 일치")

    # 길이 상한을 넘는 단어는 커널이 처리할 수 없으므로 제외한다
    keep = [(ids, tgt) for ids, tgt in zip(id_seqs, targets)
            if 0 < len(ids) <= MAX_SEQ_LEN]
    dropped = len(id_seqs) - len(keep)
    if args.n > 0:
        keep = keep[:args.n]

    lines, correct = [], 0
    for ids, tgt in keep:
        pred = net.predict(ids)
        correct += (pred == tgt)
        lines.append(f"{len(ids)} " + " ".join(str(i) for i in ids) + f" {pred}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    head = [
        "# HLS 골든 벡터 -- src/fixedpoint.py FixedRNN 출력",
        f"# weights: {args.weights}",
        f"# test   : {args.test}",
        f"# header : {args.header}",
        f"# format : {fmt.label}  |  tanh: {lut.label() if lut else 'float'}",
        "# 형식   : <len> <ids...> <pred>",
    ]
    out.write_text("\n".join(head + lines) + "\n", encoding="utf-8")

    print(f"\nvectors : {len(lines)}  -> {out}")
    if dropped:
        print(f"dropped : {dropped} (길이 > MAX_SEQ_LEN={MAX_SEQ_LEN})")
    print(f"fixed acc on these vectors: {correct / len(lines):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
