"""
커널 자체 검증 (self-contained bit-exactness check).

``gen_golden.py``는 학습된 ``.pth``와 테스트 단어 파일이 있어야 하고 torch를
끌어온다. 이 스크립트는 그 둘이 없어도 돌아간다 -- **생성된 C 헤더만으로**
파이썬 참조 모델을 복원하기 때문이다. int16 가중치를 역양자화해 float 파라미터를
만들면 ``FixedRNN``이 다시 양자화할 때 원래 정수가 정확히 복원된다
(``quantize(dequantize(q, f), f) == q``).

따라서 필요한 것은 numpy와 g++뿐이다. Vitis HLS도 FPGA도 데이터셋도 필요 없다.

하는 일:
  1. C 헤더를 파싱해 포맷 상수 / 시프트 / tanh 표 / 가중치를 읽는다
  2. 같은 값으로 ``FixedRNN``을 구성하고 왕복 양자화가 무손실인지 확인한다
  3. 무작위 + 경계 시퀀스로 골든 벡터를 만든다
  4. 커널을 ``-DRNN_NO_AP_INT``로 일반 g++ 컴파일해 테스트벤치를 돌린다
  5. UNROLL_H를 바꿔가며 결과가 동일한지 확인한다
     (병렬화가 연산 의미를 바꾸지 않았음을 보이는 검사)

사용 예:
    python hls/selftest.py --header results/tanh_lut_study/rnn_weights_mixed_lut.h
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# Windows 콘솔(cp949)에서 한글 출력이 깨지는 것을 막는다
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HLS_DIR = Path(__file__).resolve().parent
ROOT = HLS_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HLS_DIR))

from src import fixedpoint as fp            # noqa: E402
from src.tanh_lut import TanhLUT            # noqa: E402
from gen_golden import parse_header, check_consistency, MAX_SEQ_LEN   # noqa: E402

VOCAB_SIZE = 26
FMT_FIELDS = ("Wx", "Wh", "b", "Wo", "bo", "x", "h", "z", "y")


def format_from_header(defines):
    """헤더가 선언한 포맷을 그대로 FixedFormat으로 옮긴다."""
    kwargs = {n: defines[f"FRAC_BITS_{n.upper()}"] for n in FMT_FIELDS}
    return fp.FixedFormat(label="from header", **kwargs)


def params_from_header(arrays, fmt):
    """int16 가중치 -> float 파라미터. FixedRNN이 다시 양자화하면 원래 정수가 된다."""
    return {n: fp.dequantize(arrays[n], getattr(fmt, n))
            for n in ("Wx", "Wh", "b", "Wo", "bo")}


def build_sequences(n, rng, max_len):
    """무작위 시퀀스 + 경계 케이스."""
    seqs = [
        [0],                                    # 최단 길이
        [25],                                   # 마지막 글자
        [7, 4, 11, 11],                         # "hell" -- README의 데모 입력
        [0] * max_len,                          # 최장 길이 + 같은 글자 반복
        [25] * max_len,                         # 최장 길이, 반대쪽 극단
        list(range(VOCAB_SIZE)),                # 전 글자 1회씩
    ]
    seqs = [s[:max_len] for s in seqs]
    while len(seqs) < n:
        length = int(rng.integers(1, max_len + 1))
        seqs.append([int(c) for c in rng.integers(0, VOCAB_SIZE, size=length)])
    return seqs[:n]


def compile_kernel(header_path, unroll, out_dir):
    """커널 + 테스트벤치를 일반 g++로 컴파일한다 (Vitis HLS 불필요)."""
    exe = out_dir / f"csim_u{unroll}.exe"
    cmd = [
        "g++", "-O2", "-std=c++14",
        "-DRNN_NO_AP_INT",                      # ap_int 대신 표준 정수 타입
        f"-DUNROLL_H={unroll}", f"-DUNROLL_O={unroll}",
        f'-DRNN_WEIGHTS_HEADER="{header_path.name}"',
        "-Wno-unknown-pragmas",                 # #pragma HLS는 g++가 모른다
        f"-I{HLS_DIR}", f"-I{header_path.parent}",
        str(HLS_DIR / "rnn_kernel.cpp"), str(HLS_DIR / "rnn_kernel_tb.cpp"),
        "-o", str(exe),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print("[FAIL] 컴파일 실패")
        print(proc.stderr[:4000])
        return None
    if proc.stderr.strip():
        print("  (컴파일 경고)")
        print("  " + proc.stderr.strip().replace("\n", "\n  ")[:1500])
    return exe


def main():
    ap = argparse.ArgumentParser(description="C 헤더만으로 커널 비트 일치 검증")
    ap.add_argument("--header", required=True, help="생성된 C 헤더")
    ap.add_argument("--n", type=int, default=400, help="테스트 시퀀스 개수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unroll", default="1,16,128",
                    help="검증할 UNROLL_H 목록 (HIDDEN_SIZE의 약수)")
    ap.add_argument("--keep", action="store_true", help="빌드 산출물을 남긴다")
    args = ap.parse_args()

    if not shutil.which("g++"):
        print("[FAIL] g++를 찾을 수 없다. C simulation은 일반 C++ 컴파일러로 한다.")
        return 1

    header_path = Path(args.header).resolve()
    defines, arrays = parse_header(header_path)

    fmt = format_from_header(defines)
    params = params_from_header(arrays, fmt)

    if "TANH_LUT" not in arrays:
        print("[FAIL] 이 헤더에는 tanh 표가 없다. 커널은 정수 전용 경로를 요구하므로")
        print("       --tanh-lut 으로 생성한 헤더가 필요하다:")
        print("  python -m src.quantize --csv weights_csv --format auto --tanh-lut "
              "--header weights_for_FPGA/rnn_weights_mixed_lut.h")
        return 1

    lut = None
    if "TANH_LUT" in arrays:
        z_frac = defines["TANH_LUT_Z_FRAC"]
        lut = TanhLUT(z_frac_bits=z_frac,
                      h_frac_bits=defines["TANH_LUT_H_FRAC"],
                      sat_bound=defines["TANH_LUT_SAT_LIMIT"] / (2.0 ** z_frac),
                      step=defines["TANH_LUT_STEP"])

    net = fp.FixedRNN(params, fmt, tanh_lut=lut)

    print(f"header  : {header_path}")
    print(f"format  : " + "  ".join(f"{n}=Q{16 - getattr(fmt, n)}.{getattr(fmt, n)}"
                                    for n in FMT_FIELDS))
    print(f"shifts  : x={net.shift_x} h={net.shift_h} b={net.shift_b} "
          f"o={net.shift_o} bo={net.shift_bo}")
    print(f"tanh    : {lut.label() if lut else 'float reference'}")

    # 왕복 양자화가 무손실이어야 이 검증 전체가 성립한다
    problems = check_consistency(defines, arrays, net, lut)
    if problems:
        print("\n[FAIL] 헤더 복원이 정확하지 않다:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("[OK]    헤더 -> FixedRNN 복원 무손실 (상수 / 시프트 / tanh 표 / 가중치)")

    # 골든 벡터
    rng = np.random.default_rng(args.seed)
    seqs = build_sequences(args.n, rng, MAX_SEQ_LEN)
    preds = [net.predict(s) for s in seqs]

    out_dir = HLS_DIR / "build"
    out_dir.mkdir(exist_ok=True)
    golden = out_dir / "golden_selftest.txt"
    lines = ["# selftest 골든 벡터 (헤더에서 복원한 FixedRNN 출력)",
             f"# header: {header_path.name}  seed={args.seed}",
             "# 형식  : <len> <ids...> <pred>"]
    for s, p in zip(seqs, preds):
        lines.append(f"{len(s)} " + " ".join(str(i) for i in s) + f" {p}")
    golden.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK]    골든 벡터 {len(seqs)}건 생성 "
          f"(길이 {min(len(s) for s in seqs)}..{max(len(s) for s in seqs)})")

    # UNROLL_H를 바꿔가며 컴파일 + 실행
    unrolls = [int(u) for u in args.unroll.split(",")]
    failed = []
    for u in unrolls:
        if 128 % u != 0:
            print(f"\n[SKIP]  UNROLL_H={u} -- HIDDEN_SIZE(128)의 약수가 아니다")
            continue
        print(f"\n=== UNROLL_H = UNROLL_O = {u} ===")
        exe = compile_kernel(header_path, u, out_dir)
        if exe is None:
            failed.append(u)
            continue
        proc = subprocess.run([str(exe), str(golden)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
        print(proc.stdout.strip())
        if proc.stderr.strip():
            print(proc.stderr.strip()[:1000])
        if proc.returncode != 0:
            failed.append(u)

    if not args.keep:
        for f in out_dir.glob("csim_u*.exe"):
            f.unlink(missing_ok=True)

    print()
    if failed:
        print(f"[FAIL] UNROLL_H {failed} 에서 실패")
        return 1
    print(f"[PASS] UNROLL_H {unrolls} 전부 FixedRNN과 비트 일치")
    print("       -> 병렬화 정도가 연산 결과를 바꾸지 않는다 (면적/지연시간만 변한다)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
