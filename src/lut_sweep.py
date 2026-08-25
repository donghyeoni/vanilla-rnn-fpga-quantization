"""
tanh LUT 크기 스윕 (table size vs end-to-end accuracy).

표를 줄이는 두 축을 훑는다.

  * ``sat_bound`` -- ``|z|``를 어디서 자를지. tanh는 4 근처부터 상수에 가까우므로
    그 너머를 버리면 표의 앞부분만 남는다.
  * ``step``      -- 몇 칸마다 저장하고 사이를 보간할지.

중요한 것은 **표 자체의 오차가 아니라 예측이 실제로 몇 개 바뀌는가**이다.
tanh 오차는 128개 은닉 유닛을 거쳐 매 타임스텝 순환에 누적되므로, 표에서
몇 LSB 틀리는 것이 최종 argmax에 어떻게 나타나는지는 돌려봐야 안다.
(``ablation.py``에서 "z 범위만 넓히면 오히려 나빠진다"가 나온 전례가 있다.)

사용 예:
    python -m src.lut_sweep --weights weights/nextword_weights_original.pth \
        --test data/test_original.txt --out results/tanh_lut_study
"""

import argparse
import os

from . import config
from . import experiment as exp
from . import fixedpoint as fp
from .tanh_lut import TanhLUT

BRAM_KB = 37080 / 8      # XC7VX485T의 BRAM 총량(KB)

# (sat_bound, step) -- None은 int16 전 구간.
# z=Q4.12에서 sat_bound=8은 전 구간과 같으므로 넣지 않는다.
SWEEP_CONFIGS = [
    # 축 1: 자르는 지점 (보간 없음)
    (None, 1), (6, 1), (5, 1), (4, 1), (3, 1), (2, 1),
    # 축 2: 보간 간격 (자르는 지점 고정)
    (6, 16), (6, 64), (6, 256), (6, 1024),
    (4, 16), (4, 64), (4, 256), (4, 1024), (4, 4096),
    # 한계 탐색: 두 축을 함께 조인다
    (3, 256), (2, 256), (1, 64),
]


def run(params, fmt, id_seqs, targets, float_preds, configs):
    rows = []

    # 기준선: 표 없이 float tanh (지금까지의 모든 결과가 이것)
    ref = fp.evaluate(params, fmt, id_seqs, targets, float_preds=float_preds)
    rows.append({
        "config": "float tanh (reference)", "sat_bound": "", "step": "",
        "entries": "", "kb": "", "bram_pct": "", "max_error_lsb": "",
        "fixed_acc": round(ref["fixed_acc"], 4),
        "acc_drop_pp": round(ref["acc_drop"] * 100, 2),
        "agreement": round(ref["agreement"], 4),
    })
    print(f"  {'float tanh (reference)':<28}{'':>9}{'':>10}{'':>9}"
          f"acc={ref['fixed_acc']:.4f} agree={ref['agreement']:.4f}")

    seen = set()
    for sat_bound, step in configs:
        lut = TanhLUT(fmt.z, fmt.h, sat_bound=sat_bound, step=step)
        key = (lut.sat_limit, lut.step)
        if key in seen:      # 서로 다른 인자가 같은 표가 되는 경우 (예: sat=8 == 전 구간)
            continue
        seen.add(key)
        metrics = fp.evaluate(params, fmt, id_seqs, targets,
                              float_preds=float_preds, tanh_lut=lut)
        row = {
            "config": lut.label(),
            "sat_bound": "full" if sat_bound is None else sat_bound,
            "step": step,
            "entries": lut.entries,
            "kb": round(lut.nbytes / 1024, 2),
            "bram_pct": round(lut.nbytes / 1024 / BRAM_KB * 100, 3),
            "max_error_lsb": lut.max_error(),
            "fixed_acc": round(metrics["fixed_acc"], 4),
            "acc_drop_pp": round(metrics["acc_drop"] * 100, 2),
            "agreement": round(metrics["agreement"], 4),
        }
        rows.append(row)
        print(f"  {row['config']:<28}{row['entries']:>9,}{row['kb']:>8.2f} KB"
              f"{row['max_error_lsb']:>6d} LSB  "
              f"acc={row['fixed_acc']:.4f} agree={row['agreement']:.4f}")
    return rows


def pick_best(rows):
    """일치율을 지키면서 가장 작은 표를 고른다."""
    ref = rows[0]["agreement"]
    usable = [r for r in rows[1:] if r["agreement"] >= ref]
    return min(usable, key=lambda r: r["entries"]) if usable else None


def main():
    parser = argparse.ArgumentParser(
        description="Sweep tanh LUT size against end-to-end prediction accuracy.")
    parser.add_argument("--weights", type=str, default=str(config.WEIGHTS_PATH),
                        help="path to the .pth state_dict")
    parser.add_argument("--test", type=str, default=str(config.TEST_PATH),
                        help="path to test.txt")
    parser.add_argument("--out", type=str, default=str(config.ROOT_DIR / "results" / "tanh_lut_study"),
                        help="output directory for lut_sweep.csv")
    parser.add_argument("--format", type=str, default="auto",
                        help="fixed-point format: bits, 'auto', or 'mixed'")
    args = parser.parse_args()

    from .quantize import resolve_format
    params = exp.load_params(args.weights)
    id_seqs, targets = exp.load_eval_set(args.test)
    fmt = resolve_format(args.format, params, args.test)

    print(f"weights: {args.weights}")
    print(f"test words: {len(id_seqs)}")
    print(f"format: {fmt.label} (z={fp.qname(fmt.z)}, h={fp.qname(fmt.h)})\n")

    print("tanh LUT sweep")
    float_preds = exp.float_predictions(params, id_seqs)
    rows = run(params, fmt, id_seqs, targets, float_preds, SWEEP_CONFIGS)

    best = pick_best(rows)
    if best:
        print(f"\nsmallest table that keeps the reference agreement: {best['config']}")
        print(f"  {best['entries']:,} entries, {best['kb']:.2f} KB "
              f"({best['bram_pct']:.3f}% of BRAM), agreement {best['agreement']:.4f}")

    exp.write_csv(os.path.join(args.out, "lut_sweep.csv"), rows)


if __name__ == "__main__":
    main()
