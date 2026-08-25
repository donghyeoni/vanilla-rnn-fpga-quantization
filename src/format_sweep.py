"""
균일 Q포맷 스윕 (uniform Q-format sweep).

모든 텐서와 내부 신호에 같은 포맷을 적용하면서 Q1.15부터 Q8.8까지 훑어,
**표현 범위와 분해능의 트레이드오프 곡선**을 실측한다. 이는 "config의
FRAC_BITS를 12로 바꾸면 어떻게 되는가"라는 질문에 대한 직접적인 답이다.

각 포맷마다 다음을 기록한다:

  * ``fixed_acc``      -- 정수 경로의 테스트 정확도
  * ``agreement``      -- float 예측과의 일치율 (양자화 충실도)
  * ``acc_drop``       -- float 대비 정확도 손실 (%p)
  * ``clip_*``         -- 텐서별로 범위를 벗어나 잘리는 가중치 비율
  * ``z_sat``          -- tanh 이전 누산값이 포화 한계에 걸리는 비율

Q포맷은 int16 비트의 해석일 뿐이므로 어떤 포맷을 고르든 DSP/BRAM 사용량은
동일하다. 즉 이 곡선의 최적점을 고르는 것은 하드웨어 비용이 들지 않는다.

사용 예:
    python -m src.format_sweep --weights weights/nextword_weights_original.pth \
        --test data/test_original.txt --out results/q_format_study
"""

import argparse
import os

from . import config
from . import experiment as exp
from . import fixedpoint as fp


def sweep(params, id_seqs, targets, frac_bits_list, float_preds, sat_samples=1500):
    rows = []
    for frac_bits in frac_bits_list:
        fmt = fp.FixedFormat.uniform(frac_bits)
        metrics = fp.evaluate(params, fmt, id_seqs, targets, float_preds=float_preds)
        net = fp.FixedRNN(params, fmt)

        row = {
            "format": fp.qname(frac_bits),
            "frac_bits": frac_bits,
            "range_max": fp.max_pos(frac_bits),
            "lsb": fp.resolution(frac_bits),
            "float_acc": round(metrics["float_acc"], 4),
            "fixed_acc": round(metrics["fixed_acc"], 4),
            "acc_drop_pp": round(metrics["acc_drop"] * 100, 2),
            "agreement": round(metrics["agreement"], 4),
            "z_sat_pct": round(net.saturation_rate(id_seqs[:sat_samples]) * 100, 3),
        }
        for name in config.PARAM_NAMES:
            row[f"clip_{name}_pct"] = round(fp.clip_fraction(params[name], frac_bits) * 100, 2)
        rows.append(row)

        print(f"  {row['format']:<7} acc={row['fixed_acc']:.4f} "
              f"drop={row['acc_drop_pp']:+6.2f}pp  agree={row['agreement']:.4f}  "
              f"clip(Wx)={row['clip_Wx_pct']:5.2f}%  z_sat={row['z_sat_pct']:6.3f}%")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Sweep uniform Q-formats and report the range/resolution trade-off.")
    parser.add_argument("--weights", type=str, default=str(config.WEIGHTS_PATH),
                        help="path to the .pth state_dict")
    parser.add_argument("--test", type=str, default=str(config.TEST_PATH),
                        help="path to test.txt")
    parser.add_argument("--out", type=str, default=str(config.QSTUDY_DIR),
                        help="output directory for sweep.csv")
    parser.add_argument("--frac-bits", type=int, nargs="+", default=config.SWEEP_FRAC_BITS,
                        help="fractional bit widths to sweep")
    args = parser.parse_args()

    params = exp.load_params(args.weights)
    id_seqs, targets = exp.load_eval_set(args.test)
    print(f"weights: {args.weights}")
    print(f"test words: {len(id_seqs)}\n")

    stats = exp.weight_stats(params)
    exp.print_weight_stats(stats)

    print("uniform format sweep")
    float_preds = exp.float_predictions(params, id_seqs)
    rows = sweep(params, id_seqs, targets, args.frac_bits, float_preds)

    best = max(rows, key=lambda r: (r["agreement"], r["fixed_acc"]))
    print(f"\nbest uniform format: {best['format']} "
          f"(acc {best['fixed_acc']:.4f}, agreement {best['agreement']:.4f})")

    exp.write_csv(os.path.join(args.out, "sweep.csv"), rows)
    exp.write_csv(os.path.join(args.out, "weight_ranges.csv"), stats)


if __name__ == "__main__":
    main()
