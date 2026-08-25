"""
오차 원인 분리 (ablation: weight clipping vs pre-activation saturation).

Q1.15 정수 경로에서 float 대비 정확도가 떨어지는 원인은 두 가지가 겹쳐 있다.

1. **가중치 클리핑** -- ``Wx``의 38.8%가 [-1, 1)을 벗어나 +-1로 잘린다.
2. **활성값 포화**   -- tanh 직전 누산값 z가 Q1.15 범위에서 +-1.0으로 포화한다.
   ``tanh(1.0) = 0.7616``이므로 은닉 유닛이 구조적으로 +-0.76을 넘지 못한다.

README가 지목한 것은 1번이지만, 2번은 포맷을 바꾸면 함께 사라지기 때문에
"Q4.12로 바꿨더니 좋아졌다"만으로는 어느 쪽이 주범인지 알 수 없다. 이 스크립트는
두 요인을 독립적으로 켜고 꺼서 2x2로 분해한다.

  A. baseline      가중치 Q1.15(클리핑 O) + z Q1.15(포화 O)   <- 기존 구현
  B. weights-only  가중치 텐서별 최적    + z Q1.15(포화 O)
  C. acts-only     가중치 Q1.15(클리핑 O) + z Q4.12(포화 X)
  D. both          가중치 텐서별 최적    + z Q4.12(포화 X)

A~D 모두 원-핫 입력을 Q1.15로 두어(기존 구현과 동일) 입력 표현이 교란 변수가
되지 않게 하고, 마지막에 원-핫을 Q2.14로 정확히 표현한 전체 혼합 포맷(E)을
따로 덧붙인다.

사용 예:
    python -m src.ablation --weights weights/nextword_weights_original.pth \
        --test data/test_original.txt --out results/q_format_study
"""

import argparse
import os

from . import config
from . import experiment as exp
from . import fixedpoint as fp

BASE_BITS = {name: 15 for name in config.PARAM_NAMES}


def build_configs(params):
    """2x2 ablation 구성 + 전체 혼합 포맷 한 줄."""
    auto = fp.FixedFormat.from_weights(params)          # 텐서별 최적 (Wx=Q4.12 등)
    wide_bits = {name: getattr(auto, name) for name in config.PARAM_NAMES}
    z_wide = auto.z                                     # 포화가 사실상 사라지는 폭

    def make(label, weight_bits, z_bits, x_bits, note):
        fmt = fp.FixedFormat(x=x_bits, h=15, z=z_bits, y=z_bits, label=label, **weight_bits)
        return fmt, note

    return [
        make("A. baseline", BASE_BITS, 15, 15,
             "weight clipping ON, pre-activation saturation ON (original pipeline)"),
        make("B. weights-only", wide_bits, 15, 15,
             "weight clipping OFF, pre-activation saturation ON"),
        make("C. acts-only", BASE_BITS, z_wide, 15,
             "weight clipping ON, pre-activation saturation OFF"),
        make("D. both", wide_bits, z_wide, 15,
             "weight clipping OFF, pre-activation saturation OFF"),
        make("E. mixed", wide_bits, z_wide, auto.x,
             "D plus an exactly representable one-hot input (Q2.14)"),
    ]


def run(params, id_seqs, targets, float_preds, sat_samples=1500):
    rows = []
    for fmt, note in build_configs(params):
        metrics = fp.evaluate(params, fmt, id_seqs, targets, float_preds=float_preds)
        net = fp.FixedRNN(params, fmt)
        row = {
            "config": fmt.label,
            "weight_format_Wx": fp.qname(fmt.Wx),
            "z_format": fp.qname(fmt.z),
            "x_format": fp.qname(fmt.x),
            "clip_Wx_pct": round(fp.clip_fraction(params["Wx"], fmt.Wx) * 100, 2),
            "z_sat_pct": round(net.saturation_rate(id_seqs[:sat_samples]) * 100, 3),
            "fixed_acc": round(metrics["fixed_acc"], 4),
            "acc_drop_pp": round(metrics["acc_drop"] * 100, 2),
            "agreement": round(metrics["agreement"], 4),
            "note": note,
        }
        rows.append(row)
        print(f"  {row['config']:<16} Wx={row['weight_format_Wx']:<7} z={row['z_format']:<7} "
              f"acc={row['fixed_acc']:.4f} drop={row['acc_drop_pp']:+6.2f}pp "
              f"agree={row['agreement']:.4f}")
    return rows


def summarize(rows):
    """2x2에서 각 요인의 기여도를 %p로 뽑아낸다.

    값은 "기준선 대비 정확도 손실이 얼마나 줄었는가"이므로 **양수가 개선**이고,
    음수는 그 요인만 고쳤을 때 오히려 나빠졌다는 뜻이다.
    """
    by = {r["config"].split(".")[0]: r for r in rows}
    a, b, c, d = by["A"], by["B"], by["C"], by["D"]
    gain_w = round(a["acc_drop_pp"] - b["acc_drop_pp"], 2)
    gain_z = round(a["acc_drop_pp"] - c["acc_drop_pp"], 2)
    gain_both = round(a["acc_drop_pp"] - d["acc_drop_pp"], 2)
    return {
        "baseline_drop_pp": a["acc_drop_pp"],
        "gain_weights_only_pp": gain_w,
        "gain_acts_only_pp": gain_z,
        "gain_both_pp": gain_both,
        "interaction_pp": round(gain_both - gain_w - gain_z, 2),
        "additive": bool(abs(gain_both - gain_w - gain_z) < 0.5),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Decompose the fixed-point accuracy loss into weight clipping "
                    "and pre-activation saturation.")
    parser.add_argument("--weights", type=str, default=str(config.WEIGHTS_PATH),
                        help="path to the .pth state_dict")
    parser.add_argument("--test", type=str, default=str(config.TEST_PATH),
                        help="path to test.txt")
    parser.add_argument("--out", type=str, default=str(config.QSTUDY_DIR),
                        help="output directory for ablation.csv")
    args = parser.parse_args()

    params = exp.load_params(args.weights)
    id_seqs, targets = exp.load_eval_set(args.test)
    print(f"weights: {args.weights}")
    print(f"test words: {len(id_seqs)}\n")

    print("ablation (weight clipping x pre-activation saturation)")
    float_preds = exp.float_predictions(params, id_seqs)
    rows = run(params, id_seqs, targets, float_preds)

    summary = summarize(rows)
    print("\nrecovery of the Q1.15 accuracy drop (positive = better, negative = worse)")
    print(f"  baseline drop                    {summary['baseline_drop_pp']:+.2f} %p")
    print(f"  B: weights only                  {summary['gain_weights_only_pp']:+.2f} %p")
    print(f"  C: pre-activation range only     {summary['gain_acts_only_pp']:+.2f} %p")
    print(f"  D: both                          {summary['gain_both_pp']:+.2f} %p")
    print(f"  interaction (D - B - C)          {summary['interaction_pp']:+.2f} %p")
    if not summary["additive"]:
        print("\n  The two error sources are NOT additive. Widening the pre-activation")
        print("  range alone (C) lets the still-clipped weights push larger, more")
        print("  distorted values through tanh, so it lands below the baseline.")
        print("  Q4.12 works because it removes both sources at the same time.")

    exp.write_csv(os.path.join(args.out, "ablation.csv"), rows)
    exp.write_json(os.path.join(args.out, "ablation_summary.json"), summary)


if __name__ == "__main__":
    main()
