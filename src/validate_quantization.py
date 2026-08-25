"""
float 경로와 정수 경로의 예측을 비교해 양자화 충실도를 검증한다.

포맷은 ``--format``으로 고른다 (기본값은 기존과 같은 균일 Q1.15):

    --format 15      모든 텐서 Q1.15
    --format 12      모든 텐서 Q4.12
    --format auto    텐서별 absmax로 최적 포맷 선택
    --format mixed   auto + 테스트셋으로 z/y 활성값 범위 캘리브레이션

포맷을 여러 개 훑어 트레이드오프 곡선을 보려면 ``src.format_sweep``을,
정확도 손실의 원인을 분해하려면 ``src.ablation``을 쓴다.

사용 예:
    python -m src.validate_quantization --weights weights/nextword_weights.pth \
        --test data/test.txt --format auto
"""

import argparse

from . import config
from . import experiment as exp
from . import fixedpoint as fp
from .quantize import resolve_format


def main():
    parser = argparse.ArgumentParser(
        description="Compare float vs fixed-point predictions on the test set.")
    parser.add_argument("--weights", type=str, default=str(config.WEIGHTS_PATH),
                        help="path to the .pth state_dict to validate")
    parser.add_argument("--test", type=str, default=str(config.TEST_PATH),
                        help="path to test.txt")
    parser.add_argument("--format", type=str, default=str(config.FRAC_BITS),
                        help="fractional bits (e.g. 15, 12), 'auto', or 'mixed'")
    args = parser.parse_args()

    params = exp.load_params(args.weights)
    id_seqs, targets = exp.load_eval_set(args.test)
    print(f"weights: {args.weights}")
    print(f"test words: {len(id_seqs)}")

    fmt = resolve_format(args.format, params, args.test)
    print(f"format: {fmt.label}")
    print(fmt.describe())

    metrics = fp.evaluate(params, fmt, id_seqs, targets)
    net = fp.FixedRNN(params, fmt)

    print(f"\nfloat    test_acc {metrics['float_acc']:.4f}")
    print(f"fixed    test_acc {metrics['fixed_acc']:.4f}  "
          f"({fmt.label} int16 weights, wide accumulate, saturate before tanh)")
    print(f"accuracy drop vs float: {metrics['acc_drop'] * 100:+.2f} %p")
    print(f"float vs fixed prediction agreement: {metrics['agreement']:.4f}")
    print(f"pre-activation saturation rate: "
          f"{net.saturation_rate(id_seqs[:1500]) * 100:.3f}%")


if __name__ == "__main__":
    main()
