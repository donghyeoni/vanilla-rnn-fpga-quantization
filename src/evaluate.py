"""
5. 평가 (evaluation).

학습 없이, 저장된 .pth 가중치를 테스트셋에 대해 평가한다
(loss / last-char accuracy). 릴리즈로 배포된 가중치를 검증할 때 사용한다.

사용 예:
    python -m src.evaluate --weights weights/nextword_weights.pth --test data/test.txt
"""

import argparse

import torch

from . import config
from .model import VanillaRNN, VOCAB_SIZE
from .train import build_loaders, evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained state_dict on the test set.")
    parser.add_argument("--weights", type=str, default=str(config.WEIGHTS_PATH),
                        help="path to the .pth state_dict to evaluate")
    parser.add_argument("--train", type=str, default=str(config.TRAIN_PATH),
                        help="path to train.txt (문자셋 구성에만 사용)")
    parser.add_argument("--test", type=str, default=str(config.TEST_PATH),
                        help="path to test.txt")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, _, test_loader = build_loaders(args.train, args.test, args.batch_size)

    model = VanillaRNN(VOCAB_SIZE, config.HIDDEN_SIZE, VOCAB_SIZE).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))

    loss, acc = evaluate(model, test_loader, device)
    print(f"weights: {args.weights}")
    print(f"test_loss {loss:.4f} | test_acc {acc:.4f}")


if __name__ == "__main__":
    main()
