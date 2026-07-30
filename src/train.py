"""
3. 학습 루프 (sequence-to-one 분류) + CLI 진입점.

원본 노트북(VanillaRNN.ipynb)의 학습 로직을 모듈화한다.
데이터 경로/에폭 수 등은 하드코딩 대신 CLI 인자 또는 config 기본값을 사용한다.

사용 예:
    python -m src.train --train data/train.txt --test data/test.txt --epochs 20
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config
from .model import (
    VanillaRNN,
    LastCharDataset,
    collate_lastchar,
    load_words_from_txt,
    VOCAB_SIZE,
)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_n = 0.0, 0
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)                                # (B,T,O)
        B, T, O = logits.shape
        idx = (lengths - 1).clamp(min=0)
        last_logits = logits[torch.arange(B, device=device), idx, :]  # (B,O)
        loss = F.cross_entropy(last_logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item() * B
        total_n += B
    return total_loss / max(1, total_n)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_n = 0.0, 0
    total_acc = 0.0
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        logits, _ = model(x)
        B, T, O = logits.shape
        idx = (lengths - 1).clamp(min=0)
        last_logits = logits[torch.arange(B, device=device), idx, :]  # (B,O)
        loss = F.cross_entropy(last_logits, y)
        acc = (last_logits.argmax(dim=-1) == y).float().mean().item()

        total_loss += loss.item() * B
        total_acc += acc * B
        total_n += B
    return total_loss / max(1, total_n), total_acc / max(1, total_n)


def build_loaders(train_path, test_path, batch_size):
    # 1) 데이터 불러오기
    train_words = load_words_from_txt(train_path)
    test_words = load_words_from_txt(test_path)

    print(f"훈련 단어 수: {len(train_words)}")
    print(f"테스트 단어 수: {len(test_words)}")

    # 2) Dataset 생성 (문자셋 고정)
    trainset = LastCharDataset(train_words)
    testset = LastCharDataset(test_words)

    # 3) DataLoader 생성
    collate = lambda b: collate_lastchar(b, vocab_size=VOCAB_SIZE)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return trainset, testset, train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Train the vanilla RNN last-char model.")
    parser.add_argument("--train", type=str, default=str(config.TRAIN_PATH),
                        help="path to train.txt")
    parser.add_argument("--test", type=str, default=str(config.TEST_PATH),
                        help="path to test.txt")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--out", type=str, default=str(config.WEIGHTS_PATH),
                        help="output .pth path for the trained state_dict")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed torch RNG for reproducible training")
    args = parser.parse_args()

    # 1) 모델/옵티마이저 설정
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, train_loader, test_loader = build_loaders(args.train, args.test, args.batch_size)

    model = VanillaRNN(
        input_size=VOCAB_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        output_size=VOCAB_SIZE,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 2) 학습 루프
    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optim, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        print(f"[{ep:02d}] train_loss {tr_loss:.4f} | test_loss {te_loss:.4f} | test_acc {te_acc:.2f}")

    # 3) 가중치 파일 저장
    out_path = args.out
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"모델 가중치 저장 완료! -> {out_path}")


if __name__ == "__main__":
    main()
