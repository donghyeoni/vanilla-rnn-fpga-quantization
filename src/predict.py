"""
4. 추론 (inference / demo).

학습된 .pth 가중치를 로드해 prefix(마지막 글자가 빠진 단어)로부터
마지막 글자를 예측한다.

사용 예:
    python -m src.predict --weights weights/nextword_weights.pth --prefixes hell worl appl
"""

import argparse
import string

import torch
import torch.nn.functional as F

from . import config
from .model import VanillaRNN, VOCAB_SIZE


@torch.no_grad()
def predict_last_char(model, prefix, stoi, itos, device="cpu"):
    model.eval()
    if len(prefix) == 0:
        return None  # 빈 prefix는 애매
    ids = torch.tensor([stoi[ch] for ch in prefix.lower()], dtype=torch.long)
    x = F.one_hot(ids, num_classes=len(stoi)).float().unsqueeze(0).to(device)  # (1,T,V)
    logits, _ = model(x)                  # (1,T,V)
    last_logits = logits[0, -1, :]        # (V,)
    pred_id = last_logits.argmax().item()
    return itos[pred_id]


def build_vocab():
    """LastCharDataset과 동일한 문자셋으로 stoi/itos를 재구성한다."""
    chars = list(string.ascii_lowercase)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def load_model(weights_path, device="cpu"):
    model = VanillaRNN(
        input_size=VOCAB_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        output_size=VOCAB_SIZE,
    ).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


def main():
    parser = argparse.ArgumentParser(description="Predict the missing last character of a word.")
    parser.add_argument("--weights", type=str, default=str(config.WEIGHTS_PATH),
                        help="path to trained .pth")
    parser.add_argument("--prefixes", type=str, nargs="+",
                        default=["hell", "worl", "appl", "kore", "knoc"],
                        help="one or more prefixes to complete")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)
    stoi, itos = build_vocab()

    for prefix in args.prefixes:
        pred = predict_last_char(model, prefix, stoi, itos, device)
        print(f"prefix='{prefix}' → predicted last char: '{pred}'")


if __name__ == "__main__":
    main()
