"""
weight file parsing / export.

학습된 .pth state_dict를 로드하여:
  1. 각 텐서의 shape 확인
  2. 일부 값 미리보기
  3. numpy 배열로 변환 (for FPGA)
  4. 각 텐서를 per-tensor CSV로 저장 (외부 추출용)

사용 예:
    python -m src.weight_export --weights weights/nextword_weights.pth --out weights_csv
"""

import argparse
import os

import numpy as np
import torch

from . import config


def load_state(weight_path):
    # 0. 파일 불러와서 shape 확인
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    for name, tensor in state.items():
        print(f"{name}: shape = {tuple(tensor.shape)}")
    return state


def inspect_values(state):
    # 0. weight(tensor type) value check
    for name, tensor in state.items():
        print(f"=== {name} ({tuple(tensor.shape)}) ===")
        print(tensor[:3, :5] if tensor.dim() == 2 else tensor[:10])  # 일부만 출력
        print()


def to_numpy(state):
    # 1. 넘파이 배열로 변환(for FPGA)
    Wx = state["Wx"]    # 입력 → hidden 가중치
    Wh = state["Wh"]    # hidden → hidden
    b = state["b"]      # hidden bias
    Wo = state["Wo"]    # hidden → output
    bo = state["bo"]    # output bias

    arrays = {
        "Wx": Wx.numpy(),
        "Wh": Wh.numpy(),
        "b": b.numpy(),
        "Wo": Wo.numpy(),
        "bo": bo.numpy(),
    }
    return arrays


def export_csv(state, output_dir):
    # 2. CSV 파일로 뽑아서 저장(외부 추출용)
    os.makedirs(output_dir, exist_ok=True)
    for name, tensor in state.items():
        arr = tensor.numpy()
        save_path = os.path.join(output_dir, f"{name}.csv")
        np.savetxt(save_path, arr, delimiter=",")
        print(f"saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Export .pth weights to per-tensor CSV files.")
    parser.add_argument("--weights", type=str, default=str(config.WEIGHTS_PATH),
                        help="path to trained .pth")
    parser.add_argument("--out", type=str, default=str(config.WEIGHTS_CSV_DIR),
                        help="output directory for CSV files")
    parser.add_argument("--inspect", action="store_true",
                        help="also print a preview of tensor values")
    args = parser.parse_args()

    state = load_state(args.weights)
    if args.inspect:
        inspect_values(state)
    export_csv(state, args.out)


if __name__ == "__main__":
    main()
