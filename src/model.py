"""
Vanilla RNN의 목적: 미완성 단어 완성
(미완성이란, 마지막 한 글자가 생략된 상태를 의미한다. 또한 단어는 소문자 알파벳으로 구성된다.)

모델 정의 + 데이터셋 유틸리티 모듈.
  - VanillaRNN        : raw nn.Parameter 기반 vanilla RNN (nn.RNN 미사용)
  - LastCharDataset   : 마지막 글자 완성 태스크용 Dataset
  - collate_lastchar  : prefix 길이 동기화 + 원-핫 변환 collate 함수
  - load_words_from_txt : 텍스트 파일에서 소문자 단어 로드
"""

import string

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import re

# 고정 문자셋: 소문자 알파벳
LOWER_ALPHA = list(string.ascii_lowercase)  # ['a',...,'z'] → vocab_size=26
CHARSET = LOWER_ALPHA
VOCAB_SIZE = len(CHARSET)


# -----------------------------------------------------------------------------
# 1. 데이터셋 구성 단계
# -----------------------------------------------------------------------------
class LastCharDataset(Dataset):
    def __init__(self, words):
        self.chars = LOWER_ALPHA
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.chars)  # 사용할 총 알파벳 개수 의미함

        self.samples = []
        for w in words:
            if len(w) < 2 or any(ch not in self.stoi for ch in w):  # 조건 위반 단어 거르기
                continue
            x = [self.stoi[ch] for ch in w[:-1]]  # prefix
            y = self.stoi[w[-1]]                  # last char
            self.samples.append((x, y))           # x,y = 입력,정답

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_seq, y = self.samples[idx]
        return torch.tensor(x_seq, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# prefix 길이 동기화
def collate_lastchar(batch, vocab_size):
    xs, ys = zip(*batch)             # xs: list[(T_i,)], ys:(B,)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    T_max = max(lengths).item()
    B = len(xs)

    # 패딩: 0으로 채운 뒤 원-핫 변환 (패딩 위치는 모두 0벡터)
    x_onehot = torch.zeros(B, T_max, vocab_size, dtype=torch.float32)
    for i, x in enumerate(xs):
        if len(x) > 0:
            x_onehot[i, :len(x)] = F.one_hot(x, num_classes=vocab_size).float()
        else:
            # 길이 1 단어의 prefix가 빈문자열일 수 있음 → 길이 0 허용 시
            pass
    y = torch.stack(ys)  # (B,)
    return x_onehot, lengths, y


def load_words_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    # 알파벳만 추출 (공백·특수문자 제거), 2글자 이상만
    words = re.findall(r"[a-z]+", text)
    words = [w for w in words if len(w) >= 2]
    return words


# -----------------------------------------------------------------------------
# 2. AI 모델 설계
# -----------------------------------------------------------------------------
class VanillaRNN(nn.Module):
    """
    1. h_t = tanh(x_t @ Wx + h_{t-1} @ Wh + b)
    2. y_t = h_t @ Wo + bo
    3. 입력 x: (B, T, D) → B:문장 수 / T:문장의 길이 / D:알파벳 개수
    4. 출력 logits: (B, T, O) (O=클래스 수; 여기선 vocab_size), h: 현재 은닉층 상태
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wx = nn.Parameter(torch.empty(input_size, hidden_size))
        self.Wh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.zeros(hidden_size))

        self.Wo = nn.Parameter(torch.empty(hidden_size, output_size))
        self.bo = nn.Parameter(torch.zeros(output_size))

        nn.init.xavier_uniform_(self.Wx)
        nn.init.orthogonal_(self.Wh)
        nn.init.xavier_uniform_(self.Wo)

    def forward(self, x, h0=None):
        B, T, D = x.shape
        H = self.hidden_size
        assert D == self.input_size

        h = x.new_zeros(B, H) if h0 is None else h0
        logits_list = []
        for t in range(T):
            xt = x[:, t, :]                         # (B,D)
            h = torch.tanh(xt @ self.Wx + h @ self.Wh + self.b)
            yt = h @ self.Wo + self.bo              # (B,O)
            logits_list.append(yt.unsqueeze(1))     # (B,1,O)
        logits = torch.cat(logits_list, dim=1)      # (B,T,O)
        return logits, h
