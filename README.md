# vanilla-rnn-fpga-quantization

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)

밑바닥부터 구현한 **vanilla RNN**을 **Q1.15 고정소수점** int16 가중치로 변환해,
FPGA 설계에서 바로 `#include`할 수 있는 C 헤더까지 만드는
**train → export → quantize → C header** 파이프라인입니다. RNN 자체는 간단한
*마지막 글자 완성* 과제로 학습하며, 파이프라인에 실을 실제 모델을 제공하는 역할입니다.

RNN은 `torch.nn.RNN`이 아니라 raw `nn.Parameter` 텐서로 직접 구현했기 때문에,
순환식과 가중치가 완전히 투명하고 하드웨어로 옮기기 쉽습니다.

## 개요

예측 과제는 의도적으로 단순하게 골랐습니다. 이 프로젝트의 초점은 분류기가 아니라
**하드웨어로 가는 고정소수점 변환 경로**(텐서별 CSV → Q1.15 int16 → C 헤더)입니다.

**과제 — 마지막 글자 완성.** 마지막 글자를 제거한 소문자 단어(예: `"hell"`)를 입력하면
모델이 빠진 마지막 글자(`"o"`)를 예측합니다. 어휘는 소문자 `a`–`z` 26자입니다.
**sequence-to-one** 분류 문제로, 접두사 전체를 순회한 뒤 **마지막 타임스텝**의
logits로 예측합니다.

**순환식.**

```
h_t = tanh(x_t @ Wx + h_{t-1} @ Wh + b)
y_t = h_t @ Wo + bo
```

- `x` shape: `(B, T, D)` — B: 단어 수, T: 접두사 길이, D: 어휘 크기(26, one-hot)
- `logits` shape: `(B, T, O)`, O = 어휘 크기(26)

**모델 구성:** input = 26, hidden = 128, output = 26.
**학습:** AdamW, lr = 3e-3, gradient clipping(norm) = 1.0, batch size = 256,
20 epochs, `Wx`/`Wo`는 xavier-uniform, `Wh`는 orthogonal 초기화.

**스택:** PyTorch(raw `nn.Parameter`), NumPy, pandas.

## 파이프라인

```
train  ──▶  export CSV  ──▶  quantize to Q1.15  ──▶  C header (.h)
(.pth)      (per-tensor)     (int16)                (const int16_t arrays)
```

1. **Train** — RNN을 학습하고 PyTorch `state_dict`(`.pth`)로 저장합니다.
2. **Export** — 5개 파라미터(`Wx`, `Wh`, `b`, `Wo`, `bo`)를 텐서별 `.csv`로 내보냅니다.
3. **Quantize** — 각 float 값을 `round(x * 2^15)` 후 int16 범위 `[-32768, 32767]`로
   클리핑해 **Q1.15 signed int16**으로 양자화합니다. 고정소수점 연산을 float 모델과
   대조 검증할 수 있도록 정수 전용 참조 RNN 스텝
   (`fixed_mul` / `fixed_matvec` / `fixed_tanh` / `rnn_step_fixed`)을 제공합니다.
4. **Emit** — FPGA/HLS에서 쓸 `const int16_t` 배열의 C 헤더를 생성합니다.

## 결과

여기서 정확도는 **파이프라인 검증 지표**입니다 — 산출물은 분류기가 아니라 양자화된
하드웨어 가중치입니다. 두 결과 세트가 커밋되어 있습니다.

- [`results/synthetic/`](results/synthetic/) — **합성 코퍼스.** 외부 데이터 없이
  명령 한 번으로 재현됩니다: `python run_all.py`
- [`results/real_data/`](results/real_data/) — **실제 영어 단어 리스트.** 데이터셋과
  학습된 가중치는 [Releases](https://github.com/donghyeoni/vanilla-rnn-fpga-quantization/releases)에 있습니다.

| | 합성 코퍼스 | 실제 단어 리스트 |
| --- | --- | --- |
| train / test 단어 수 | 4,000 / 800 | 263,739 / 13,881 |
| 과제 난이도 | 결정적 규칙 → 1.00 도달 가능 | 모호함 (예: `hel` → `l`? `p`?) |
| 최종 test accuracy | **1.00** | **0.66** (릴리스 가중치: **0.675**) |
| float vs Q1.15 예측 일치율 | **1.0000** | **0.7792** |
| 용도 | 파이프라인 검증 | 실제 모델 |

일치율은 **정수 전용 데이터패스**(FPGA가 구현하는 것과 동일한 연산을 소프트웨어로
시뮬레이션 — 보드 불필요)가 float 모델과 같은 글자를 예측한 테스트 단어의 비율입니다.

### 합성 코퍼스

합성 단어는 `prefix + successor(prefix[-1])`(예: `ab` → 마지막 글자 `c`) 규칙으로
만들어, 마지막 글자가 접두사의 결정적 함수가 되고 vanilla RNN이 학습할 수 있습니다.
4,000 train / 800 test 단어, 15 epochs, hidden 128. RNN은 test accuracy **1.00**에
도달해 successor 규칙을 정확히 학습합니다
([`metrics.json`](results/synthetic/metrics.json), [`train.log`](results/synthetic/train.log)).
추론 데모([`predict.log`](results/synthetic/predict.log))도 이를 확인해 줍니다:
`hell` → `m` (l→m), `kore` → `f` (e→f), `knoc` → `d` (c→d).

학습된 가중치는 텐서별 CSV로 내보낸 뒤([`export.log`](results/synthetic/export.log))
**Q1.15 signed int16**으로 양자화하고 C 헤더로 생성합니다
([`quantize.log`](results/synthetic/quantize.log)). 생성된 하드웨어 산출물의 샘플이
커밋되어 있습니다: [`rnn_weights_q15.h`](results/synthetic/rnn_weights_q15.h),
[`Wx.csv`](results/synthetic/Wx.csv).

**고정소수점 검증.** `src/validate_quantization.py`가 float 모델과 정수 전용 참조
구현(`rnn_step_fixed` — int16 가중치, 확장 누산, 산술 시프트, tanh 앞 포화;
FPGA가 구현하는 것과 동일한 데이터패스)을 전체 테스트셋에 대해 실행해 예측을
비교합니다([`validate.log`](results/synthetic/validate.log)):

| | float | Q1.15 fixed |
| --- | --- | --- |
| test accuracy | 1.0000 | **1.0000** |
| 예측 일치율 | — | **1.0000 (800/800)** |

합성 코퍼스의 가중치는 전부 Q1.15 범위 `[-1, 1)` 안에 들어가므로 양자화가 사실상
무손실이고, 정수 데이터패스가 float 모델을 정확히 재현합니다.

### 실제 단어 리스트

[Releases](https://github.com/donghyeoni/vanilla-rnn-fpga-quantization/releases)의
실제 영어 단어 리스트(`train_original.txt` / `test_original.txt` →
`data/train.txt` / `data/test.txt`)로 같은 파이프라인을 실행한 결과입니다.
20 epochs, seed 0. 산출물은 [`results/real_data/`](results/real_data/) 아래에 있습니다
([`train.log`](results/real_data/train.log), [`metrics.json`](results/real_data/metrics.json)).

최종 test accuracy는 **0.66**이며, 릴리스된 `nextword_weights_original.pth`는 같은
테스트셋에서 **0.6750**을 기록합니다([`eval_original.log`](results/real_data/eval_original.log)).

합성 규칙과 달리 실제 철자는 본질적으로 모호해서(`hel` → `hell`? `help`?) 정확도가
1.0에 한참 못 미치는 지점에서 포화합니다 — 이는 과제 자체의 상한이지 파이프라인
결함이 아닙니다(위의 합성 실행이 파이프라인 자체는 건전함을 보여줍니다). 추론 데모
([`predict.log`](results/real_data/predict.log))는 영어답게 동작합니다:
`appl` → `e`, `tabl` → `e`.

실데이터 가중치의 export/양자화 로그와 생성 산출물:
[`export.log`](results/real_data/export.log),
[`quantize.log`](results/real_data/quantize.log),
[`rnn_weights_q15.h`](results/real_data/rnn_weights_q15.h),
[`Wx.csv`](results/real_data/Wx.csv).

**고정소수점 검증.** 릴리스된 원본 가중치로 13,881개 테스트 단어 전체에 대해 같은
float 대 정수 비교를 실행한 결과입니다([`validate.log`](results/real_data/validate.log)):

| | float | Q1.15 fixed |
| --- | --- | --- |
| test accuracy | 0.6750 | **0.6272** |
| 예측 일치율 | — | **0.7792** |

합성 가중치와 달리 실데이터 가중치는 Q1.15에 전부 들어가지 않습니다.
**`Wx` 값의 38.8%가 `[-1, 1)` 범위를 벗어나** 양자화 시점에 ±1로 클리핑되며, 이로
인해 **약 4.8%p의 정확도 손실**이 생깁니다. 클리핑에도 불구하고 `tanh` 포화 덕분에
순환은 사용 가능한 상태로 유지됩니다. 알려진 개선안은 `Wx`에 더 넓은 정수 포맷
(예: **Q4.12**)을 쓰거나 **텐서별 스케일 팩터**를 두는 것이며, 헤더 포맷과 이 검증
하네스가 그 변경이 반영될 지점입니다.

## 데이터셋

모델은 로컬 `.txt` 파일(학습용 1개, 테스트용 1개)에서 영어 **단어 리스트**를 읽습니다.
각 파일은 소문자로 변환되고, 길이 2 이상의 알파벳 연속 구간만 사용합니다
(`load_words_from_txt`).

이 프로젝트에서는 두 가지 데이터셋을 사용합니다.

- **실제 단어 리스트** (263,739 train / 13,881 test) — 릴리스된 가중치를 학습한
  데이터셋입니다. [Releases](https://github.com/donghyeoni/vanilla-rnn-fpga-quantization/releases)에서
  `train_original.txt` / `test_original.txt`를 받아 `data/train.txt` / `data/test.txt`에
  두면 됩니다(또는 `--train` / `--test`로 경로 지정). 결과 지표는
  [`results/real_data/`](results/real_data/)에 있습니다.
- **합성 코퍼스** — **파이프라인 재현에 다운로드가 필요 없습니다.** `run_all.py`가
  결정적이고 학습 가능한 코퍼스(각 단어가 `prefix + successor(prefix[-1])`,
  예: `ab` → `c`)를 `data/train.txt` / `data/test.txt`로 합성하고 고정 시드로 전체
  플로우를 실행합니다 — `results/synthetic/`의 커밋된 결과가 여기서 나왔습니다.

`data/train.txt` / `data/test.txt` 위치에 둔 다른 영어 평문 코퍼스도 그대로 동작합니다.

## 저장소 구조

```
vanilla-rnn-fpga-quantization/
├── src/
│   ├── config.py         # hidden_size, frac_bits(=15), 경로, 하이퍼파라미터
│   ├── model.py          # VanillaRNN + LastCharDataset, collate_lastchar, load_words_from_txt
│   ├── train.py          # train_epoch, evaluate, CLI 진입점 (로더 구성, 학습, .pth 저장)
│   ├── evaluate.py       # 저장된 .pth를 테스트셋으로 평가 (학습 없음)
│   ├── predict.py        # predict_last_char 추론 / 데모
│   ├── weight_export.py  # .pth 로드, shape 확인, 텐서별 CSV 내보내기
│   ├── quantize.py       # quantize_to_int16 (Q1.15), 고정소수점 참조 RNN, dump_c_array -> .h
│   └── validate_quantization.py  # float vs Q1.15 정수 데이터패스, 예측 일치율
├── run_all.py            # 학습 가능한 코퍼스 합성 + train->export->quantize->validate->predict 일괄 실행
├── results/
│   ├── synthetic/        # 커밋된 산출물(합성 코퍼스): 로그, metrics.json, 샘플 C 헤더 + CSV
│   └── real_data/        # 실제 단어 리스트 데이터셋의 커밋된 산출물 (Releases 참고)
├── requirements.txt
├── .gitignore
└── README.md
```

## 설치

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 사용법

합성 코퍼스로 전체 파이프라인을 처음부터 끝까지 재현 (데이터 준비 불필요):

```bash
python run_all.py        # results/synthetic/ 생성 (로그, metrics.json, C 헤더) — 위 '결과' 절 참고
```

또는 저장소 루트에서 모듈을 개별 실행합니다 (패키지 상대 임포트를 쓰므로 `python -m` 사용):

```bash
# 1. 학습 (기본 저장 위치: weights/nextword_weights.pth)
python -m src.train --train data/train.txt --test data/test.txt --epochs 20 --seed 0

# 2. 저장된 .pth를 테스트셋으로 평가 (예: Releases에서 받은 가중치)
python -m src.evaluate --weights weights/nextword_weights.pth --test data/test.txt

# 3. 추론 데모
python -m src.predict --weights weights/nextword_weights.pth --prefixes hell worl appl

# 4. 가중치를 텐서별 CSV로 내보내기
python -m src.weight_export --weights weights/nextword_weights.pth --out weights_csv

# 5. Q1.15로 양자화하고 C 헤더 생성
python -m src.quantize --csv weights_csv --header weights_for_FPGA/rnn_weights_q15.h

# 6. 양자화 검증: float vs 정수 전용 데이터패스 (FPGA 불필요)
python -m src.validate_quantization --weights weights/nextword_weights.pth --test data/test.txt
```

## 참고

### 생성되는 가중치 산출물 (git-ignored)

다음 디렉터리는 **파이프라인 실행으로 생성되며**, 버전 관리에서 의도적으로 제외됩니다
(`.gitignore` 참고):

- **`weights/`** — 학습된 모델의 PyTorch `.pth` state_dict.
- **`weights_csv/`** — 텐서별 CSV:
  `Wx`(26×128), `Wh`(128×128), `Wo`(128×26), `b`(128), `bo`(26).
- **`weights_for_FPGA/`** — 다섯 파라미터의 Q1.15 `const int16_t` 배열이 담긴
  C 헤더 `rnn_weights_q15.h`.

생성된 `rnn_weights_q15.h`와 `Wx.csv`의 커밋된 **샘플**, 실행 로그, `metrics.json`은
`results/` 아래에 있습니다 — 위 [결과](#결과) 절 참고.

### Releases

학습된 가중치와 실제 단어 리스트 데이터셋은
[Releases](https://github.com/donghyeoni/vanilla-rnn-fpga-quantization/releases)로 배포합니다:

- `nextword_weights_original.pth` — 실제 단어 리스트로 학습한 가중치
  (test accuracy ≈ 0.68, [결과](#결과) 절 참고).
- `nextword_weights_synthetic.pth` — 합성 코퍼스로 학습한 가중치
  (`results/synthetic/`의 산출물과 짝을 이룸).
- `train_original.txt` / `test_original.txt` — 실제 단어 리스트 데이터셋.
- `train_synthetic.txt` / `test_synthetic.txt` — 합성 코퍼스 1회 생성본
  (`run_all.py`로 재생성 가능).

### FPGA 타깃

이 프로젝트의 타깃 FPGA 플랫폼은 Virtex-7 XC7VX485T(XC7VX485T-2FFG1761C) 기반의
Xilinx VC707 Evaluation Kit입니다.

생성된 `rnn_weights_q15.h`는 다섯 파라미터를 Q1.15 포맷(부호 1비트, 소수부 15비트)의
`const int16_t` 배열로 제공합니다. Xilinx Vivado, Vitis HLS 등의 도구로 VC707을
타깃하는 FPGA/HLS 프로젝트에 `#include`해 쓰는 것을 전제로 합니다.

순환은 정수 연산으로 구현됩니다. `src/quantize.py`의 `rnn_step_fixed` 참조 구현이
의도된 하드웨어 데이터패스를 그대로 반영합니다: 행렬-벡터 곱의 확장 누산(DSP
누산기와 같이), FRAC_BITS만큼의 산술 우측 시프트, 그리고 tanh 비선형 직전의
포화(랩어라운드가 아님).

현재 tanh는 부동소수점 소프트웨어 참조 구현이며, FPGA에 올릴 때는 룩업 테이블(LUT)
같은 하드웨어 친화적 근사로 대체하는 것을 전제로 합니다.

`src/validate_quantization.py`가 이 정수 데이터패스를 float 모델과 전체 테스트셋에
대해 대조 실행하므로, VC707에 배포하기 전에 양자화된 가중치와 고정소수점 연산을
소프트웨어에서 검증할 수 있습니다.
