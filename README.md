# vanilla-rnn-fpga-quantization

직접 구현한 **vanilla RNN**을 **Q1.15 고정소수점** int16 가중치로 변환해,
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
3. **Quantize** — 각 float 값을 `round(x * 2^f)` 후 int16 범위 `[-32768, 32767]`로
   클리핑해 **Q(16-f).f signed int16**으로 양자화합니다(기본 `f = 15`, 즉 Q1.15).
   고정소수점 연산을 float 모델과 대조 검증할 수 있도록 정수 전용 참조 데이터패스
   (`src/fixedpoint.py`의 `FixedRNN`)를 제공합니다.
4. **Emit** — FPGA/HLS에서 쓸 `const int16_t` 배열의 C 헤더를 생성합니다.

포맷은 텐서마다 다르게 줄 수 있습니다. `f`가 텐서·신호별로 달라지면 누산기 정렬
시프트도 함께 달라지며, 그 계산은 `FixedRNN`이 담당하고 생성된 헤더에
`SHIFT_*` 상수로 함께 실립니다. 어떤 포맷을 고를지에 대한 실측은
아래 [Q포맷 연구](#q포맷-연구) 절에 있습니다.

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
| float vs Q4.12 예측 일치율 | **1.0000** | **1.0000** ([Q포맷 연구](#q포맷-연구)) |
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
구현(`src/fixedpoint.py`의 `FixedRNN` — int16 가중치, 확장 누산, 산술 시프트, tanh 앞 포화;
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
인해 **약 4.8%p의 정확도 손실**이 생깁니다. 이 손실은 아래 [Q포맷 연구](#q포맷-연구)에서
**Q4.12로 전량 회수**됩니다.

## Q포맷 연구

위 4.8%p 손실을 출발점으로, 포맷 선택이 정수 경로의 충실도에 어떤 영향을 주는지
실측한 결과입니다. 산출물은 [`results/q_format_study/`](results/q_format_study/)에
있으며, 모두 릴리스된 `nextword_weights_original.pth`와 13,881개 테스트 단어
전체로 측정했습니다(float 기준 정확도 0.6750).

Q포맷은 int16 비트를 어떻게 **해석**하는지의 문제일 뿐이므로, 어떤 포맷을 고르든
DSP·BRAM 사용량은 변하지 않습니다. 바뀌는 것은 누산기의 시프트량뿐입니다.

### 1. 균일 포맷 스윕

모든 텐서와 내부 신호에 같은 포맷을 적용해 Q1.15부터 Q8.8까지 훑었습니다
([`sweep.csv`](results/q_format_study/sweep.csv),
[`sweep.log`](results/q_format_study/sweep.log)).

| 포맷 | 범위 | LSB | fixed acc | 손실 | 일치율 | `Wx` 클리핑 | z 포화율 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q1.15 | ±1 | 3.05e-05 | 0.6272 | +4.78%p | 0.7792 | 38.79% | 55.39% |
| Q2.14 | ±2 | 6.10e-05 | 0.6680 | +0.70%p | 0.9446 | 11.27% | 36.58% |
| Q3.13 | ±4 | 1.22e-04 | 0.6752 | −0.02%p | 0.9996 | 0.45% | 12.22% |
| **Q4.12** | **±8** | **2.44e-04** | **0.6750** | **+0.00%p** | **1.0000** | **0.00%** | **0.96%** |
| Q5.11 | ±16 | 4.88e-04 | 0.6750 | −0.01%p | 0.9995 | 0.00% | 0.01% |
| Q6.10 | ±32 | 9.77e-04 | 0.6747 | +0.03%p | 0.9986 | 0.00% | 0.00% |
| Q7.9 | ±64 | 1.95e-03 | 0.6750 | −0.01%p | 0.9979 | 0.00% | 0.00% |
| Q8.8 | ±128 | 3.91e-03 | 0.6743 | +0.06%p | 0.9952 | 0.00% | 0.00% |

**Q4.12에서 일치율이 정확히 1.0000이 되고 4.8%p 손실이 완전히 사라집니다.**
곡선은 양쪽에서 열화합니다 — 아래로는 범위 부족(클리핑), 위로는 분해능 부족입니다.
`Wx`의 absmax가 5.32이므로 Q3.13(±4)으로는 부족하고 Q4.12(±8)가 이 모델에서
필요한 **최소** 정수부 폭입니다.

### 2. 오차 원인 분리

Q1.15 손실에는 두 요인이 겹쳐 있습니다. 하나는 README가 지목한 **가중치 클리핑**이고,
다른 하나는 **tanh 직전 누산값의 포화**입니다 — Q1.15에서는 누산값 `z`가 ±1.0으로
잘리는데 `tanh(1.0) = 0.7616`이라 은닉 유닛이 구조적으로 ±0.76을 넘지 못합니다.
기준선에서 이 포화는 **전체 은닉 활성값의 55.4%**에서 발생합니다.

두 요인을 독립적으로 켜고 끈 2×2 결과입니다
([`ablation.csv`](results/q_format_study/ablation.csv),
[`ablation.log`](results/q_format_study/ablation.log)):

| | 가중치 클리핑 | z 포화 | fixed acc | 손실 | 일치율 |
| --- | --- | --- | --- | --- | --- |
| A. baseline | O | O | 0.6272 | +4.78%p | 0.7792 |
| B. weights-only | X | O | 0.6587 | +1.63%p | 0.8789 |
| C. acts-only | O | X | **0.5865** | **+8.85%p** | 0.7011 |
| D. both | X | X | 0.6749 | +0.01%p | 0.9999 |

**두 요인은 가산적이지 않습니다**(상호작용 항 +5.69%p). 특히 C가 결정적입니다 —
가중치를 잘린 채로 두고 `z` 범위만 넓히면 기준선보다 **오히려 4.07%p 더 나빠집니다**.
±1.0 포화가 왜곡된 가중치의 영향을 일괄 압축해 우연히 완충 역할을 하고 있었고,
그 완충을 걷어내면 잘린 값이 더 크게 전파되기 때문입니다(B에서 z 포화율이 55.4% →
61.8%로 올라가는 것도 같은 메커니즘입니다).

즉 **Q4.12가 통하는 이유는 두 원인을 동시에 제거하기 때문**이며, 어느 한쪽만
손보는 개선안은 효과가 없거나 역효과입니다.

### 3. 텐서별 혼합 포맷

텐서마다 실제 범위가 크게 다릅니다
([`weight_ranges.csv`](results/q_format_study/weight_ranges.csv)):

| 텐서 | shape | absmax | `\|w\| ≥ 1` | 필요한 최소 포맷 |
| --- | --- | --- | --- | --- |
| `Wx` | 26×128 | 5.3176 | 38.79% | **Q4.12** |
| `Wh` | 128×128 | 1.6892 | 0.01% | Q2.14 |
| `b` | 128 | 1.0996 | 1.56% | Q2.14 |
| `Wo` | 128×26 | 1.1238 | 0.09% | Q2.14 |
| `bo` | 26 | 0.4249 | 0.00% | Q1.15 |

넓은 범위가 필요한 것은 `Wx` 하나뿐이므로, 전부 Q4.12로 두면 나머지 텐서에서
정수부 2~3비트를 낭비하게 됩니다. `--format auto`는 텐서별로 잘리지 않는 선에서
가장 정밀한 포맷을 고릅니다. 은닉 상태 `h`는 `tanh` 출력이라 항상 `(-1, 1)`이므로
Q1.15가 최적점이고, 원-핫 입력은 Q2.14에서 `1.0`이 정확히 `16384`로 표현됩니다
(Q1.15에서는 `32768`이 `32767`로 잘려 정확히 1이 아니었습니다).

| 신호 | `Wx` | `Wh` | `b` | `Wo` | `bo` | `x` | `h` | `z` | `y` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 포맷 | Q4.12 | Q2.14 | Q2.14 | Q2.14 | Q1.15 | Q2.14 | Q1.15 | Q4.12 | Q4.12 |

혼합 포맷의 정확도는 **0.6748 / 일치율 0.9999**로, 균일 Q4.12(0.6750 / 1.0000)와
13,881개 중 1개 차이입니다. **정확도 면에서는 균일 Q4.12로 충분하다**는 뜻이며,
혼합 포맷의 값어치는 `Wh`·`b`·`Wo`·`h`에 소수부 2~3비트를 더 남겨 두는 데 있습니다 —
총 워드 폭을 16비트 아래로 줄이는 후속 실험에서 쓰일 여유입니다.

생성된 헤더에는 텐서별 포맷과 누산기 시프트가 함께 들어갑니다
([`rnn_weights_mixed.h`](results/q_format_study/rnn_weights_mixed.h),
[`quantize_mixed.log`](results/q_format_study/quantize_mixed.log)):

```c
#define FRAC_BITS_WX  12
#define FRAC_BITS_WH  14
...
#define SHIFT_X   14      /* Wx(12) + x(14) - z(12) */
#define SHIFT_H   17      /* Wh(14) + h(15) - z(12) */
#define SHIFT_B   2       /* b(14)  - z(12)         */
#define SHIFT_O   17      /* Wo(14) + h(15) - y(12) */
#define SHIFT_BO  3       /* bo(15) - y(12)         */
```

### 재현

```bash
# 균일 포맷 스윕
python -m src.format_sweep --weights weights/nextword_weights_original.pth \
    --test data/test_original.txt --out results/q_format_study

# 오차 원인 분리
python -m src.ablation --weights weights/nextword_weights_original.pth \
    --test data/test_original.txt --out results/q_format_study

# 혼합 포맷 C 헤더 생성
python -m src.quantize --csv weights_csv --format auto \
    --header weights_for_FPGA/rnn_weights_mixed.h
```

### 한계

- 단일 모델·단일 시드로 측정했습니다. 포맷 선택이 학습 결과에 얼마나 민감한지는
  다루지 않았습니다.
- 생성된 헤더의 값과 시프트 상수는 파이썬 모델과 대조 검증했지만, 실제 VC707
  합성·배치는 수행하지 않았습니다. 자원 사용량과 지연시간 수치는 아직 없습니다.
- 학습 중 포맷 범위를 강제하는 QAT는 다루지 않았습니다. `Wx`가 애초에 범위 안에
  들어오게 만들면 더 좁은 포맷도 가능할 수 있습니다.

## tanh 룩업 테이블

FPGA에는 `tanh`를 계산할 수단이 없습니다. 정수 연산(`+`, `*`, `>>`)만으로는
초월함수를 만들 수 없기 때문입니다. 그런데 양자화를 거치고 나면 `tanh`의 입력
`z`가 Q4.12 int16, 즉 **유한한 65,536가지 값**뿐이므로 답을 미리 전부 계산해
표에 넣어둘 수 있습니다. 실행 시에는 표를 한 번 읽으면 끝이고 곱셈기도
덧셈기도 쓰지 않습니다.

이 절의 산출물은 [`results/tanh_lut_study/`](results/tanh_lut_study/)에 있습니다.

### 표 줄이기

전체 표는 65,536 × 2바이트 = 128 KB입니다. 세 가지로 줄입니다.

1. **대칭** — `tanh`는 기함수이므로 `z ≥ 0`만 저장하고 부호는 따로 붙입니다.
   오차 없이 크기가 절반이 됩니다.
2. **자르기** — `|z| ≥ sat_bound`는 표의 마지막 값으로 고정합니다.
   `tanh(4) = 0.99933`이라 4 근처부터는 사실상 상수입니다.
3. **간격 + 선형보간** — `step`칸마다 하나씩만 저장하고 사이는 보간합니다.
   `step`이 2의 거듭제곱이면 나눗셈이 시프트가 되어 하드웨어에서 쌉니다.

보간 산술은 전부 정수로 수행하고 나눗셈 대신 산술 우시프트를 쓰므로, 파이썬
모델의 출력은 FPGA가 내놓을 값과 비트 단위로 같습니다.

### 표 크기 대 예측 정확도

중요한 것은 표 자체의 오차가 아니라 **예측이 실제로 몇 개 바뀌는가**입니다.
`tanh` 오차는 128개 은닉 유닛을 거쳐 매 타임스텝 순환에 누적되기 때문입니다
([`lut_sweep.csv`](results/tanh_lut_study/lut_sweep.csv),
[`lut_sweep.log`](results/tanh_lut_study/lut_sweep.log)):

| 표 설정 | 엔트리 | 크기 | BRAM | 표 오차 | fixed acc | 일치율 |
| --- | --- | --- | --- | --- | --- | --- |
| float tanh (기준) | — | — | — | — | 0.6748 | 0.9999 |
| 전 구간, 보간 없음 | 32,769 | 64 KB | 1.38% | 1 LSB | 0.6749 | 0.9999 |
| `sat=4`, 보간 없음 | 16,385 | 32 KB | 0.69% | 22 LSB | 0.6750 | 0.9999 |
| `sat=6`, `step=256` | 98 | 0.19 KB | 0.004% | 13 LSB | 0.6749 | 0.9999 |
| **`sat=4`, `step=256`** | **66** | **0.13 KB** | **0.003%** | **22 LSB** | **0.6750** | **0.9999** |
| `sat=4`, `step=1024` | 18 | 0.04 KB | 0.001% | 196 LSB | 0.6752 | 0.9981 |
| `sat=4`, `step=4096` | 6 | 0.01 KB | 0.000% | 2679 LSB | 0.6738 | 0.9641 |
| `sat=1`, `step=64` | 66 | 0.13 KB | 0.003% | 7812 LSB | 0.6586 | 0.8788 |

**66 엔트리(0.13 KB)로 float `tanh`와 같은 일치율을 유지합니다** — 전체 표 대비
500배 작습니다. XC7VX485T의 BRAM 총량 대비 0.003%라 사실상 공짜입니다.

곡선에서 읽히는 것은 세 가지입니다.

- **표 크기를 결정하는 것은 자르는 지점이지 샘플 간격이 아닙니다.** `sat=4`에서
  `step`을 1에서 256으로 256배 성기게 해도 오차가 22 LSB로 동일합니다. 오차가
  전부 자르기에서 오기 때문입니다.
- **보간은 사실상 공짜입니다.** 정확도를 지키면서 표를 1/256로 줄여 줍니다.
- **마지막 행이 잘라내기의 위험을 보여줍니다.** 같은 66 엔트리라도 `|z| ≥ 1`에서
  자르면 일치율이 0.8788로 무너집니다. 이는 위 [오차 원인 분리](#2-오차-원인-분리)의
  B 구성(0.8789)과 사실상 같은 값입니다 — `tanh` 출력이 `tanh(1) = 0.7616`에
  묶이는 동일한 고장이며, 두 실험이 독립적으로 같은 숫자에 도달했습니다.

### 산출물

`--tanh-lut`을 주면 헤더에 가중치와 함께 표가 들어갑니다. **이 헤더 하나로
순환 전체를 부동소수점 없이 구현할 수 있습니다**
([`rnn_weights_mixed_lut.h`](results/tanh_lut_study/rnn_weights_mixed_lut.h)):

```c
#define TANH_LUT_ENTRIES   66
#define TANH_LUT_SAT_LIMIT 16384      /* |z| >= 4.0 은 마지막 값으로 고정 */
#define TANH_LUT_STEP_LOG2 8          /* 256 칸마다 저장, 나눗셈 대신 시프트 */

const int16_t TANH_LUT[66] = { 0, 2045, 4075, ... };

int16_t tanh_lookup(int32_t z) {
    int32_t s   = (z < 0) ? -1 : 1;
    int32_t mag = abs(z);
    if (mag > TANH_LUT_SAT_LIMIT) mag = TANH_LUT_SAT_LIMIT;
    int32_t i   = mag >> TANH_LUT_STEP_LOG2;
    int32_t f   = mag & (TANH_LUT_STEP - 1);
    int32_t lo  = TANH_LUT[i], hi = TANH_LUT[i + 1];
    return s * (lo + (((hi - lo) * f) >> TANH_LUT_STEP_LOG2));
}
```

표의 마지막 값은 한 칸 복제되어 있습니다. `|z| = SAT_LIMIT`일 때 `i`가 마지막
인덱스가 되어 `TANH_LUT[i + 1]`이 범위를 벗어나므로, 2바이트를 더 써서 C 쪽에서
경계 검사 없이 읽을 수 있게 한 것입니다.

### 재현

```bash
# 표 하나의 크기와 오차 확인
python -m src.tanh_lut --z-frac-bits 12 --sat-bound 4 --step 256

# 표 크기 대 예측 정확도 스윕
python -m src.lut_sweep --weights weights/nextword_weights_original.pth \
    --test data/test_original.txt --out results/tanh_lut_study

# 가중치 + tanh 표가 함께 담긴 C 헤더 생성
python -m src.quantize --csv weights_csv --format auto --tanh-lut \
    --header weights_for_FPGA/rnn_weights_mixed_lut.h
```

### 한계

- 표는 여전히 파이썬이 `np.tanh`로 만듭니다. FPGA는 그 결과를 읽을 뿐이며,
  이는 의도된 설계입니다(하드웨어에서 근사할 필요가 없어집니다).
- 일치율은 float 모델 대비 예측 일치 비율입니다. C/RTL로 옮긴 뒤 같은 값이
  나오는지는 co-simulation으로 다시 확인해야 합니다.
- 자르는 지점과 간격은 이 모델의 `z` 분포에 맞춘 것입니다. 다른 모델에서는
  `lut_sweep.py`를 다시 돌려야 합니다.

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
│   ├── config.py         # hidden_size, 기본 frac_bits(=15), 경로, 하이퍼파라미터
│   ├── model.py          # VanillaRNN + LastCharDataset, collate_lastchar, load_words_from_txt
│   ├── train.py          # train_epoch, evaluate, CLI 진입점 (로더 구성, 학습, .pth 저장)
│   ├── evaluate.py       # 저장된 .pth를 테스트셋으로 평가 (학습 없음)
│   ├── predict.py        # predict_last_char 추론 / 데모
│   ├── weight_export.py  # .pth 로드, shape 확인, 텐서별 CSV 내보내기
│   ├── fixedpoint.py     # FixedFormat(텐서별 Q포맷) + FixedRNN(정수 데이터패스)
│   ├── quantize.py       # 포맷 선택, CSV -> int16 양자화, dump_c_array -> .h (+ SHIFT_* 상수)
│   ├── validate_quantization.py  # float vs 정수 데이터패스, 예측 일치율
│   ├── experiment.py     # 포맷 실험 공통 로딩/측정/표 출력 유틸
│   ├── format_sweep.py   # 균일 포맷 스윕 (범위 vs 분해능 트레이드오프 곡선)
│   ├── ablation.py       # 가중치 클리핑 x 활성값 포화 2x2 오차 원인 분리
│   ├── tanh_lut.py       # 정수 전용 tanh 룩업 테이블 (대칭/자르기/보간) + C 배열 출력
│   └── lut_sweep.py      # 표 크기 대 예측 정확도 스윕
├── run_all.py            # 학습 가능한 코퍼스 합성 + train->export->quantize->validate->predict 일괄 실행
├── results/
│   ├── synthetic/        # 커밋된 산출물(합성 코퍼스): 로그, metrics.json, 샘플 C 헤더 + CSV
│   ├── real_data/        # 실제 단어 리스트 데이터셋의 커밋된 산출물 (Releases 참고)
│   ├── q_format_study/   # Q포맷 연구: 스윕/ablation CSV, 혼합 포맷 헤더, metrics.json
│   └── tanh_lut_study/   # tanh LUT 연구: 표 크기 스윕 CSV, LUT 포함 헤더, metrics.json
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

# 5. 양자화하고 C 헤더 생성 (--format: 15 | 12 | auto | mixed, 기본 15)
python -m src.quantize --csv weights_csv --header weights_for_FPGA/rnn_weights_q15.h
python -m src.quantize --csv weights_csv --format auto --header weights_for_FPGA/rnn_weights_mixed.h

# 6. 양자화 검증: float vs 정수 전용 데이터패스 (FPGA 불필요)
python -m src.validate_quantization --weights weights/nextword_weights.pth --test data/test.txt
python -m src.validate_quantization --weights weights/nextword_weights.pth --test data/test.txt --format 12

# 7. 포맷 연구 (아래 'Q포맷 연구' 절 참고)
python -m src.format_sweep --weights weights/nextword_weights.pth --test data/test.txt
python -m src.ablation --weights weights/nextword_weights.pth --test data/test.txt

# 8. tanh 룩업 테이블 (아래 'tanh 룩업 테이블' 절 참고)
python -m src.lut_sweep --weights weights/nextword_weights.pth --test data/test.txt
python -m src.quantize --csv weights_csv --format auto --tanh-lut     --header weights_for_FPGA/rnn_weights_mixed_lut.h
```

## 참고

### 생성되는 가중치 산출물 (git-ignored)

다음 디렉터리는 **파이프라인 실행으로 생성되며**, 버전 관리에서 의도적으로 제외됩니다
(`.gitignore` 참고):

- **`weights/`** — 학습된 모델의 PyTorch `.pth` state_dict.
- **`weights_csv/`** — 텐서별 CSV:
  `Wx`(26×128), `Wh`(128×128), `Wo`(128×26), `b`(128), `bo`(26).
- **`weights_for_FPGA/`** — 다섯 파라미터의 `const int16_t` 배열이 담긴 C 헤더.
  기본은 Q1.15의 `rnn_weights_q15.h`이며, `--format auto`로 텐서별 혼합 포맷
  헤더(`rnn_weights_mixed.h`)를 함께 생성할 수 있습니다.

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
헤더에는 포맷별 `FRAC_BITS_*`와 누산기 정렬용 `SHIFT_*` 상수가 함께 들어가므로,
HLS/RTL에서 그 값을 그대로 쓰면 파이썬 참조 모델과 일치합니다.

Q포맷은 int16 비트의 해석일 뿐이라 포맷을 바꿔도 DSP·BRAM 사용량은 동일하고,
달라지는 것은 시프트량뿐입니다. 실데이터 가중치에서는 Q1.15 대신 **Q4.12**를 쓰는
것만으로 float 대비 예측 일치율이 0.7792에서 1.0000이 됩니다
([Q포맷 연구](#q포맷-연구)).

순환은 정수 연산으로 구현됩니다. `src/fixedpoint.py`의 `FixedRNN` 참조 구현이
의도된 하드웨어 데이터패스를 그대로 반영합니다: 행렬-벡터 곱의 확장 누산(DSP
누산기와 같이), 포맷에 맞춘 산술 우측 시프트, 그리고 tanh 비선형 직전의
포화(랩어라운드가 아님).

tanh는 `--tanh-lut`으로 생성한 정수 룩업 테이블로 대체할 수 있습니다. 66 엔트리
(0.13 KB)면 부동소수점 tanh와 같은 예측 일치율이 유지되므로, 그 헤더 하나로
순환 전체를 부동소수점 없이 구현할 수 있습니다 — 아래 [tanh 룩업 테이블](#tanh-룩업-테이블) 절 참고.

`src/validate_quantization.py`가 이 정수 데이터패스를 float 모델과 전체 테스트셋에
대해 대조 실행하므로, VC707에 배포하기 전에 양자화된 가중치와 고정소수점 연산을
소프트웨어에서 검증할 수 있습니다.
