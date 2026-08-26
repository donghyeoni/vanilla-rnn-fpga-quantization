# vanilla-rnn-fpga-quantization

**vanilla RNN**을 **Q1.15 고정소수점** int16 가중치로 변환해,
FPGA 설계에서 바로 `#include`할 수 있는 C 헤더 제작
(**train → export → quantize → C header**). 

RNN 자체는 *마지막 글자 예측* 모델로 raw `nn.Parameter` 텐서로 직접 구현 -> 순환식과 가중치가 완전히 투명 & 하드웨어로 옮기기 쉬움.

## 개요

이 프로젝트의 초점은 분류기가 아니라
**하드웨어로 가는 고정소수점 변환 경로**(텐서별 CSV → Q1.15 int16 → C 헤더)임.

**마지막 글자 완성.** 마지막 글자를 제거한 소문자 단어(예: `"hell"`)를 입력하면
모델이 빠진 마지막 글자(`"o"`)를 예측. 어휘는 소문자 `a`–`z` 26자임.
**sequence-to-one** 분류 문제로, 접두사 전체를 순회한 뒤 **마지막 타임스텝**의
logits로 예측함.

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

1. **Train** — RNN을 학습하고 PyTorch `state_dict`(`.pth`)로 저장.
2. **Export** — 5개 파라미터(`Wx`, `Wh`, `b`, `Wo`, `bo`)를 텐서별 `.csv`로 정리.
3. **Quantize** — 각 float 값을 `round(x * 2^f)` 후 int16 범위 `[-32768, 32767]`로
   클리핑해 **Q(16-f).f signed int16**으로 양자화(기본 `f = 15`, 즉 Q1.15).
   고정소수점 연산을 float 모델과 대조 검증할 수 있도록 정수 전용 참조 데이터패스
   (`src/fixedpoint.py`의 `FixedRNN`)를 제공
4. **Emit** — FPGA/HLS에서 쓸 `const int16_t` 배열의 C 헤더를 생성.

※포맷은 텐서마다 다르게 줄 수 있습니다. `f`가 텐서·신호별로 달라지면 누산기 정렬
시프트도 함께 달라지며, 그 계산은 `FixedRNN`이 담당하고 생성된 헤더에
`SHIFT_*` 상수로 함께 실립니다. 어떤 포맷을 고를지에 대한 실측은
아래 [Q포맷 연구](#q포맷-연구48p-손실-대응) 절에 있습니다.

## 결과

정확도 -> 파이프라인 검증 지표 
산출물은 ->양자화된 하드웨어 가중치
두 결과 세트가 커밋되어 있음.

- [`results/synthetic/`](results/synthetic/) — **합성 코퍼스.** 외부 데이터 없이
  명령 한 번으로 재현: `python run_all.py`
- [`results/real_data/`](results/real_data/) — **실제 영어 단어 리스트.** 데이터셋과
  학습된 가중치는 [Releases](https://github.com/donghyeoni/vanilla-rnn-fpga-quantization/releases)에 있음.

| | 합성 코퍼스 | 실제 단어 리스트 |
| --- | --- | --- |
| train / test 단어 수 | 4,000 / 800 | 263,739 / 13,881 |
| 과제 난이도 | 결정적 규칙 → 1.00 도달 가능 | 모호함 (예: `hel` → `l`? `p`?) |
| 최종 test accuracy | **1.00** | **0.66** (릴리스 가중치: **0.675**) |
| float vs Q1.15 예측 일치율 | **1.0000** | **0.7792** |
| float vs Q4.12 예측 일치율 | **1.0000** | **1.0000** ([Q포맷 연구](#q포맷-연구48p-손실-대응)) |
| 용도 | 파이프라인 검증 | 실제 모델 |

일치율은 **정수 전용 데이터패스**(FPGA가 구현하는 것과 동일한 연산을 소프트웨어로
시뮬레이션 — 보드 불필요)가 float 모델과 같은 글자를 예측한 테스트 단어의 비율임.

### 합성 코퍼스

합성 단어는 `prefix + successor(prefix[-1])`(예: `ab` → 마지막 글자 `c`) 규칙으로
만들어, 마지막 글자가 접두사의 결정적 함수가 되고 vanilla RNN이 학습할 수 있음.
4,000 train / 800 test 단어, 15 epochs, hidden 128. RNN은 test accuracy **1.00**에
도달해 successor 규칙을 정확히 학습함.
([`metrics.json`](results/synthetic/metrics.json), [`train.log`](results/synthetic/train.log)).
추론 데모([`predict.log`](results/synthetic/predict.log))도 이를 확인함:
`hell` → `m` (l→m), `kore` → `f` (e→f), `knoc` → `d` (c→d).

학습된 가중치는 텐서별 CSV로 내보낸 뒤([`export.log`](results/synthetic/export.log))
**Q1.15 signed int16**으로 양자화하고 C 헤더로 생성
([`quantize.log`](results/synthetic/quantize.log)). 생성된 하드웨어 산출물의 샘플도
커밋되어 있음: [`rnn_weights_q15.h`](results/synthetic/rnn_weights_q15.h),
[`Wx.csv`](results/synthetic/Wx.csv).

**고정소수점 검증.** `src/validate_quantization.py`가 float 모델과 정수 전용 참조
구현(`src/fixedpoint.py`의 `FixedRNN` — int16 가중치, 확장 누산, 산술 시프트, tanh 앞 포화;
FPGA가 구현하는 것과 동일한 데이터패스)을 전체 테스트셋에 대해 실행해 예측을
비교([`validate.log`](results/synthetic/validate.log)):

| | float | Q1.15 fixed |
| --- | --- | --- |
| test accuracy | 1.0000 | **1.0000** |
| 예측 일치율 | — | **1.0000 (800/800)** |

합성 코퍼스의 가중치는 전부 Q1.15 범위 `[-1, 1)` 안에 들어가므로 양자화가 사실상
무손실이고, 정수 데이터패스가 float 모델을 재현.

### 실제 단어 리스트

[Releases](https://github.com/donghyeoni/vanilla-rnn-fpga-quantization/releases)의
실제 영어 단어 리스트(`train_original.txt` / `test_original.txt` →
`data/train.txt` / `data/test.txt`)로 같은 파이프라인을 실행한 결과.
20 epochs, seed 0. 산출물은 [`results/real_data/`](results/real_data/)
([`train.log`](results/real_data/train.log), [`metrics.json`](results/real_data/metrics.json)).

최종 test accuracy는 **0.66**, 릴리스된 `nextword_weights_original.pth`는 같은
테스트셋에서 **0.6750**([`eval_original.log`](results/real_data/eval_original.log)).

실데이터 가중치의 export/양자화 로그와 생성 산출물:
[`export.log`](results/real_data/export.log),
[`quantize.log`](results/real_data/quantize.log),
[`rnn_weights_q15.h`](results/real_data/rnn_weights_q15.h),
[`Wx.csv`](results/real_data/Wx.csv).

**고정소수점 검증.** 릴리스된 원본 가중치로 13,881개 테스트 단어 전체에 대해 같은
float 대 정수 비교 결과([`validate.log`](results/real_data/validate.log)):

| | float | Q1.15 fixed |
| --- | --- | --- |
| test accuracy | 0.6750 | **0.6272** |
| 예측 일치율 | — | **0.7792** |

합성 가중치와 달리 실데이터 가중치는 Q1.15에 전부 들어가지 않음.
**`Wx` 값의 38.8%가 `[-1, 1)` 범위를 벗어나** 양자화 시점에 ±1로 클리핑되며, 이로
인해 **약 4.8%p의 정확도 손실**발생. 이 손실은 아래 [Q포맷 연구](#q포맷-연구48p-손실-대응)에서
**Q4.12로 전량 회수**.

## Q포맷 연구(4.8%p 손실 대응)

포맷 선택의 영향 비교. 산출물은 [`results/q_format_study/`](results/q_format_study/)에
있으며, 모두 릴리스된 `nextword_weights_original.pth`와 13,881개 테스트 단어
전체로 측정(float 기준 정확도 0.6750).

※Q포맷은 int16 비트를 어떻게 **해석**하는지의 문제일 뿐이므로, 어떤 포맷을 고르든
DSP·BRAM 사용량은 변하지 않습니다. 바뀌는 것은 누산기의 시프트량뿐입니다.

### 1. 균일 포맷 스윕

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

※산출물: [`sweep.csv`](results/q_format_study/sweep.csv), [`sweep.log`](results/q_format_study/sweep.log)

**Q4.12에서 일치율이 정확히 1.0000이 됨.**
곡선은 양쪽에서 열화 — 아래로는 범위 부족(클리핑), 위로는 분해능 부족.
`Wx`의 absmax가 5.32이므로 Q3.13(±4)으로는 부족하고 Q4.12(±8)가 이 모델에서
필요한 **최소** 정수부 폭.

### 2. 오차 원인 분리

Q1.15 손실에는 두 요인이 겹쳐 있음. 하나는 **가중치 클리핑**, 다른 하나는
**tanh 직전 누산값의 포화**임 — Q1.15에서는 누산값 `z`가 ±1.0으로 잘리는데
`tanh(1.0) = 0.7616`이라 은닉 유닛이 구조적으로 ±0.76을 넘지 못함.
기준선에서 이 포화가 **전체 은닉 활성값의 55.4%**에서 발생.

두 요인을 독립적으로 켜고 끈 2×2 결과
([`ablation.csv`](results/q_format_study/ablation.csv),
[`ablation.log`](results/q_format_study/ablation.log)):

| | 가중치 클리핑 | z 포화 | fixed acc | 손실 | 일치율 |
| --- | --- | --- | --- | --- | --- |
| A. baseline | O | O | 0.6272 | +4.78%p | 0.7792 |
| B. weights-only | X | O | 0.6587 | +1.63%p | 0.8789 |
| C. acts-only | O | X | **0.5865** | **+8.85%p** | 0.7011 |
| D. both | X | X | 0.6749 | +0.01%p | 0.9999 |

**두 요인은 가산적이지 않음**(상호작용 항 +5.69%p). 특히 C가 결정적 —
가중치를 잘린 채로 두고 `z` 범위만 넓히면 기준선보다 **오히려 4.07%p 더 나빠짐**.
±1.0 포화가 왜곡된 가중치의 영향을 일괄 압축해 우연히 완충 역할을 하고 있었고,
그 완충을 걷어내면 잘린 값이 더 크게 전파되기 때문임(B에서 z 포화율이
55.4% → 61.8%로 오르는 것도 같은 메커니즘).

즉 **Q4.12가 통하는 이유는 두 원인을 동시에 제거하기 때문**이며, 어느 한쪽만
손보는 개선안은 효과가 없거나 역효과임.

### 3. 텐서별 혼합 포맷

텐서마다 실제 범위가 크게 다름
([`weight_ranges.csv`](results/q_format_study/weight_ranges.csv)):

| 텐서 | shape | absmax | `\|w\| ≥ 1` | 필요한 최소 포맷 |
| --- | --- | --- | --- | --- |
| `Wx` | 26×128 | 5.3176 | 38.79% | **Q4.12** |
| `Wh` | 128×128 | 1.6892 | 0.01% | Q2.14 |
| `b` | 128 | 1.0996 | 1.56% | Q2.14 |
| `Wo` | 128×26 | 1.1238 | 0.09% | Q2.14 |
| `bo` | 26 | 0.4249 | 0.00% | Q1.15 |

넓은 범위가 필요한 것은 `Wx` 하나뿐 → 전부 Q4.12로 두면 나머지 텐서에서
정수부 2~3비트를 낭비함. `--format auto`는 텐서별로 잘리지 않는 선에서 가장
정밀한 포맷을 선택. 은닉 상태 `h`는 `tanh` 출력이라 항상 `(-1, 1)`이므로
Q1.15가 최적점이고, 원-핫 입력은 Q2.14에서 `1.0`이 정확히 `16384`로 표현됨
(Q1.15에서는 `32768`이 `32767`로 잘려 정확히 1이 아니었음).

| 신호 | `Wx` | `Wh` | `b` | `Wo` | `bo` | `x` | `h` | `z` | `y` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 포맷 | Q4.12 | Q2.14 | Q2.14 | Q2.14 | Q1.15 | Q2.14 | Q1.15 | Q4.12 | Q4.12 |

혼합 포맷은 **0.6748 / 일치율 0.9999**로, 균일 Q4.12(0.6750 / 1.0000)와
13,881개 중 1개 차이. **정확도 면에서는 균일 Q4.12로 충분**하다는 뜻이며,
혼합 포맷의 값어치는 `Wh`·`b`·`Wo`·`h`에 소수부 2~3비트를 더 남겨 두는 데 있음 —
총 워드 폭을 16비트 아래로 줄이는 후속 실험에서 쓸 여유임.

생성된 헤더에는 텐서별 포맷과 누산기 시프트가 함께 들어감
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

### 한계

- 단일 모델·단일 시드로 측정. 포맷 선택이 학습 결과에 얼마나 민감한지는 다루지 않음.
- 학습 중 포맷 범위를 강제하는 QAT는 다루지 않음. `Wx`가 애초에 범위 안에
  들어오게 만들면 더 좁은 포맷도 가능할 수 있음.

## tanh 룩업 테이블

FPGA에는 `tanh`를 계산할 수단이 없음. 정수 연산(`+`, `*`, `>>`)만으로는
초월함수를 만들 수 없기 때문임. 그런데 양자화를 거치고 나면 `tanh`의 입력 `z`가
Q4.12 int16, 즉 **유한한 65,536가지 값**뿐이므로 답을 미리 전부 계산해 표에
넣어둘 수 있음. 실행 시에는 표를 한 번 읽으면 끝이고 곱셈기도 덧셈기도 안 씀.

산출물은 [`results/tanh_lut_study/`](results/tanh_lut_study/)에 있음.

### 표 줄이기

전체 표는 65,536 × 2바이트 = 128 KB. 세 가지로 줄임.

1. **대칭** — `tanh`는 기함수이므로 `z ≥ 0`만 저장하고 부호는 따로 붙임.
   오차 없이 크기가 절반.
2. **자르기** — `|z| ≥ sat_bound`는 표의 마지막 값으로 고정.
   `tanh(4) = 0.99933`이라 4 근처부터는 사실상 상수.
3. **간격 + 선형보간** — `step`칸마다 하나씩만 저장하고 사이는 보간.
   `step`이 2의 거듭제곱이면 나눗셈이 시프트가 되어 하드웨어에서 쌈.

※보간 산술은 전부 정수로 수행하고 나눗셈 대신 산술 우시프트를 쓰므로, 파이썬
모델의 출력은 FPGA가 내놓을 값과 비트 단위로 같음.

### 표 크기 대 예측 정확도

중요한 것은 표 자체의 오차가 아니라 **예측이 실제로 몇 개 바뀌는가**임.
`tanh` 오차는 128개 은닉 유닛을 거쳐 매 타임스텝 순환에 누적되기 때문
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

**66 엔트리(0.13 KB)로 float `tanh`와 같은 일치율 유지** — 전체 표 대비 500배 작고
XC7VX485T BRAM 총량 대비 0.003%라 사실상 공짜임.

곡선에서 읽히는 것 세 가지:

- **표 크기를 결정하는 것은 자르는 지점이지 샘플 간격이 아님.** `sat=4`에서 `step`을
  1 → 256으로 256배 성기게 해도 오차가 22 LSB로 동일. 오차가 전부 자르기에서 오기 때문.
- **보간은 사실상 공짜.** 정확도를 지키면서 표를 1/256로 줄여 줌.
- **마지막 행이 잘라내기의 위험을 보여줌.** 같은 66 엔트리라도 `|z| ≥ 1`에서 자르면
  일치율이 0.8788로 무너짐. 이는 위 [오차 원인 분리](#2-오차-원인-분리)의 B 구성(0.8789)과
  사실상 같은 값 — `tanh` 출력이 `tanh(1) = 0.7616`에 묶이는 동일한 고장이며,
  두 실험이 독립적으로 같은 숫자에 도달함.

### 산출물

`--tanh-lut`을 주면 헤더에 가중치와 함께 표가 들어감. **이 헤더 하나로 순환 전체를
부동소수점 없이 구현 가능**
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

※표의 마지막 값은 한 칸 복제되어 있음. `|z| = SAT_LIMIT`일 때 `i`가 마지막
인덱스가 되어 `TANH_LUT[i + 1]`이 범위를 벗어나므로, 2바이트를 더 써서 C 쪽에서
경계 검사 없이 읽게 한 것임.

### 한계

- 표는 파이썬이 `np.tanh`로 만듦. FPGA는 그 결과를 읽을 뿐이며 이는 의도된 설계임
  (하드웨어에서 근사할 필요가 없어짐).
- 일치율은 float 모델 대비 예측 일치 비율임. C/RTL로 옮긴 뒤 같은 값이 나오는지는
  co-simulation으로 다시 확인해야 함.
- 자르는 지점과 간격은 이 모델의 `z` 분포에 맞춘 값임. 다른 모델에서는
  `lut_sweep.py`를 다시 돌려야 함.

## 데이터셋

모델은 로컬 `.txt` 파일(학습용 1개, 테스트용 1개)에서 영어 **단어 리스트**를 읽음.
각 파일은 소문자로 변환되고, 길이 2 이상의 알파벳 연속 구간만 사용
(`load_words_from_txt`). 두 가지를 사용함.

- **실제 단어 리스트** (263,739 train / 13,881 test) — 릴리스된 가중치를 학습한
  데이터셋. [Releases](#releases)에서 받아 `data/train.txt` / `data/test.txt`에
  두거나 `--train` / `--test`로 경로 지정.
- **합성 코퍼스** — **다운로드 불필요.** `run_all.py`가 결정적이고 학습 가능한
  코퍼스(각 단어가 `prefix + successor(prefix[-1])`, 예: `ab` → `c`)를 합성하고
  고정 시드로 전체 플로우를 실행 — `results/synthetic/`의 커밋된 결과가 여기서 나옴.

`data/train.txt` / `data/test.txt` 위치에 둔 다른 영어 평문 코퍼스도 그대로 동작함.

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
python run_all.py        # results/synthetic/ 생성 (로그, metrics.json, C 헤더)
```

또는 저장소 루트에서 모듈을 개별 실행 (패키지 상대 임포트를 쓰므로 `python -m` 사용):

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
```

[Q포맷 연구](#q포맷-연구48p-손실-대응)와 [tanh 룩업 테이블](#tanh-룩업-테이블)의
커밋된 결과를 재현하려면 (릴리스된 원본 가중치 기준):

```bash
W=weights/nextword_weights_original.pth
T=data/test_original.txt

# 균일 포맷 스윕 + 오차 원인 분리
python -m src.format_sweep --weights $W --test $T --out results/q_format_study
python -m src.ablation     --weights $W --test $T --out results/q_format_study

# tanh 표 하나의 크기·오차 확인 + 표 크기 스윕
python -m src.tanh_lut --z-frac-bits 12 --sat-bound 4 --step 256
python -m src.lut_sweep --weights $W --test $T --out results/tanh_lut_study

# 가중치 + tanh 표가 함께 담긴 C 헤더
python -m src.quantize --csv weights_csv --format auto --tanh-lut \
    --header weights_for_FPGA/rnn_weights_mixed_lut.h
```

## 참고

### 생성되는 가중치 산출물 (git-ignored)

다음 디렉터리는 **파이프라인 실행으로 생성**되며 버전 관리에서 의도적으로 제외됨
(`.gitignore` 참고). 커밋된 **샘플**과 실행 로그는 `results/` 아래에 있음.

- **`weights/`** — 학습된 모델의 PyTorch `.pth` state_dict.
- **`weights_csv/`** — 텐서별 CSV: `Wx`(26×128), `Wh`(128×128), `Wo`(128×26),
  `b`(128), `bo`(26).
- **`weights_for_FPGA/`** — `const int16_t` 배열이 담긴 C 헤더. 기본은 Q1.15의
  `rnn_weights_q15.h`이고, `--format auto` / `--tanh-lut`으로 혼합 포맷·LUT 포함
  헤더를 생성.

### Releases

학습된 가중치와 실제 단어 리스트 데이터셋은
[Releases](https://github.com/donghyeoni/vanilla-rnn-fpga-quantization/releases)로 배포:

- `nextword_weights_original.pth` — 실제 단어 리스트로 학습한 가중치
  (test accuracy 0.6750).
- `nextword_weights_synthetic.pth` — 합성 코퍼스로 학습한 가중치
  (`results/synthetic/`의 산출물과 짝을 이룸).
- `train_original.txt` / `test_original.txt` — 실제 단어 리스트 데이터셋.
- `train_synthetic.txt` / `test_synthetic.txt` — 합성 코퍼스 1회 생성본
  (`run_all.py`로 재생성 가능).

### FPGA 타깃

타깃 플랫폼은 Virtex-7 XC7VX485T(XC7VX485T-2FFG1761C) 기반의
Xilinx VC707 Evaluation Kit. 생성된 헤더를 Vivado / Vitis HLS 프로젝트에
`#include`해 쓰는 것을 전제로 함.

`src/fixedpoint.py`의 `FixedRNN`이 의도된 하드웨어 데이터패스를 그대로 반영함 —
행렬-벡터 곱의 확장 누산(DSP 누산기와 같이), 포맷에 맞춘 산술 우측 시프트,
tanh 직전의 포화(랩어라운드가 아님). 헤더의 `FRAC_BITS_*` / `SHIFT_*` / `TANH_LUT`을
그대로 쓰면 HLS·RTL이 이 참조 모델과 일치함.

**남은 작업.** 합성·배치는 아직 수행하지 않았으므로 자원 사용량과 지연시간
수치는 없음. 순서는 ① HLS 커널 작성 → ② C/RTL co-simulation으로 예측 일치 확인
→ ③ `xc7vx485tffg1761-2` 타깃 합성·구현.

※XC7VX485T는 무료 Vivado WebPACK 대상이 아님. VC707 키트에 딸린 device-locked
라이선스가 필요함.
