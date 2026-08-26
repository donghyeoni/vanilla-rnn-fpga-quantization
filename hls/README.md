# HLS 커널

`src/fixedpoint.py`의 `FixedRNN`을 **비트 단위로 재현하는** Vitis HLS 커널.
가중치·포맷 상수·tanh 표는 파이프라인이 생성한 C 헤더에서 가져오고, 커널은
로직만 담는다. 헤더를 바꾸면(예: 균일 Q4.12로) 커널은 손대지 않고 재합성만 한다.

## 구성

| 파일 | 역할 |
| --- | --- |
| [`rnn_kernel.h`](rnn_kernel.h) | 형상·병렬화 파라미터·최상위 함수 선언 |
| [`rnn_types.h`](rnn_types.h) | 정수 타입 폭 정의 (`ap_int` / 표준 정수 양쪽) |
| [`rnn_kernel.cpp`](rnn_kernel.cpp) | 커널 본체 |
| [`rnn_kernel_tb.cpp`](rnn_kernel_tb.cpp) | 테스트벤치 (csim / cosim 공용) |
| [`selftest.py`](selftest.py) | **C 헤더만으로** 비트 일치 검증 (torch·데이터셋·Vitis 불필요) |
| [`gen_golden.py`](gen_golden.py) | 실데이터 테스트셋으로 골든 벡터 생성 |
| [`run_hls.tcl`](run_hls.tcl) | csim / csynth / cosim / sweep 실행 |
| [`collect_reports.py`](collect_reports.py) | sweep 리포트를 면적-지연시간 CSV로 |

## 검증 사슬

```
FixedRNN (파이썬)  ==  C++ 커널  ==  생성된 Verilog
     └── csim ──────────┘   └── cosim ──┘
```

통과 조건은 "정확도가 비슷함"이 아니라 **전건 완전 일치**다. 같은 정수 연산을
같은 순서로 했다면 한 개도 틀릴 수 없다.

### 1. 자체 검증 (라이선스·데이터셋·torch 불필요)

```bash
python hls/selftest.py --header results/tanh_lut_study/rnn_weights_mixed_lut.h
```

C 헤더의 int16 가중치를 역양자화해 `FixedRNN`을 복원하고
(`quantize(dequantize(q, f), f) == q` 이므로 무손실), 골든 벡터를 만든 뒤 커널을
일반 `g++`로 컴파일해 대조한다. `UNROLL_H`를 1/16/128로 바꿔가며 결과가 동일한지도
확인한다 — 병렬화가 연산 의미를 바꾸지 않았다는 검사다.

실행 결과 (400 벡터, 길이 1~32):

```
[OK]    헤더 -> FixedRNN 복원 무손실 (상수 / 시프트 / tanh 표 / 가중치)
[PASS]  UNROLL_H [1, 16, 128] 전부 FixedRNN과 비트 일치
```

### 2. 실데이터 골든 벡터 (torch 필요)

```bash
python hls/gen_golden.py \
    --weights weights/nextword_weights_original.pth \
    --test    data/test_original.txt \
    --header  results/tanh_lut_study/rnn_weights_mixed_lut.h \
    --n 512 --out hls/golden.txt
```

벡터를 뽑기 전에 **헤더와 파이썬 모델이 같은 데이터·같은 포맷인지** 먼저 확인한다
(포맷 상수, 시프트량, tanh 표, 양자화된 가중치 전체). 어긋난 상태로 cosim을 돌리면
원인 불명의 불일치만 남는다.

cosim은 csim보다 수백 배 느리므로 기본값을 512로 잡았다. csim은 전건, cosim은
부분집합이 현실적인 조합이다.

### 3. HLS 실행

```bash
vitis_hls -f hls/run_hls.tcl -tclargs csim              # C simulation
vitis_hls -f hls/run_hls.tcl -tclargs all  16          # csim + csynth + cosim
vitis_hls -f hls/run_hls.tcl -tclargs sweep 0 xc7a200tfbg484-2
python hls/collect_reports.py                           # 면적-지연시간 표
```

## 설계 결정

### 비트 일치를 위해 지킨 것

1. **세 항을 각각 시프트한 뒤 더한다.** 합산 후 한 번에 시프트하면 버려지는 하위
   비트가 달라져 결과가 갈린다 (`FixedRNN.accumulate`와 같은 순서).
2. **포화는 tanh 직전에만.** `y`는 argmax만 취하므로 포화시키지 않고 넓게 둔다.
3. **argmax 비교는 `>`.** 동점이면 낮은 인덱스가 이긴다 (`np.argmax`와 같은 규칙).
   `>=`로 쓰면 동점에서 참조 구현과 갈린다.
4. **은닉 상태 이중 버퍼.** 한 배열을 제자리에서 갱신하면 `j`번째 계산이 이미
   갱신된 `h[k]`를 읽어 참조 구현과 달라진다.

### 폭 산정

| 신호 | 폭 | 근거 |
| --- | --- | --- |
| 가중치 / 은닉 상태 | 16 | 헤더의 int16 그대로 |
| 곱 하나 | 32 | `|w| <= 2^15`, `|h| <= 2^15` |
| 행렬-벡터 누산기 | 40 | 위 곱을 128개 더하면 `|acc| <= 2^37`, 부호 포함 39비트 |
| 시프트 후 합산 | 48 | 포맷이 바뀌어 시프트가 음수(좌시프트)가 되는 경우까지 여유 |

`int64`를 쓰면 합성은 되지만 쓰지 않는 상위 비트에 게이트가 낭비된다. 반대로
누산기를 32비트로 줄이면 **csim은 통과하고 cosim에서 틀린다** — 폭 산정이
cosim이 잡아내는 대표적인 오류다.

### 성능 구조

시간축은 `h_t`가 `h_{t-1}`에 의존하므로 파이프라인할 수 없다. RNN의 구조적
한계이며, 성능은 타임스텝 **내부**의 MAC 병렬화로만 얻는다.

```
은닉 상태 갱신 = HIDDEN_SIZE * HIDDEN_SIZE / UNROLL_H  사이클/타임스텝
```

`UNROLL_H`가 유일한 조절 지점이고, 이 값을 훑은 것이 `sweep` 모드다.
`ARRAY_PARTITION`이 함께 있어야 의미가 있다 — 파티션 없이 unroll만 하면 메모리
포트 1개에서 직렬화되어 아무 효과가 없다.

두 가지 절약도 넣었다:

- **입력층에 곱셈기를 쓰지 않는다.** 입력이 원-핫이라 `Wx^T x`가 행 선택으로
  환원된다. `ONEHOT_Q`가 2의 거듭제곱이면 HLS가 곱셈을 시프트로 접어버린다.
- **출력층은 마지막 타임스텝에서만 계산한다.** 참조 구현은 매 스텝 `y`를 구하지만
  argmax는 마지막 `y`에만 적용하므로 결과가 같고 `128x26` MAC을 `T-1`번 아낀다.

### 인터페이스

제어·데이터 전부 AXI4-Lite 슬레이브 하나로 묶었다. 호스트가 `seq`/`seq_len`을
쓰고 start를 올린 뒤 `pred`를 읽는다. 보드에서 소프트웨어를 짜지 않으려면
**JTAG-to-AXI Master IP**를 붙여 Vivado Hardware Manager에서 Tcl로 직접 레지스터를
읽고 쓰면 된다. 시퀀스가 훨씬 길어지면 `seq`를 `m_axi`나 `hls::stream`으로 바꾼다.

## 라이선스 / 디바이스

- **커널 작성과 C simulation은 라이선스가 필요 없다.** `selftest.py`가 일반 `g++`로
  컴파일한다 (`-DRNN_NO_AP_INT`로 `ap_int` 대신 표준 정수 타입 사용).
- **XC7VX485T는 무료 Vivado WebPACK 대상이 아니다.** VC707 키트의 device-locked
  라이선스가 필요하다.
- 라이선스가 없으면 `part`를 WebPACK 대상 7-series(예: `xc7a200tfbg484-2`)로 바꿔
  리소스 특성을 확인하고 최종 타깃만 VC707로 명시하면 된다. 같은 7-series라
  DSP48E1 / BRAM18 구조가 같다.

## 남은 작업

전체 9단계 로드맵과 각 단계의 툴·상태는 루트 [`README.md`의 「HLS 커널」 절](../README.md#hls-커널)에
있다. 여기서는 중복하지 않는다. 현재 완료된 것은 ① C simulation이고, 다음은
② `csynth`(C++ → Verilog)다.

`csynth`의 LUT/DSP/BRAM은 **추정치**이고 구현(place & route)을 거치면 특히 LUT이
상당히 달라진다. 최종 수치로 쓸 것은 post-route 리포트다.
