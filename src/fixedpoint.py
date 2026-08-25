"""
일반화된 고정소수점 데이터패스 (generalized fixed-point datapath).

기존 ``quantize.py``는 Q1.15 하나만 가정하고 누산기 시프트를 ``>> FRAC_BITS``로
하드코딩했다. 이 모듈은 **텐서/신호마다 서로 다른 Q포맷**을 허용하도록 그 가정을
풀어낸 것으로, 포맷 스윕·오차 원인 분리·혼합 포맷 실험의 공통 엔진이다.

포맷 표기
---------
``Q(16-f).f`` signed int16  ->  실수값 = 정수값 / 2^f

  * f=15 -> Q1.15, 범위 [-1, 0.99997],  LSB 3.05e-05
  * f=12 -> Q4.12, 범위 [-8, 7.99976],  LSB 2.44e-04

시프트 규칙
-----------
고정소수점 곱의 소수부 비트 수는 두 피연산자의 합이다. 따라서 소수부 a비트와
b비트를 곱해 out비트로 맞추려면 ``(a + b - out)``만큼 산술 시프트한다.
시프트량이 음수면 좌시프트가 된다. 포맷이 전부 같으면 ``a + b - out == f``가
되어 기존 구현의 ``>> FRAC_BITS``와 정확히 일치한다.

하드웨어 대응
-------------
행렬-벡터 곱은 int64로 넓게 누산한 뒤 한 번만 시프트하고(DSP 누산기),
tanh 직전에만 int16으로 포화시킨다. 출력 로짓 y는 argmax만 취하므로
기존 구현과 동일하게 포화시키지 않고 넓은 폭을 유지한다.
"""

from dataclasses import dataclass, asdict

import numpy as np

from . import config

INT16_MIN = config.INT16_MIN
INT16_MAX = config.INT16_MAX
TOTAL_BITS = 16

# 포맷을 갖는 가중치 텐서 / 내부 신호 이름
TENSOR_FIELDS = tuple(config.PARAM_NAMES)          # Wx, Wh, b, Wo, bo
SIGNAL_FIELDS = ("x", "h", "z", "y")               # 입력, 은닉, tanh 이전, 로짓
ALL_FIELDS = TENSOR_FIELDS + SIGNAL_FIELDS


# -----------------------------------------------------------------------------
# 0. Q포맷 유틸리티
# -----------------------------------------------------------------------------
def qname(frac_bits):
    """소수부 비트 수 -> "Q4.12" 형태의 이름."""
    return f"Q{TOTAL_BITS - frac_bits}.{frac_bits}"


def max_pos(frac_bits):
    """해당 포맷이 표현할 수 있는 최댓값."""
    return INT16_MAX / (2.0 ** frac_bits)


def min_neg(frac_bits):
    """해당 포맷이 표현할 수 있는 최솟값."""
    return INT16_MIN / (2.0 ** frac_bits)


def resolution(frac_bits):
    """해당 포맷의 LSB(분해능)."""
    return 1.0 / (2.0 ** frac_bits)


def fit_frac_bits(absmax, max_frac_bits=15):
    """``absmax``를 클리핑 없이 담을 수 있는 **최대** 소수부 비트 수.

    범위는 넓을수록 안전하지만 분해능이 나빠지므로, 잘리지 않는 선에서 가장
    정밀한 포맷을 고르는 것이 텐서별 최적 포맷 선택 규칙이 된다.
    """
    for f in range(max_frac_bits, -1, -1):
        if absmax <= max_pos(f):
            return f
    return 0


def clip_fraction(arr, frac_bits):
    """해당 포맷에서 표현 범위를 벗어나 잘리는 원소의 비율."""
    a = np.asarray(arr, dtype=np.float64)
    return float(((a < min_neg(frac_bits)) | (a > max_pos(frac_bits))).mean())


# -----------------------------------------------------------------------------
# 1. 정수 연산 프리미티브
# -----------------------------------------------------------------------------
def quantize(arr, frac_bits):
    """실수 배열 -> Q(16-f).f int16 (반올림 후 포화)."""
    scaled = np.round(np.asarray(arr, dtype=np.float64) * (2.0 ** frac_bits))
    return np.clip(scaled, INT16_MIN, INT16_MAX).astype(np.int16)


def dequantize(arr_q, frac_bits):
    """Q(16-f).f 정수 배열 -> 실수 배열."""
    return np.asarray(arr_q, dtype=np.float64) / (2.0 ** frac_bits)


def arshift(v, shift):
    """산술 시프트. ``shift > 0``이면 우시프트(하위 비트 버림), 음수면 좌시프트."""
    v = np.asarray(v, dtype=np.int64)
    if shift == 0:
        return v
    if shift > 0:
        return v >> shift          # numpy의 int64 >> 는 산술 시프트(내림)
    return v << (-shift)


def saturate16(v):
    """int16 범위로 포화."""
    return np.clip(v, INT16_MIN, INT16_MAX).astype(np.int16)


def fixed_matvec(W_q, x_q, shift):
    """정수 행렬-벡터 곱. int64로 넓게 누산한 뒤 한 번만 시프트한다."""
    acc = W_q.astype(np.int64) @ x_q.astype(np.int64)
    return arshift(acc, shift)


# -----------------------------------------------------------------------------
# 2. 포맷 명세
# -----------------------------------------------------------------------------
@dataclass
class FixedFormat:
    """가중치 5종 + 내부 신호 4종의 소수부 비트 수 명세."""

    Wx: int = 15
    Wh: int = 15
    b: int = 15
    Wo: int = 15
    bo: int = 15
    x: int = 15     # 입력 원-핫
    h: int = 15     # 은닉 상태 (tanh 출력)
    z: int = 15     # tanh 이전 누산값
    y: int = 15     # 출력 로짓
    label: str = ""

    # -- 생성자 -----------------------------------------------------------
    @classmethod
    def uniform(cls, frac_bits, label=None):
        """모든 텐서/신호에 같은 포맷을 쓰는 균일 포맷 (기존 구현의 일반화)."""
        kwargs = {name: frac_bits for name in ALL_FIELDS}
        kwargs["label"] = label or qname(frac_bits)
        return cls(**kwargs)

    @classmethod
    def from_weights(cls, params, z_bits=None, y_bits=None, h_bits=15, label="auto"):
        """가중치 absmax만으로 텐서별 포맷을 정한다 (활성값 캘리브레이션 없음).

        ``z``/``y``를 지정하지 않으면 ``Wx``와 같은 포맷을 쓴다. 원-핫 입력에
        지배되는 누산값의 범위가 ``Wx``의 범위와 같은 자릿수이기 때문이며,
        데이터셋 없이 CSV만 있을 때의 보수적인 기본값이다.
        """
        bits = {n: fit_frac_bits(float(np.abs(params[n]).max())) for n in TENSOR_FIELDS}
        wx_bits = bits["Wx"]
        return cls(x=fit_frac_bits(1.0), h=h_bits,
                   z=wx_bits if z_bits is None else z_bits,
                   y=wx_bits if y_bits is None else y_bits,
                   label=label, **bits)

    @classmethod
    def calibrate(cls, params, id_seqs, h_bits=15, label="mixed"):
        """가중치는 absmax로, ``z``/``y``는 float 모델의 실제 활성값 범위로 정한다."""
        z_absmax, y_absmax = measure_activation_ranges(params, id_seqs)
        bits = {n: fit_frac_bits(float(np.abs(params[n]).max())) for n in TENSOR_FIELDS}
        return cls(x=fit_frac_bits(1.0), h=h_bits,
                   z=fit_frac_bits(z_absmax), y=fit_frac_bits(y_absmax),
                   label=label, **bits)

    # -- 조회 -------------------------------------------------------------
    def is_uniform(self):
        return len({getattr(self, n) for n in ALL_FIELDS}) == 1

    def to_dict(self):
        return asdict(self)

    def describe(self):
        lines = []
        for name in ALL_FIELDS:
            f = getattr(self, name)
            lines.append(f"  {name:>2s}  {qname(f):<8s} "
                         f"range [{min_neg(f):+9.5f}, {max_pos(f):+9.5f}]  "
                         f"lsb {resolution(f):.3e}")
        return "\n".join(lines)


def measure_activation_ranges(params, id_seqs):
    """float 모델을 돌려 tanh 이전 누산값 z와 출력 로짓 y의 absmax를 잰다."""
    z_absmax = 0.0
    y_absmax = 0.0
    eye = np.eye(config.VOCAB_SIZE, dtype=np.float64)
    for ids in id_seqs:
        h = np.zeros(config.HIDDEN_SIZE, dtype=np.float64)
        for i in ids:
            z = eye[i] @ params["Wx"] + h @ params["Wh"] + params["b"]
            z_absmax = max(z_absmax, float(np.abs(z).max()))
            h = np.tanh(z)
        y = h @ params["Wo"] + params["bo"]
        y_absmax = max(y_absmax, float(np.abs(y).max()))
    return z_absmax, y_absmax


# -----------------------------------------------------------------------------
# 3. 정수 RNN
# -----------------------------------------------------------------------------
class FixedRNN:
    """주어진 포맷으로 양자화한 정수 vanilla RNN.

        z_t = (Wx^T x_t >> s_x) + (Wh^T h_{t-1} >> s_h) + (b >> s_b)  -> 포화 -> int16
        h_t = quantize(tanh(z_t))
        y_t = (Wo^T h_t >> s_o) + (bo >> s_bo)
    """

    def __init__(self, params, fmt):
        self.fmt = fmt
        self.q = {n: quantize(params[n], getattr(fmt, n)) for n in TENSOR_FIELDS}

        # 누산 결과를 목표 포맷으로 맞추는 시프트량 (곱의 소수부 = 두 피연산자의 합)
        self.shift_x = fmt.Wx + fmt.x - fmt.z
        self.shift_h = fmt.Wh + fmt.h - fmt.z
        self.shift_b = fmt.b - fmt.z
        self.shift_o = fmt.Wo + fmt.h - fmt.y
        self.shift_bo = fmt.bo - fmt.y

        # 원-핫 입력을 미리 양자화 (Q2.14에서는 1.0이 정확히 16384로 표현된다)
        self.onehot_q = quantize(np.eye(config.VOCAB_SIZE), fmt.x)

        # 전치 행렬 캐시 (스윕에서 반복 호출되므로)
        self.WxT = np.ascontiguousarray(self.q["Wx"].T)
        self.WhT = np.ascontiguousarray(self.q["Wh"].T)
        self.WoT = np.ascontiguousarray(self.q["Wo"].T)

    def accumulate(self, x_q, h_q):
        """tanh 이전 누산값 z를 포화 전 상태(int64)로 돌려준다."""
        return (fixed_matvec(self.WxT, x_q, self.shift_x)
                + fixed_matvec(self.WhT, h_q, self.shift_h)
                + arshift(self.q["b"], self.shift_b))

    def step(self, x_q, h_q):
        z = saturate16(self.accumulate(x_q, h_q))             # tanh 직전 포화
        h = quantize(np.tanh(dequantize(z, self.fmt.z)), self.fmt.h)
        y = (fixed_matvec(self.WoT, h, self.shift_o)
             + arshift(self.q["bo"], self.shift_bo))
        return h, y

    def predict(self, ids):
        h = np.zeros(config.HIDDEN_SIZE, dtype=np.int16)
        y = None
        for i in ids:
            h, y = self.step(self.onehot_q[i], h)
        return int(np.argmax(y))

    def saturation_rate(self, id_seqs):
        """tanh 이전 누산값이 포화 한계에 걸리는 비율 (오차 원인 분리용)."""
        hit = total = 0
        for ids in id_seqs:
            h = np.zeros(config.HIDDEN_SIZE, dtype=np.int16)
            for i in ids:
                acc = self.accumulate(self.onehot_q[i], h)
                hit += int(((acc <= INT16_MIN) | (acc >= INT16_MAX)).sum())
                total += acc.size
                z = saturate16(acc)
                h = quantize(np.tanh(dequantize(z, self.fmt.z)), self.fmt.h)
        return hit / total if total else 0.0


# -----------------------------------------------------------------------------
# 4. float 기준 모델
# -----------------------------------------------------------------------------
def predict_float(params, ids):
    h = np.zeros(config.HIDDEN_SIZE, dtype=np.float32)
    y = None
    for i in ids:
        x = np.zeros(config.VOCAB_SIZE, dtype=np.float32)
        x[i] = 1.0
        h = np.tanh(x @ params["Wx"] + h @ params["Wh"] + params["b"])
        y = h @ params["Wo"] + params["bo"]
    return int(np.argmax(y))


def evaluate(params, fmt, id_seqs, targets, float_preds=None):
    """float / 고정소수점 정확도와 예측 일치율을 한 번에 계산한다.

    ``float_preds``를 넘기면 float 추론을 다시 돌리지 않는다 (스윕에서 재사용).
    """
    net = FixedRNN(params, fmt)
    n = len(id_seqs)
    agree = float_ok = fixed_ok = 0
    for idx, (ids, target) in enumerate(zip(id_seqs, targets)):
        pf = float_preds[idx] if float_preds is not None else predict_float(params, ids)
        pq = net.predict(ids)
        agree += pf == pq
        float_ok += pf == target
        fixed_ok += pq == target
    return {
        "float_acc": float_ok / n,
        "fixed_acc": fixed_ok / n,
        "agreement": agree / n,
        "acc_drop": (float_ok - fixed_ok) / n,
    }
