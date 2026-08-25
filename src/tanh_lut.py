"""
tanh 룩업 테이블 (hardware-friendly tanh via lookup).

FPGA에는 tanh를 계산할 수단이 없다. 정수 연산(``+``, ``*``, ``>>``)만으로는
초월함수를 만들 수 없기 때문이다. 그런데 양자화를 거치고 나면 tanh의 입력
``z``가 Q(16-f).f int16, 즉 **유한한 65,536가지 값**뿐이므로 답을 미리 전부
계산해 표에 넣어둘 수 있다. 실행 시에는 표를 한 번 읽으면 끝이고, 곱셈기도
덧셈기도 쓰지 않는다.

표 줄이기
---------
전체 표는 65,536 x 2바이트 = 128 KB다. 세 가지로 줄인다.

1. **대칭** -- tanh는 기함수이므로 ``z >= 0``만 저장하고 부호는 따로 붙인다.
   오차 0, 크기 절반.
2. **포화 구간 잘라내기** -- ``|z| >= sat_bound``는 표의 마지막 값으로 고정한다.
   ``tanh(4) = 0.99933``이라 4 근처부터는 사실상 상수다.
3. **간격 + 선형보간** -- ``step``칸마다 하나씩만 저장하고 사이는 보간한다.
   ``step``이 2의 거듭제곱이면 나눗셈이 시프트가 되어 하드웨어에서 싸다.

보간 산술은 전부 정수로 수행하며(하드웨어와 동일), 나눗셈 대신 산술 우시프트를
쓴다. 따라서 이 모듈의 출력은 FPGA가 내놓을 값과 비트 단위로 같다.

사용 예:
    python -m src.tanh_lut --z-frac-bits 12 --sat-bound 4 --step 16
"""

import argparse

import numpy as np

from . import config
from . import fixedpoint as fp


class TanhLUT:
    """정수 전용 tanh 룩업 테이블.

    Parameters
    ----------
    z_frac_bits : int
        입력 ``z``의 소수부 비트 수 (Q4.12이면 12).
    h_frac_bits : int
        출력 ``h``의 소수부 비트 수 (Q1.15이면 15).
    sat_bound : float or None
        이 값 이상의 ``|z|``는 표의 마지막 값으로 고정한다. ``None``이면
        int16이 표현할 수 있는 전 구간을 담는다 (오차 0).
    step : int
        표에 저장하는 간격. 1이면 모든 점을 저장하고 보간하지 않는다.
        2의 거듭제곱이어야 한다 (시프트로 나누기 위해).
    symmetric : bool
        기함수 대칭을 이용해 ``z >= 0``만 저장할지 여부.
    """

    def __init__(self, z_frac_bits=12, h_frac_bits=15,
                 sat_bound=None, step=1, symmetric=True):
        if step < 1 or (step & (step - 1)) != 0:
            raise ValueError(f"step은 2의 거듭제곱이어야 한다: {step}")
        if not symmetric:
            raise NotImplementedError("비대칭 표는 구현하지 않았다 "
                                      "(대칭은 오차가 0이라 포기할 이유가 없다)")

        self.z_frac_bits = z_frac_bits
        self.h_frac_bits = h_frac_bits
        self.step = step
        self.step_log2 = int(step).bit_length() - 1
        self.symmetric = symmetric

        # 표가 담을 |z|의 상한. step의 배수로 내림해 보간 구간이 딱 떨어지게 한다.
        if sat_bound is None:
            limit = -config.INT16_MIN          # 32768: |z|의 최댓값
        else:
            limit = int(round(sat_bound * (2.0 ** z_frac_bits)))
            limit = min(limit, -config.INT16_MIN)
        self.sat_limit = (limit >> self.step_log2) << self.step_log2
        self.sat_bound = self.sat_limit / (2.0 ** z_frac_bits)

        # 표 자체: |z| = 0, step, 2*step, ... , sat_limit 지점의 tanh 값
        knots = np.arange(0, self.sat_limit + 1, step, dtype=np.int64)
        table = fp.quantize(np.tanh(knots / (2.0 ** z_frac_bits)), h_frac_bits)

        # 보간은 table[i]와 table[i+1]을 함께 읽는다. |z| == sat_limit이면
        # i가 마지막 인덱스가 되므로 i+1이 표를 벗어난다. 마지막 값을 한 칸
        # 복제해 두면 C 쪽에서 경계 검사 없이 그대로 읽을 수 있다 (2바이트 비용).
        if step > 1:
            table = np.append(table, table[-1])
        self.table = table

    # -- 조회 -------------------------------------------------------------
    def lookup(self, z_q):
        """Q(16-f).f 정수 ``z`` -> Q(16-g).g 정수 ``h``. 전부 정수 연산."""
        z = np.asarray(z_q, dtype=np.int32)
        sign = np.where(z < 0, -1, 1).astype(np.int32)
        mag = np.minimum(np.abs(z), self.sat_limit)        # 포화 구간 클램프

        idx = mag >> self.step_log2
        if self.step == 1:
            val = self.table[idx].astype(np.int32)
        else:
            frac = mag & (self.step - 1)                   # step이 2의 거듭제곱
            lo = self.table[idx].astype(np.int32)
            hi = self.table[np.minimum(idx + 1, len(self.table) - 1)].astype(np.int32)
            val = lo + (((hi - lo) * frac) >> self.step_log2)   # 나눗셈 대신 시프트

        return fp.saturate16(sign * val)

    # -- 조회 -------------------------------------------------------------
    @property
    def entries(self):
        return len(self.table)

    @property
    def nbytes(self):
        return self.entries * 2

    def exact(self, z_q):
        """같은 입력에 대한 무손실 tanh 값 (오차 측정 기준)."""
        return fp.quantize(np.tanh(fp.dequantize(z_q, self.z_frac_bits)),
                           self.h_frac_bits)

    def max_error(self):
        """int16 z 전 구간에 대한 최대 오차 (h의 LSB 단위)."""
        z = np.arange(config.INT16_MIN, config.INT16_MAX + 1, dtype=np.int32)
        return int(np.abs(self.lookup(z).astype(np.int32)
                          - self.exact(z).astype(np.int32)).max())

    def label(self):
        parts = [f"z={fp.qname(self.z_frac_bits)}"]
        parts.append("full" if self.sat_limit >= -config.INT16_MIN
                     else f"sat={self.sat_bound:g}")
        parts.append("exact" if self.step == 1 else f"step={self.step}")
        return " ".join(parts)

    def describe(self):
        bram_kb = 37080 / 8      # XC7VX485T의 BRAM 총량(KB)
        return (f"{self.label()}: {self.entries:,} entries, "
                f"{self.nbytes / 1024:.1f} KB "
                f"({self.nbytes / 1024 / bram_kb * 100:.2f}% of BRAM), "
                f"max error {self.max_error()} LSB")

    # -- C 헤더 출력 -------------------------------------------------------
    def write_c_array(self, file, name="TANH_LUT"):
        file.write(f"/*\n * tanh lookup table -- {self.label()}\n")
        file.write(f" * {self.entries} entries, {self.nbytes} bytes, "
                   f"max error {self.max_error()} LSB\n")
        file.write(" *\n")
        file.write(" * int16_t tanh_lookup(int32_t z) {\n")
        file.write(f" *     int32_t s   = (z < 0) ? -1 : 1;\n")
        file.write(f" *     int32_t mag = abs(z);\n")
        file.write(f" *     if (mag > {name}_SAT_LIMIT) mag = {name}_SAT_LIMIT;\n")
        file.write(f" *     int32_t i   = mag >> {name}_STEP_LOG2;\n")
        if self.step == 1:
            file.write(f" *     return s * {name}[i];\n")
        else:
            file.write(f" *     int32_t f  = mag & ({name}_STEP - 1);\n")
            file.write(f" *     int32_t lo = {name}[i], hi = {name}[i + 1];\n")
            file.write(f" *     return s * (lo + (((hi - lo) * f) >> {name}_STEP_LOG2));\n")
        file.write(" * }\n */\n\n")

        file.write(f"#define {name}_ENTRIES   {self.entries}\n")
        file.write(f"#define {name}_SAT_LIMIT {self.sat_limit}\n")
        file.write(f"#define {name}_STEP      {self.step}\n")
        file.write(f"#define {name}_STEP_LOG2 {self.step_log2}\n")
        file.write(f"#define {name}_Z_FRAC    {self.z_frac_bits}\n")
        file.write(f"#define {name}_H_FRAC    {self.h_frac_bits}\n\n")

        file.write(f"const int16_t {name}[{self.entries}] = {{\n")
        for i in range(0, self.entries, 16):
            row = ", ".join(str(int(v)) for v in self.table[i:i + 16])
            file.write(f"    {row},\n")
        file.write("};\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build a tanh lookup table and report its size and error.")
    parser.add_argument("--z-frac-bits", type=int, default=12,
                        help="fractional bits of the pre-activation z")
    parser.add_argument("--h-frac-bits", type=int, default=15,
                        help="fractional bits of the hidden state h")
    parser.add_argument("--sat-bound", type=float, default=None,
                        help="clamp |z| beyond this value (default: full range)")
    parser.add_argument("--step", type=int, default=1,
                        help="store every Nth point and interpolate (power of two)")
    args = parser.parse_args()

    lut = TanhLUT(args.z_frac_bits, args.h_frac_bits, args.sat_bound, args.step)
    print(lut.describe())


if __name__ == "__main__":
    main()
