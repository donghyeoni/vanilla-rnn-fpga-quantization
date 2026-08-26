/*
 * rnn_kernel.cpp -- vanilla RNN 고정소수점 추론 커널 (Vitis / Vivado HLS)
 *
 * 참조 구현: src/fixedpoint.py의 FixedRNN
 *
 *     z_t = (Wx^T x_t >> SHIFT_X) + (Wh^T h_{t-1} >> SHIFT_H) + (b >> SHIFT_B)
 *           -> int16 포화 -> tanh 표 -> h_t
 *     y   = (Wo^T h_T  >> SHIFT_O) + (bo >> SHIFT_BO)          (마지막 스텝만)
 *     pred = argmax(y)
 *
 * 비트 일치를 위해 반드시 지킨 세 가지:
 *   1. 세 항을 **각각** 시프트한 뒤 더한다. 합산 후 한 번에 시프트하면 버려지는
 *      하위 비트가 달라져 결과가 갈린다 (FixedRNN.accumulate와 동일한 순서).
 *   2. 포화는 tanh 직전에만 한다. y는 argmax만 취하므로 포화시키지 않고 넓게 둔다.
 *   3. 우시프트는 산술 시프트(내림)여야 한다. numpy int64의 >> 와 같은 동작이며
 *      ap_int의 >> 도 부호 있는 타입에서 산술 시프트다.
 */

#include "rnn_kernel.h"

#if UNROLL_O > UNROLL_H
#error "UNROLL_O는 UNROLL_H보다 클 수 없다 (h 배열의 파티션 폭이 UNROLL_H다)"
#endif

/* 원-핫 입력의 양자화 값. quantize(1.0, FRAC_BITS_X) 와 같다.
 * FRAC_BITS_X=14 이면 정확히 16384(=2^14)이고, 15이면 32768이 int16을 넘어
 * 32767로 잘린다 (README의 "Q1.15에서는 1.0이 정확히 표현되지 않는다"가 이것).
 * 2의 거듭제곱이면 아래 곱셈은 HLS가 시프트로 접어버리므로 곱셈기를 쓰지 않는다. */
#define ONEHOT_Q  (((1L << FRAC_BITS_X) > 32767L) ? 32767L : (1L << FRAC_BITS_X))

/* -------------------------------------------------------------------------
 * 정수 프리미티브 (src/fixedpoint.py의 arshift / saturate16과 1:1 대응)
 * ------------------------------------------------------------------------- */

/* 산술 시프트. shift가 음수면 좌시프트 (혼합 포맷에서 발생할 수 있다).
 * shift는 항상 컴파일 타임 상수이므로 분기는 합성 시 사라지고 배선만 남는다. */
static inline wide_t ashr(wide_t v, int shift)
{
#pragma HLS INLINE
    if (shift > 0)  return v >> shift;
    if (shift < 0)  return v << (-shift);
    return v;
}

static inline data_t saturate16(wide_t v)
{
#pragma HLS INLINE
    if (v >  32767) return (data_t) 32767;
    if (v < -32768) return (data_t)-32768;
    return (data_t)v;
}

/* -------------------------------------------------------------------------
 * tanh 표 조회 -- src/tanh_lut.py의 TanhLUT.lookup과 비트 단위로 같다.
 *
 * 곱셈기도 덧셈기도 거의 쓰지 않는다. 부호 분리 -> 클램프 -> 표 2칸 읽기 ->
 * 시프트 보간. TANH_LUT_STEP이 2의 거듭제곱이라 나눗셈이 시프트가 된다.
 * ------------------------------------------------------------------------- */
static data_t tanh_lookup(data_t z)
{
#pragma HLS INLINE
    const bool     neg = (z < 0);
    prod_t     mag = neg ? (prod_t)(-(prod_t)z) : (prod_t)z;
    if (mag > TANH_LUT_SAT_LIMIT) mag = TANH_LUT_SAT_LIMIT;   /* 포화 구간 클램프 */

    const lidx_t i = (lidx_t)(mag >> TANH_LUT_STEP_LOG2);

#if TANH_LUT_STEP > 1
    /* 표 마지막 값이 한 칸 복제되어 있으므로 i+1 경계 검사가 필요 없다
     * (tanh_lut.py가 그 목적으로 2바이트를 더 붙여 둔다). */
    const prod_t frac = mag & (TANH_LUT_STEP - 1);
    const prod_t lo   = (prod_t)TANH_LUT[i];
    const prod_t hi   = (prod_t)TANH_LUT[i + 1];
    const prod_t val  = lo + (((hi - lo) * frac) >> TANH_LUT_STEP_LOG2);
#else
    const prod_t val  = (prod_t)TANH_LUT[i];
#endif

    return saturate16(neg ? (wide_t)(-val) : (wide_t)val);
}

/* -------------------------------------------------------------------------
 * 커널 최상위
 * ------------------------------------------------------------------------- */
void rnn_kernel(const idx_t seq[MAX_SEQ_LEN], int seq_len, idx_t *pred)
{
    /* 제어·데이터 전부 AXI4-Lite 슬레이브 하나로. 호스트(MicroBlaze 또는
     * JTAG-to-AXI)가 seq/seq_len을 쓰고 start를 올린 뒤 pred를 읽는다.
     * 시퀀스가 훨씬 길어지면 seq를 m_axi나 hls::stream으로 바꾸면 된다. */
#pragma HLS INTERFACE s_axilite port=seq     bundle=CTRL
#pragma HLS INTERFACE s_axilite port=seq_len bundle=CTRL
#pragma HLS INTERFACE s_axilite port=pred    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return  bundle=CTRL

    /* 가중치 ROM 분할. UNROLL_H개를 한 사이클에 읽으려면 서로 다른 뱅크에
     * 있어야 한다. 이 pragma가 없으면 unroll을 해도 메모리 포트 1개에서
     * 직렬화되어 아무 효과가 없다 (HLS 초보가 가장 많이 놓치는 지점). */
#pragma HLS ARRAY_PARTITION variable=Wh cyclic factor=UNROLL_H dim=1
#pragma HLS ARRAY_PARTITION variable=Wo cyclic factor=UNROLL_O dim=1
#pragma HLS ARRAY_PARTITION variable=TANH_LUT complete dim=1

    /* 은닉 상태 이중 버퍼. h_nxt를 다 채운 뒤 옮긴다. 한 배열을 제자리에서
     * 갱신하면 j번째 계산이 이미 갱신된 h[k]를 읽어 참조 구현과 달라진다. */
    data_t h_cur[HIDDEN_SIZE];
    data_t h_nxt[HIDDEN_SIZE];
#pragma HLS ARRAY_PARTITION variable=h_cur cyclic factor=UNROLL_H dim=1
#pragma HLS ARRAY_PARTITION variable=h_nxt cyclic factor=UNROLL_H dim=1

    int T = seq_len;
    if (T > MAX_SEQ_LEN) T = MAX_SEQ_LEN;
    if (T < 0)           T = 0;

INIT:
    for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UNROLL_H
        h_cur[j] = 0;
    }

    /* 시간축은 h_t가 h_{t-1}에 의존하므로 파이프라인할 수 없다. RNN의 구조적
     * 한계이며, 성능은 타임스텝 **내부**의 MAC 병렬화(UNROLL_H)로만 얻는다. */
TIME:
    for (int t = 0; t < T; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_SEQ_LEN avg=8

        const idx_t c = seq[t];

    HIDDEN:
        for (int j = 0; j < HIDDEN_SIZE; j++) {

            acc_t acc = 0;
        MAC:
            for (int k = 0; k < HIDDEN_SIZE; k += UNROLL_H) {
#pragma HLS PIPELINE II=1
                acc_t partial = 0;
                for (int u = 0; u < UNROLL_H; u++) {
#pragma HLS UNROLL
                    /* int16 x int16 -> int32. 피연산자를 16비트로 유지해야
                     * DSP48 한 개에 그대로 매핑된다. acc_t끼리 곱하면 40x40
                     * 곱셈기가 되어 DSP를 여러 개 물린다. */
                    const prod_t prod =
                        (data_t)Wh[k + u][j] * h_cur[k + u];
                    partial += prod;
                }
                acc += partial;
            }

            /* 세 항을 각각 정렬한 뒤 합산 (순서가 곧 비트 정확성) */
            const wide_t term_x = ashr((wide_t)Wx[c][j] * ONEHOT_Q, SHIFT_X);
            const wide_t term_h = ashr((wide_t)acc,                 SHIFT_H);
            const wide_t term_b = ashr((wide_t)b[j],                SHIFT_B);

            const data_t z = saturate16(term_x + term_h + term_b);
            h_nxt[j] = tanh_lookup(z);
        }

    SWAP:
        for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UNROLL_H
            h_cur[j] = h_nxt[j];
        }
    }

    /* 출력층은 마지막 타임스텝에서만 계산한다. 참조 구현은 매 스텝 y를 구하지만
     * argmax는 마지막 y에만 적용하므로 결과가 같고, 128x26 MAC을 T-1번 아낀다. */
    wide_t y[VOCAB_SIZE];
#pragma HLS ARRAY_PARTITION variable=y complete dim=1

OUTPUT:
    for (int o = 0; o < VOCAB_SIZE; o++) {
        acc_t acc = 0;
    MAC_O:
        for (int k = 0; k < HIDDEN_SIZE; k += UNROLL_O) {
#pragma HLS PIPELINE II=1
            acc_t partial = 0;
            for (int u = 0; u < UNROLL_O; u++) {
#pragma HLS UNROLL
                const prod_t prod = (data_t)Wo[k + u][o] * h_cur[k + u];
                partial += prod;
            }
            acc += partial;
        }
        y[o] = ashr((wide_t)acc, SHIFT_O) + ashr((wide_t)bo[o], SHIFT_BO);
    }

    /* argmax. 동점이면 낮은 인덱스가 이긴다 (numpy argmax와 같은 규칙이므로
     * 비교는 반드시 > 여야 한다. >= 로 쓰면 동점에서 참조 구현과 갈린다). */
    wide_t best = y[0];
    idx_t  arg  = 0;
ARGMAX:
    for (int o = 1; o < VOCAB_SIZE; o++) {
#pragma HLS UNROLL
        if (y[o] > best) {
            best = y[o];
            arg  = (idx_t)o;
        }
    }

    *pred = arg;
}
