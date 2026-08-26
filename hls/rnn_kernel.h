/*
 * rnn_kernel.h -- vanilla RNN 고정소수점 추론 커널 (HLS)
 *
 * src/fixedpoint.py의 FixedRNN을 비트 단위로 재현하는 것이 이 커널의 목표다.
 * 두 구현이 같은 예측을 내는지는 hls/rnn_kernel_tb.cpp가 골든 벡터로 검증한다.
 *
 * 가중치·포맷 상수·tanh 표는 파이프라인이 생성한 C 헤더에서 온다. 커널은 로직만
 * 담고 데이터는 전부 헤더가 담는다. 헤더를 바꾸면 (예: 균일 Q4.12로) 커널은
 * 손대지 않고 재합성만 하면 된다.
 */

#ifndef RNN_KERNEL_H
#define RNN_KERNEL_H

/* 생성된 가중치 헤더. 합성 스크립트에서 -D로 바꿀 수 있다.
 * (헤더에 include guard가 없으므로 번역 단위마다 한 번만 포함해야 한다.) */
#ifndef RNN_WEIGHTS_HEADER
#define RNN_WEIGHTS_HEADER "rnn_weights_mixed_lut.h"
#endif
#include RNN_WEIGHTS_HEADER

/* -------------------------------------------------------------------------
 * 모델 형상 (src/config.py와 일치해야 한다)
 * ------------------------------------------------------------------------- */
#define VOCAB_SIZE   26
#define HIDDEN_SIZE  128
#define MAX_SEQ_LEN  32          /* 접두사 최대 길이. 하드웨어 자원과 무관하며
                                  * 입력 버퍼 크기와 지연시간 상한만 정한다. */

/* -------------------------------------------------------------------------
 * 병렬화 파라미터 -- 면적 대 지연시간 트레이드오프의 유일한 조절 지점
 *
 * 한 사이클에 수행하는 MAC 개수. 은닉 상태 갱신 비용이
 *     HIDDEN_SIZE * HIDDEN_SIZE / UNROLL_H  사이클/타임스텝
 * 이므로 UNROLL_H를 키우면 빨라지고 DSP를 더 쓴다. 1/2/4/8/16/32/64/128 중
 * HIDDEN_SIZE의 약수여야 한다.
 * ------------------------------------------------------------------------- */
#ifndef UNROLL_H
#define UNROLL_H 16
#endif
#ifndef UNROLL_O
#define UNROLL_O 16
#endif

#include "rnn_types.h"

/* -------------------------------------------------------------------------
 * 커널 최상위 함수
 *
 *   seq      : 접두사의 글자 인덱스 (0..25), seq[0]이 첫 글자
 *   seq_len  : 유효 글자 수 (1..MAX_SEQ_LEN)
 *   pred     : 예측한 마지막 글자의 인덱스 (0..25)
 * ------------------------------------------------------------------------- */
void rnn_kernel(const idx_t seq[MAX_SEQ_LEN], int seq_len, idx_t *pred);

#endif /* RNN_KERNEL_H */
