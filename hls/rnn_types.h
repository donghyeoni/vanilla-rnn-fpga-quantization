/*
 * rnn_types.h -- 커널이 쓰는 정수 타입 정의
 *
 * 폭을 직접 정하는 것이 HLS의 핵심이다. int32/int64를 쓰면 합성은 되지만 쓰지
 * 않는 상위 비트에 실제 게이트가 낭비된다.
 *
 * RNN_NO_AP_INT를 정의하면 ap_int 대신 표준 정수 타입을 쓴다. Vitis HLS 설치
 * 없이 일반 g++/MSVC로 C simulation을 돌리기 위한 경로다. 아래 폭 산정에서
 * 오버플로가 불가능함을 보였으므로 두 경로의 연산 결과는 비트 단위로 같다
 * (표준 타입 쪽이 상위 비트를 더 들고 있을 뿐이다).
 */

#ifndef RNN_TYPES_H
#define RNN_TYPES_H

/*
 * 폭 산정
 *   가중치 / 은닉 상태 : 헤더의 int16 그대로            -> 16비트
 *   곱 하나            : |w| <= 2^15, |h| <= 2^15       -> 32비트
 *   행렬-벡터 누산기   : 위 곱을 HIDDEN_SIZE(=2^7)개 더하면 |acc| <= 2^37
 *                        부호 포함 39비트 -> 40비트로 잡으면 오버플로 불가능
 *                        (int64 대비 24비트 절약)
 *   시프트 후 합산     : 포맷이 바뀌어 시프트가 음수(좌시프트)가 되는 경우까지
 *                        여유를 둔 48비트
 */

#ifdef RNN_NO_AP_INT

#include <cstdint>

typedef int16_t   data_t;     /* 가중치 / 은닉 상태 */
typedef uint8_t   idx_t;      /* 글자 인덱스 0..25 */
typedef int32_t   prod_t;     /* int16 x int16 곱 */
typedef uint16_t  lidx_t;     /* tanh 표 인덱스 */
typedef int64_t   acc_t;      /* 행렬-벡터 누산기 (합성 시 40비트) */
typedef int64_t   wide_t;     /* 시프트 후 합산 (합성 시 48비트) */

#else

#include <ap_int.h>

typedef ap_int<16>  data_t;
typedef ap_uint<5>  idx_t;
typedef ap_int<32>  prod_t;
typedef ap_uint<16> lidx_t;
typedef ap_int<40>  acc_t;
typedef ap_int<48>  wide_t;

#endif

#endif /* RNN_TYPES_H */
