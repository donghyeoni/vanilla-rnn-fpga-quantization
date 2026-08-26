/*
 * rnn_kernel_tb.cpp -- 커널 테스트벤치 (C simulation / C-RTL co-simulation 공용)
 *
 * hls/gen_golden.py가 src/fixedpoint.py의 FixedRNN으로 뽑아 둔 골든 벡터를 읽어
 * 커널 출력과 한 글자씩 비교한다. 통과 조건은 "정확도가 비슷함"이 아니라
 * **전건 완전 일치**다. 같은 정수 연산을 같은 순서로 했다면 한 개도 틀릴 수 없다.
 *
 * main의 반환값이 0이 아니면 HLS가 시뮬레이션 실패로 처리한다. csim에서 통과한
 * 뒤 cosim에서도 통과하면, 합성된 RTL이 파이썬 참조 모델과 같다는 뜻이 된다.
 *
 * 골든 벡터 형식 (한 줄에 한 단어):
 *     <seq_len> <id_0> ... <id_{n-1}> <expected_pred>
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rnn_kernel.h"

#define GOLDEN_DEFAULT "golden.txt"
#define MAX_LINE       512

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : GOLDEN_DEFAULT;

    FILE *fp = fopen(path, "r");
    if (!fp) {
        std::printf("[TB] cannot open golden vectors: %s\n"
                    "[TB] run hls/selftest.py or hls/gen_golden.py first, and make\n"
                    "     sure the path is reachable from the csim working dir.\n", path);
        return 1;
    }

    char line[MAX_LINE];
    long total = 0, mismatch = 0;
    long skipped_too_long = 0;

    while (std::fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n') continue;

        char *cur = line;
        const long n = std::strtol(cur, &cur, 10);
        if (n <= 0) continue;

        if (n > MAX_SEQ_LEN) {          /* MAX_SEQ_LEN을 넘는 단어는 커널 범위 밖 */
            skipped_too_long++;
            continue;
        }

        idx_t seq[MAX_SEQ_LEN];
        for (int i = 0; i < MAX_SEQ_LEN; i++) seq[i] = 0;
        for (long i = 0; i < n; i++)
            seq[i] = (idx_t)std::strtol(cur, &cur, 10);

        const long expect = std::strtol(cur, &cur, 10);

        idx_t got = 0;
        rnn_kernel(seq, (int)n, &got);

        total++;
        if ((long)got != expect) {
            mismatch++;
            if (mismatch <= 10) {       /* 처음 10건만 자세히 */
                std::printf("[TB] MISMATCH  len=%ld  ids=", n);
                for (long i = 0; i < n; i++)
                    std::printf("%c", (char)('a' + (int)seq[i]));
                std::printf("  expected='%c'  got='%c'\n",
                            (char)('a' + (int)expect), (char)('a' + (int)got));
            }
        }
    }
    std::fclose(fp);

    std::printf("\n[TB] weights header : %s\n", RNN_WEIGHTS_HEADER);
    std::printf("[TB] UNROLL_H / O   : %d / %d\n", UNROLL_H, UNROLL_O);
    std::printf("[TB] compared       : %ld\n", total);
    std::printf("[TB] mismatch       : %ld\n", mismatch);
    if (skipped_too_long)
        std::printf("[TB] skipped (>%d)  : %ld\n", MAX_SEQ_LEN, skipped_too_long);

    if (total == 0) {
        std::printf("[TB] FAIL -- no vectors were compared\n");
        return 1;
    }
    if (mismatch != 0) {
        std::printf("[TB] FAIL -- mismatch against the reference FixedRNN\n");
        return 1;
    }
    std::printf("[TB] PASS -- bit-exact with FixedRNN on all %ld vectors\n", total);
    return 0;
}
