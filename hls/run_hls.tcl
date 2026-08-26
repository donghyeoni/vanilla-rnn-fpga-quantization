# run_hls.tcl -- Vitis HLS / Vivado HLS 실행 스크립트
#
# 사용법 (저장소 루트에서):
#   vitis_hls -f hls/run_hls.tcl
#   vitis_hls -f hls/run_hls.tcl -tclargs <mode> <unroll> <part> <period>
#
#   mode   : csim | csynth | cosim | all | sweep   (기본 all)
#   unroll : UNROLL_H = UNROLL_O 값, 128의 약수     (기본 16)
#   part   : 대상 디바이스                          (기본 xc7vx485tffg1761-2)
#   period : 클럭 주기 ns                           (기본 10 = 100MHz)
#
# 예)
#   vitis_hls -f hls/run_hls.tcl -tclargs csim
#   vitis_hls -f hls/run_hls.tcl -tclargs sweep 0 xc7a200tfbg484-2
#
# sweep 모드는 UNROLL_H를 1..128로 훑어 solution을 하나씩 만든다. 끝난 뒤
#   python hls/collect_reports.py
# 로 면적/지연시간 표를 CSV로 뽑는다. README의 트레이드오프 표가 그 산출물이다.
#
# 주의: XC7VX485T는 무료 WebPACK 대상이 아니다. 라이선스가 없으면 part를
# WebPACK 대상 7-series(예: xc7a200tfbg484-2)로 바꿔 리소스 특성만 확인하고,
# 최종 타깃은 VC707로 명시하면 된다. 같은 7-series라 DSP48E1/BRAM18 구조가 같다.

# ---------------------------------------------------------------------------
# 인자
# ---------------------------------------------------------------------------
set mode   "all"
set unroll 16
set part   "xc7vx485tffg1761-2"
set period 10

if {[llength $argv] > 0} { set mode   [lindex $argv 0] }
if {[llength $argv] > 1} { set unroll [lindex $argv 1] }
if {[llength $argv] > 2} { set part   [lindex $argv 2] }
if {[llength $argv] > 3} { set period [lindex $argv 3] }

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
set ROOT       [file normalize [file dirname [info script]]/..]
set HLS_DIR    ${ROOT}/hls
set HEADER_DIR ${ROOT}/results/tanh_lut_study
set PROJ       ${HLS_DIR}/hls_proj

# 골든 벡터. gen_golden.py(실데이터) 쪽을 우선 쓰고, 없으면 selftest.py 것을 쓴다.
set GOLDEN ${HLS_DIR}/golden.txt
if {![file exists $GOLDEN]} {
    set GOLDEN ${HLS_DIR}/build/golden_selftest.txt
}
if {![file exists $GOLDEN]} {
    puts "ERROR: 골든 벡터가 없다. 먼저 아래 중 하나를 실행할 것:"
    puts "  python hls/selftest.py  --header results/tanh_lut_study/rnn_weights_mixed_lut.h"
    puts "  python hls/gen_golden.py --header results/tanh_lut_study/rnn_weights_mixed_lut.h ..."
    exit 1
}
set GOLDEN [file normalize $GOLDEN]

# RNN_WEIGHTS_HEADER의 기본값이 rnn_weights_mixed_lut.h이므로 -I 경로만 주면 된다.
# 다른 헤더를 쓰려면 -I 대상 디렉터리를 바꾸고 hls/rnn_kernel.h의 기본값을 수정한다.
proc kernel_cflags {u} {
    global HLS_DIR HEADER_DIR
    return "-I${HLS_DIR} -I${HEADER_DIR} -DUNROLL_H=${u} -DUNROLL_O=${u}"
}

proc setup_solution {sol u part period} {
    global PROJ HLS_DIR
    open_project $PROJ
    set_top rnn_kernel
    add_files ${HLS_DIR}/rnn_kernel.cpp -cflags [kernel_cflags $u]
    add_files -tb ${HLS_DIR}/rnn_kernel_tb.cpp -cflags [kernel_cflags $u]
    open_solution $sol -reset
    set_part $part
    create_clock -period $period -name default
}

# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
if {$mode eq "sweep"} {
    # 면적 대 지연시간 곡선. 각 값이 solution 하나가 된다.
    foreach u {1 2 4 8 16 32 64 128} {
        puts "==== sweep: UNROLL_H = $u ===="
        setup_solution "u${u}" $u $part $period
        csynth_design
        close_project
    }
    puts "sweep 완료. 표를 뽑으려면: python hls/collect_reports.py"
    exit 0
}

setup_solution "u${unroll}" $unroll $part $period

# csim: 커널이 FixedRNN과 같은 답을 내는지 (회로 무관, 초 단위)
if {$mode eq "csim" || $mode eq "all"} {
    csim_design -argv $GOLDEN
}

# csynth: C++ -> Verilog. 리소스/지연시간 리포트가 여기서 나온다.
if {$mode eq "csynth" || $mode eq "cosim" || $mode eq "all"} {
    csynth_design
}

# cosim: 생성된 RTL을 같은 테스트벤치로 검증 (csim보다 수백배 느리다)
if {$mode eq "cosim" || $mode eq "all"} {
    cosim_design -argv $GOLDEN -trace_level none
}

# IP 패키징이 필요하면 주석을 풀 것 (Vivado IP Integrator에 붙일 때)
# export_design -format ip_catalog -rtl verilog

exit 0
