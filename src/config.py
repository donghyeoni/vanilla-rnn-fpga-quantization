"""
중앙 설정 모듈 (central configuration).

모델 하이퍼파라미터, 고정소수점 포맷, 파일 경로를 한 곳에서 관리한다.
원본 노트북은 Linux 절대 경로(/home/dh/...)를 하드코딩했으나,
여기서는 프로젝트 루트 기준 상대 경로 + CLI 인자로 대체한다.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# 프로젝트 경로 (project paths)
# -----------------------------------------------------------------------------
# src/config.py -> 프로젝트 루트는 한 단계 위
ROOT_DIR = Path(__file__).resolve().parent.parent

# 데이터셋 (사용자가 직접 준비; 저장소에는 포함되지 않음)
DATA_DIR = ROOT_DIR / "data"
TRAIN_PATH = DATA_DIR / "train.txt"   # 원본: /home/dh/dataset/words/Train/train.txt
TEST_PATH = DATA_DIR / "test.txt"     # 원본: /home/dh/dataset/words/Test/test.txt

# 생성 산출물 디렉토리 (gitignore 대상)
WEIGHTS_DIR = ROOT_DIR / "weights"
WEIGHTS_CSV_DIR = ROOT_DIR / "weights_csv"
WEIGHTS_FPGA_DIR = ROOT_DIR / "weights_for_FPGA"

# 가중치 파일 이름 표준화.
# 주의: 원본 노트북은 'nextword_weights2.pth'로 저장/로드하지만,
#       디스크에 실제 존재했던 파일은 'nextword_weights.pth'였다.
#       혼동을 피하기 위해 아래 한 가지 이름으로 통일한다.
WEIGHTS_FILENAME = "nextword_weights.pth"
WEIGHTS_PATH = WEIGHTS_DIR / WEIGHTS_FILENAME

# FPGA용 C 헤더 출력 파일
C_HEADER_PATH = WEIGHTS_FPGA_DIR / "rnn_weights_q15.h"

# -----------------------------------------------------------------------------
# 모델 하이퍼파라미터 (model hyperparameters)
# -----------------------------------------------------------------------------
VOCAB_SIZE = 26        # 소문자 알파벳 a-z
INPUT_SIZE = VOCAB_SIZE
HIDDEN_SIZE = 128
OUTPUT_SIZE = VOCAB_SIZE

# -----------------------------------------------------------------------------
# 학습 설정 (training config)
# -----------------------------------------------------------------------------
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 3e-3
GRAD_CLIP = 1.0

# -----------------------------------------------------------------------------
# 고정소수점 포맷 (fixed-point format): Q1.15 signed int16
# -----------------------------------------------------------------------------
FRAC_BITS = 15
SCALE = 2 ** FRAC_BITS   # 32768
INT16_MIN = -32768
INT16_MAX = 32767

# 내보낼 텐서 이름 순서 (export order)
PARAM_NAMES = ["Wx", "Wh", "b", "Wo", "bo"]
