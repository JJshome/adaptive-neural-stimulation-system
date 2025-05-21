# 적응형 신경 전기자극 시스템 (Adaptive Neural Stimulation System)

신경재생을 위한 고도화된 적응형 신경 전기자극 시스템 설계 프로젝트입니다. 이 시스템은 신경 손상 후 재생 과정에서 실시간으로 신경 상태를 분석하고, 최적화된 전기자극 파라미터를 적용하여 재생 효과를 극대화합니다.

## 프로젝트 개요

신경 손상 후 재생을 위한 전기자극(Electrical Stimulation, ES)은 다양한 연구에서 그 효과가 입증되었습니다. 그러나 기존의 전기자극 시스템은 고정된 파라미터를 사용하여 개인별 신경 상태나 회복 단계에 따른 최적화가 부족했습니다. 본 프로젝트는 기계학습과 신호처리 기술을 활용하여 실시간으로 신경 신호를 분석하고, 개인화된 최적 자극 파라미터를 도출하는 적응형 시스템을 개발하고자 합니다.

## 주요 기능

1. **실시간 신경 신호 분석**: 다양한 전처리 및 특성 추출 기법을 통해 신경 상태를 정확히 평가
2. **신경 상태 분류**: LSTM 기반 딥러닝 모델을 활용한 신경 상태(정상, 손상, 회복 중) 분류
3. **맞춤형 자극 파라미터 최적화**: 베이지안 최적화를 통한 개인별 최적 자극 파라미터 도출
4. **적응형 자극 제어**: 신경 상태 변화에 따라 자동으로 자극 파라미터를 조정하는 피드백 시스템
5. **효과 모니터링 및 분석**: 자극 효과의 지속적인 모니터링 및 분석 도구

## 폴더 구조

```
.
├── code/                    # 주요 코드 및 알고리즘
├── data/                    # 데이터 및 샘플
│   ├── neural_recordings/   # 신경 신호 데이터
│   └── processed/           # 전처리된 데이터
├── docs/                    # 문서 및 설명서
├── images/                  # 이미지 및 다이어그램
├── models/                  # 저장된 모델 및 파라미터
│   └── stimulation_protocols/ # 최적화된 자극 프로토콜
├── notebooks/               # Jupyter 노트북
├── tests/                   # 테스트 코드
├── utils/                   # 유틸리티 함수 및 모듈
├── adaptive_stimulation_system.py  # 메인 시스템 코드
└── requirements.txt         # 의존성 패키지 목록
```

## 주요 노트북

- `notebooks/neural_signal_exploration.ipynb`: 신경 신호 데이터 탐색 및 기초 분석
- `notebooks/neural_signal_preprocessing.ipynb`: 신경 신호 전처리 및 특성 추출
- `notebooks/lstm_neural_signal_classification.ipynb`: LSTM 모델을 활용한 신경 상태 분류
- `notebooks/stimulation_parameter_optimization.ipynb`: 전기자극 파라미터 최적화 및 효과 분석

## 유틸리티 모듈

- `utils/data_utils.py`: 데이터 로딩 및 전처리 유틸리티
- `utils/model_utils.py`: 모델 관련 유틸리티 함수
- `utils/stimulation_utils.py`: 전기자극 시뮬레이션 및 분석 유틸리티

## 설치 방법

### 요구 사항

- Python 3.8+
- TensorFlow 2.x
- NumPy, Pandas, SciPy, Matplotlib
- scikit-learn

### 설치

```bash
# 저장소 복제
git clone https://github.com/JJshome/adaptive-neural-stimulation-system.git
cd adaptive-neural-stimulation-system

# 가상 환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 패키지 설치
pip install -r requirements.txt
```

## 사용 방법

### 데이터 준비

- `data/neural_recordings/` 폴더에 신경 신호 데이터(.csv 또는 .npy 형식)를 넣습니다.
- 데이터 형식은 (시간샘플 × 채널) 구조를 가지며, 마지막 열에는 신경 상태 레이블이 있어야 합니다.

### 신경 신호 분석 및 모델 훈련

```python
# 예시 코드
from utils.data_utils import load_neural_data, preprocess_neural_signals
from utils.model_utils import train_lstm_model

# 데이터 로드 및 전처리
data = load_neural_data('data/neural_recordings/')
signals = data['signals']
labels = data['labels']
X, y, feature_names = preprocess_neural_signals(signals, labels)

# 모델 훈련
train_lstm_model(X_train, y_train, X_val, y_val, input_shape, num_classes)
```

### 전기자극 파라미터 최적화

```python
# 예시 코드
from utils.stimulation_utils import simulate_stimulation_response

# 전기자극 시뮬레이션
params = {
    'frequency': 50,  # Hz
    'amplitude': 2.0,  # mA
    'pulse_width': 300,  # µs
    'duty_cycle': 50,  # %
    'duration': 30  # minutes
}

response = simulate_stimulation_response(params, nerve_state='damaged')
```

## 논문 및 참고자료

- Yao L, et al. (2018). "Electrical stimulation for peripheral nerve regeneration: Current developments and challenges."
- Huang J, et al. (2021). "Machine learning applications in neural engineering and neuroscience."

## 기여 방법

본 프로젝트에 기여하고 싶으신 분들은 [CONTRIBUTING.md](CONTRIBUTING.md) 문서를 참고해주세요. 모든 의견과 제안을 환영합니다.

## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

## 연락처

프로젝트에 관한 문의나 제안은 GitHub 이슈를 통해 남겨주시기 바랍니다.