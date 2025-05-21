# Adaptive Neural Stimulation System

신경재생을 위한 적응형 전기자극 최적화 시스템

## 개요

이 프로젝트는 신경 재생과 회복을 촉진하기 위한 적응형 전기자극 시스템을 구현합니다. 강화학습과 딥러닝 기술을 활용하여 신경 신호 패턴을 분석하고 최적의 자극 매개변수를 실시간으로 조정합니다.

### 연구 배경

전기자극(ES)은 신경 재생을 촉진하는 유망한 치료법으로, 다음과 같은 메커니즘을 통해 작용합니다:

- 신경영양인자(BDNF, GDNF) 상향 조절
- cAMP 증가
- 재생 관련 유전자(RAGs) 발현 촉진
- 슈반세포 활성화
- 혈류 개선
- 대식세포 M2 극성화 촉진

이러한 메커니즘은 축삭 성장을 촉진하고 기능 회복을 가속화하는 것으로 알려져 있습니다.

## 주요 기능

- **신경 신호 처리**: 노이즈 제거, 특성 추출, 패턴 인식
- **강화학습 기반 최적화**: DQN 에이전트를 통한 자극 매개변수 최적화
- **시계열 예측**: LSTM 모델을 활용한 신경 반응 예측
- **적응형 자극 제어**: 실시간 피드백에 기반한 자극 매개변수 조정
- **데이터 시각화**: 신호, 스펙트럼, 최적화 과정 시각화

## 시스템 구조

프로젝트는 다음과 같은 모듈로 구성되어 있습니다:

### 유틸리티 모듈
- `signal_processor.py`: 신경 신호 데이터 처리
- `data_handler.py`: 데이터 로드 및 변환
- `stimulation_controller.py`: 전기자극 패턴 생성 및 제어
- `parameter_optimizer.py`: 자극 매개변수 최적화
- `visualizer.py`: 데이터 시각화

### 모델 모듈
- `dqn_agent.py`: DQN 강화학습 에이전트
- `lstm_model.py`: LSTM 기반 시계열 예측 모델
- `stimulation_environment.py`: 강화학습 환경

### 메인 애플리케이션
- `adaptive_stimulation_system.py`: 시스템 통합 및 메인 애플리케이션

## 설치 방법

이 프로젝트를 실행하기 위해 필요한 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

## 사용 방법

### 기본 실행

```bash
python adaptive_stimulation_system.py
```

### 커스텀 설정으로 실행

```python
from adaptive_stimulation_system import AdaptiveStimulationSystem

# 시스템 설정
config = {
    'sampling_rate': 1000.0,
    'sequence_length': 100,
    'feature_dim': 5,
    'use_lstm': True,
    'use_reinforcement_learning': True,
    'save_path': 'results',
    'model_path': 'models/saved'
}

# 시스템 인스턴스 생성
system = AdaptiveStimulationSystem(config)

# 데이터 로드
data, sampling_rate = system.load_data('path_to_data.csv')

# 데이터 전처리
processed_data = system.preprocess_data(data)

# 강화학습 에이전트 학습
rewards = system.train_dqn(num_episodes=100)

# 적응형 자극 적용
result = system.adaptive_stimulation(processed_data, duration=5.0)

# 결과 시각화
system.visualize_results({
    'signal': processed_data,
    'stimulation_waveform': result['stimulation_waveform'],
    'rewards': rewards
})
```

## 개발 방향

이 프로젝트의 향후 개발 방향은 다음과 같습니다:

1. **최적화 알고리즘 개선**: 더 효율적인 매개변수 탐색을 위한 알고리즘 개발
2. **생체 모델 통합**: 더 현실적인 신경 반응 모델링을 위한 생체 모델 통합
3. **하드웨어 인터페이스**: 실제 자극 장치와의 인터페이스 구현
4. **임상 데이터 분석**: 실제 환자 데이터에 기반한 모델 검증 및 개선

## 연구 의의

본 시스템은 다음과 같은 측면에서 신경 재생 연구에 기여할 수 있습니다:

- 환자별 맞춤형 자극 매개변수 최적화
- 신경 재생 과정의 실시간 모니터링 및 피드백
- 다양한 신경 손상 유형에 대한 최적 자극 패턴 발견
- 신경 재생 메커니즘에 대한 깊은 이해 촉진

## 참고 문헌

- Zhang, X., & Ji, L. (2022). Electrical stimulation for neural regeneration: A comprehensive review. *Neural Regeneration Research*.
- Smith, A., et al. (2023). Adaptive neural stimulation systems: Current approaches and future directions. *Journal of Neural Engineering*.
- Johnson, B. (2024). Reinforcement learning applications in neuromodulation. *Frontiers in Neuroscience*.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.
