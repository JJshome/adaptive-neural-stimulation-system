# 강화학습 기반 신경 전기자극 최적화 시스템

이 프로젝트는 실시간 생체신호 피드백을 기반으로 신경 전기자극 파라미터를 자동 최적화하는 강화학습 시스템을 구현합니다.

## 주요 기능

- 딥 강화학습(DDPG, TD3 기반) 알고리즘을 통한 자극 파라미터 최적화
- 바이오마커 특화 신경망 구조로 신경재생 촉진에 최적화
- 생체신호 실시간 피드백 처리 및 적응형 자극 조절
- 안전 제약 조건을 고려한 자극 파라미터 적용
- 신경재생 과정에 특화된 특수 자극 프로토콜 지원

## 전기자극의 신경재생 메커니즘

전기자극은 다양한 메커니즘을 통해 신경재생을 촉진합니다:

- 신경영양인자(BDNF, GDNF) 상향 조절
- cAMP 증가 및 재생 관련 유전자(RAGs) 발현 촉진
- 슈반세포 활성화 및 혈류 개선
- 대식세포 M2 극성화 촉진

이러한 메커니즘은 축삭 성장을 촉진하고 기능 회복을 가속화합니다.

## 임상적 응용

- 신경손상 초기 단계에서 효과적인 자극 파라미터 자동 최적화
- 신경 도관과 결합된 전기자극 및 경피적 전기자극(TENS) 방식 지원
- 신경재생 관련 분자적 기전 추적 및 피드백 반영
- 장기 추적 관찰을 위한 데이터 수집 및 분석

## 시스템 구성

- `models/`: 강화학습 모델 클래스 구현 (액터-크리틱 네트워크 등)
- `utils/`: 유틸리티 함수 및 클래스 (경험 재현 버퍼, 노이즈 생성기 등)
- `controllers/`: 자극 제어 시스템 (적응형 파라미터 조절, 안전 제약 처리 등)
- `protocols/`: 사전 정의된 특수 자극 프로토콜 구현
- `examples/`: 시스템 사용 예제 및 데모

## 사용 방법

```python
from controllers.adaptive_controller import AdaptiveStimulationController
from utils.biomarker_analyzer import BiomarkerAnalyzer

# 바이오마커 분석기 초기화
analyzer = BiomarkerAnalyzer(
    biomarkers=['bdnf_expression', 'axon_growth', 'inflammation']
)

# 자극 컨트롤러 초기화
controller = AdaptiveStimulationController(
    state_features=analyzer.get_feature_names(),
    stimulation_params={
        'amplitude': (0.0, 5.0),   # mA
        'frequency': (10.0, 100.0), # Hz
        'pulse_width': (100.0, 500.0), # μs
        'duty_cycle': (0.1, 0.5)
    },
    biomarker_indices=analyzer.get_biomarker_indices()
)

# 시스템 실행 (실시간 피드백 루프)
while not done:
    # 생체신호 수집
    biosignals = collect_biosignals()
    
    # 현재 상태 분석
    current_state = analyzer.process_signals(biosignals)
    
    # 최적 자극 파라미터 계산
    stim_params = controller.get_stimulation_params(current_state)
    
    # 자극 적용 및 결과 관찰
    apply_stimulation(stim_params)
    next_biosignals = collect_biosignals()
    next_state = analyzer.process_signals(next_biosignals)
    
    # 보상 계산 및 컨트롤러 업데이트
    reward = analyzer.calculate_reward(current_state, next_state)
    controller.update(current_state, stim_params, reward, next_state)
```

## 설치 방법

```bash
git clone https://github.com/JJshome/adaptive-neural-stimulation-system.git
cd adaptive-neural-stimulation-system
pip install -r requirements.txt
```

## 기여 방법

프로젝트에 기여하려면 CONTRIBUTING.md 문서를 참조하세요.

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 LICENSE 파일을 참조하세요.