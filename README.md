# 신경재생을 위한 고도화된 적응형 신경 전기자극 시스템

## 시스템 개요

최근 전기자극(ES)을 활용한 신경재생 및 기능회복 연구에서는 단순한 전기자극을 넘어 생체신호 피드백과 정밀한 자극 파라미터 조절이 치료 효과를 크게 향상시킬 수 있음이 밝혀졌습니다. 본 시스템은 이러한 연구 결과를 토대로, 실시간 생체신호 모니터링, 인공지능 기반 분석, 그리고 미세 조정 가능한 자극 프로토콜을 통합하여 개인화된 신경재생 치료 솔루션을 제공합니다.

특히 주목할 점은 분자생물학적 신경재생 메커니즘(BDNF/TrkB 경로, cAMP/PKA/CREB 활성화, GAP-43 발현 등)을 직접 타겟팅하는 자극 프로토콜과, 다채널 무선 자극 기술, 신경 섬유 유형별 선택적 자극 기능을 통합했다는 점입니다. 또한 위상차가 정밀하게 제어되는 전기자극 인가 방식을 도입하여 신경 타겟팅의 정확성과 효율성을 획기적으로 향상시켰습니다.

### 신경재생 메커니즘

전기자극(ES)은 여러 경로를 통해 신경재생을 촉진합니다:

1. **신경영양인자 상향 조절**: BDNF, GDNF와 같은 신경영양인자의 발현을 증가시켜 축삭 성장을 촉진
2. **cAMP/PKA/CREB 경로 활성화**: 세포 내 cAMP 수준을 증가시켜 재생 관련 유전자(RAGs) 발현 촉진
3. **슈반세포 활성화**: 손상된 신경의 재수초화와 영양 지원을 향상
4. **국소 혈류 개선**: 신경 주변 미세혈관 확장과 혈류 증가로 영양 공급 개선
5. **염증 환경 조절**: 대식세포 M2 극성화 촉진으로 재생에 유리한 면역 환경 조성

## 주요 특징

1. **통합적 시스템 설계**: 실시간 생체신호 모니터링, 인공지능 기반 분석, 미세 조정 가능한 자극 프로토콜이 통합된 개인화된 신경재생 치료 솔루션을 제공합니다.

2. **분자생물학적 메커니즘 타겟팅**: BDNF/TrkB 경로, cAMP/PKA/CREB 활성화, GAP-43 발현 등 신경재생의 핵심 분자경로를 직접 타겟팅하는 정밀 자극 프로토콜을 제공합니다.

3. **고급 하드웨어 아키텍처**: 
   - 멀티모달 센서 어레이로 다양한 생체신호 감지
   - 128채널 독립 제어 자극 모듈
   - 다중재질 복합 전극 어레이
   - 무선 전력 및 데이터 시스템

4. **지능형 소프트웨어 시스템**:
   - 고급 생체신호 처리 파이프라인
   - 다층 피드백 제어 아키텍처
   - 강화학습 기반 자극 최적화
   - 신경 회복 예측 모델과 치료 반응 패턴 분석

5. **특수 신경재생 프로토콜**:
   - 분자 메커니즘 기반 자극 (BDNF/TrkB, cAMP/CREB 경로 타겟팅)
   - 세포 유형별 타겟팅 (슈반 세포, 대식세포, 혈관내피세포)
   - 신경 섬유 유형별 선택적 자극 (운동, 감각, 자율신경)
   - 재생 단계별 맞춤형 프로토콜 (급성기부터 장기 기능 회복까지)

6. **다양한 임상 응용 분야**:
   - 말초신경 손상 재활
   - 척수 손상 기능 회복
   - 만성 신경병증성 통증 관리
   - 뇌졸중 후 재활
   - 당뇨병성 신경병증

## 최적 자극 매개변수

연구 결과에 따른 최적의 전기자극 매개변수:

| 매개변수 | 권장 범위 | 효과 |
|---------|----------|-----|
| 주파수 | 2-100Hz | 20Hz: BDNF 발현 최적화, 50Hz: cAMP 수준 촉진 |
| 강도 | 0.1-5mA | 조직 손상 없이 신경 활성화에 필요한 최소 강도 |
| 파형 | 양극성 펄스, 정현파 | 세포 유형에 따라 선택적 자극 가능 |
| 기간 | 20분/일, 2-4주 | 급성기: 짧은 고빈도, 만성기: 긴 저빈도 |
| 적용 시점 | 손상 후 24-48시간 내 시작 | 초기 적용이 세포 사멸 방지와 재생 촉진에 효과적 |

## 시스템 설치 및 사용 방법

### 필요 조건
- Python 3.8 이상
- TensorFlow 2.5 이상
- PyTorch 1.9 이상
- Arduino IDE (하드웨어 컨트롤러 프로그래밍용)
- 관련 하드웨어 컴포넌트 (상세 목록은 [하드웨어 요구사항](./docs/hardware-requirements.md) 참조)

### 설치 방법
```bash
# 저장소 클론
git clone https://github.com/JJshome/adaptive-neural-stimulation-system.git
cd adaptive-neural-stimulation-system

# 필요 패키지 설치
pip install -r requirements.txt

# 구성 설정
python setup.py
```

### 기본 사용법
```python
from neural_stim import AdaptiveStimulator

# 시스템 초기화
stimulator = AdaptiveStimulator(config_path='configs/default.yaml')

# 환자 데이터 로드 및 분석
patient_data = stimulator.load_patient_data('patient001.json')
analyzed_data = stimulator.analyze_baseline(patient_data)

# 자극 프로토콜 자동 생성
protocol = stimulator.generate_optimal_protocol(analyzed_data)

# 자극 시작 (실제 하드웨어 연결 필요)
stimulator.start_stimulation(protocol, duration_minutes=30)

# 실시간 모니터링 및 프로토콜 자동 조정
stimulator.enable_adaptive_feedback()
```

## 문서 구조

- [신경재생 특수 프로토콜](./neural-regen-protocols.md)
- [구현 로드맵 및 발전 방향](./implementation-roadmap.md)
- [응용 분야별 특화 프로토콜](./specialized-applications.md)
- [사용자 인터페이스 디자인](./user-interface-design.md)
- [구현 성공을 위한 핵심 조건](./key-success-factors.md)
- [시스템 아키텍처 다이어그램](./images/system-architecture.svg)
- [연구 결과 및 검증 데이터](./docs/research-validation.md)
- [API 문서](./docs/api-reference.md)
- [하드웨어 설계 상세](./docs/hardware-design.md)

## 코드 예시

알고리즘 및 모델링 코드 예시를 확인할 수 있습니다:

- [AI 분석 모델 코드](./code/ai_models/)
  - [생체신호 전처리 모듈](./code/ai_models/biosignal_preprocessing.py)
  - [강화학습 기반 자극 최적화](./code/ai_models/reinforcement_learning.py)
  - [신경회복 예측 모델](./code/ai_models/recovery_prediction.py)
  
- [시스템 제어 모듈](./code/control_system/)
  - [실시간 피드백 제어](./code/control_system/feedback_controller.py)
  - [자극 프로토콜 생성기](./code/control_system/protocol_generator.py)

## 프로젝트 현황

본 프로젝트는 현재 **개발 진행 중**이며, 다음 단계를 완료했습니다:
- ✅ 시스템 개념 설계 및 아키텍처 정의
- ✅ 핵심 알고리즘 프로토타입 구현
- ✅ 생체신호 처리 파이프라인 구축
- ⏳ 하드웨어 프로토타입 제작 (진행 중)
- ⏳ 소프트웨어-하드웨어 통합 (진행 중)
- 🔜 초기 실험 및 검증
- 🔜 전체 시스템 통합 및 최적화

## 기여 방법

이 프로젝트에 기여하고 싶으시다면:

1. 저장소를 포크(Fork)하고 클론(Clone)합니다
2. 새 브랜치를 생성합니다: `git checkout -b feature/your-feature-name`
3. 변경사항을 커밋합니다: `git commit -am 'Add some feature'`
4. 브랜치를 푸시합니다: `git push origin feature/your-feature-name`
5. Pull Request를 제출합니다

자세한 기여 가이드라인은 [CONTRIBUTING.md](./CONTRIBUTING.md)를 참조하세요.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](./LICENSE) 파일을 참조하세요.

## 연락처 및 팀 정보

- 프로젝트 리더: [이름](mailto:email@example.com)
- 연구 책임자: [이름](mailto:email@example.com)
- 개발 책임자: [이름](mailto:email@example.com)

문의사항이나 협업 제안은 [project@example.com](mailto:project@example.com)으로 연락 주세요.