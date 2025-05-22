# 웹 인터페이스 모듈

이 디렉토리는 적응형 신경 전기자극 시스템의 웹 인터페이스를 제공합니다. Flask 웹 프레임워크를 사용하여 구현되었으며, 실시간 신경 신호 시각화, 자극 파라미터 조정, 회복 지표 모니터링 등의 기능을 제공합니다.

## 주요 기능

- **실시간 신호 시각화**: 다중 채널 신경 신호의 실시간 시각화
- **신호 분석**: 주파수 스펙트럼 분석 및 시간-주파수 분석(스펙트로그램)
- **파라미터 조정**: 자극 주파수, 진폭, 펄스 폭, 듀티 사이클 조정
- **제어 방식 선택**: PID, Q-Learning, Actor-Critic, 수동 제어 방식 지원
- **회복 지표 추적**: 축삭 밀도, 전도 속도, 기능적 회복 지표 시각화
- **이벤트 관리**: 자극 시작/중지, 파라미터 변경 등의 이벤트 기록
- **데이터 저장**: 수집된 신호 데이터 및 설정 저장/불러오기

## 기술 스택

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **시각화**: Chart.js
- **스타일링**: Bootstrap 5
- **비동기 통신**: Fetch API

## 디렉토리 구조

```
web_interface/
├── app.py                # Flask 애플리케이션 서버
├── templates/            # HTML 템플릿
│   ├── index.html        # 메인 대시보드 템플릿
│   ├── monitoring.html   # 신호 모니터링 템플릿
│   ├── parameters.html   # 자극 파라미터 템플릿
│   ├── analytics.html    # 분석 및 보고서 템플릿
│   └── settings.html     # 시스템 설정 템플릿
├── static/               # 정적 파일
│   ├── css/              # CSS 스타일시트
│   │   └── style.css     # 메인 스타일시트
│   ├── js/               # JavaScript 파일
│   │   ├── dashboard.js  # 대시보드 기능
│   │   ├── monitoring.js # 모니터링 기능
│   │   └── parameters.js # 파라미터 조정 기능
│   └── img/              # 이미지 파일
└── README.md             # 이 파일
```

## 설치 및 실행

### 요구사항

- Python 3.8+
- Flask
- NumPy
- Chart.js (CDN으로 제공됨)
- Bootstrap 5 (CDN으로 제공됨)

### 설치

```bash
# 가상 환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 패키지 설치
pip install flask numpy
```

### 실행

```bash
# 웹 서버 시작
cd code/web_interface
python app.py
```

기본적으로 웹 서버는 http://localhost:5000 에서 실행됩니다.

## API 엔드포인트

웹 인터페이스는 다음과 같은 RESTful API 엔드포인트를 제공합니다:

### 페이지 라우트

- `GET /`: 메인 대시보드 페이지
- `GET /monitoring`: 신호 모니터링 페이지
- `GET /parameters`: 자극 파라미터 페이지
- `GET /analytics`: 분석 및 보고서 페이지
- `GET /settings`: 시스템 설정 페이지

### 데이터 API

- `GET /api/system-state`: 현재 시스템 상태 정보 반환
- `GET /api/signal-data`: 최신 신경 신호 데이터 반환
- `GET /api/history-data`: 파라미터 및 상태 히스토리 데이터 반환
- `GET /api/recovery-metrics`: 회복 지표 데이터 반환
- `GET /api/list-settings`: 저장된 설정 파일 목록 반환

### 제어 API

- `POST /api/set-neural-state`: 현재 신경 상태 설정
- `POST /api/set-target-state`: 목표 신경 상태 설정
- `POST /api/set-parameters`: 자극 파라미터 설정
- `POST /api/set-control-method`: 제어 방식 설정
- `POST /api/start-stimulation`: 자극 시작
- `POST /api/stop-stimulation`: 자극 중지
- `POST /api/save-settings`: 현재 설정 저장
- `POST /api/load-settings`: 저장된 설정 불러오기

## 특징 및 개선점

### 실시간 신호 시각화

- 다중 채널 신호의 실시간 표시
- 시간 창 및 진폭 스케일 조정 가능
- 자동 스케일링 옵션
- 채널별 표시/숨김 제어

### 고급 신호 분석

- 주파수 스펙트럼 분석 (FFT 기반)
- 스펙트로그램 시각화 (시간-주파수 분석)
- 신호 통계 계산 (RMS, SNR, 엔트로피 등)
- 디지털 필터링 기능 (저역, 고역, 대역, 노치 필터)

### 적응형 제어

- 다양한 제어 알고리즘 선택 (PID, Q-Learning, Actor-Critic)
- 신경 상태에 따른 자동 파라미터 최적화
- 실시간 피드백 기반 자극 조정

### 회복 추적

- 축삭 밀도, 전도 속도, 기능적 회복 지표 시각화
- 시간에 따른 회복 진행 추적
- 자극 효과의 장기 분석

## 향후 개발 계획

1. **보안 강화**: 사용자 인증 및 권한 관리 시스템 추가
2. **데이터베이스 통합**: 관계형 데이터베이스를 이용한 데이터 영구 저장
3. **알림 시스템**: 중요 이벤트 및 상태 변화에 대한 실시간 알림
4. **모바일 최적화**: 모바일 장치에서의 사용성 개선
5. **오프라인 모드**: 네트워크 연결 없이도 작동하는 기능 구현
6. **데이터 내보내기**: 다양한 형식(CSV, JSON, Excel)으로 데이터 내보내기
7. **3D 시각화**: 신경 구조 및 자극 효과의 3D 시각화

## 기여

이 웹 인터페이스 모듈에 기여하고자 하는 경우, 다음 가이드라인을 참고하세요:

1. 기능 추가나 버그 수정은 별도의 브랜치에서 작업
2. 코드 스타일 가이드라인 준수
3. 코드 변경 시 적절한 테스트 추가
4. 문서화 업데이트

## 라이선스

이 웹 인터페이스 모듈은 메인 프로젝트와 동일한 라이선스 하에 배포됩니다.
