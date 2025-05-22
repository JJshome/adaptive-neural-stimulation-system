# 모델 훈련 및 평가 구현 가이드

이 문서는 적응형 신경 전기자극 시스템에서 구현된 모델 훈련 및 평가 모듈에 대한 가이드입니다.

## 구현된 모델 목록

### 1. 지도 학습 모델
- **LSTM 네트워크**: 시계열 신경 신호 분석을 위한 주 모델 (`models/lstm_model.py`)
  - 기본 LSTM
  - 양방향 LSTM
  - 인코더-디코더 LSTM
- **CNN-LSTM 하이브리드**: 공간-시간적 특성 추출 (`models/cnn_lstm_model.py`)
  - 기본 CNN-LSTM
  - 병렬 CNN-LSTM
  - 계층적 CNN-LSTM
- **Transformer 기반 모델**: 장기 의존성 파악 (`models/transformer_model.py`)
  - 기본 Transformer
  - CNN-Transformer 하이브리드
- **Random Forest, XGBoost**: 기준선 성능 평가 (외부 라이브러리 사용)

### 2. 비지도 학습 모델
- **변이형 오토인코더(VAE)**: 특성 추출 및 이상 탐지 (`models/unsupervised_models.py`)
  - 기본 VAE
  - 컨볼루션 VAE
- **K-means 클러스터링**: 신호 패턴 그룹화 (`models/unsupervised_models.py`)
- **주성분 분석(PCA)**: 차원 축소 및 특성 추출 (`models/unsupervised_models.py`)

### 3. 강화학습 모델
- **PPO(Proximal Policy Optimization)**: 자극 파라미터 최적화 (`models/ppo_model.py`)
- **DQN(Deep Q-Network)**: 이산적 자극 패턴 결정 (`models/dqn_agent.py`)
- **DDPG(Deep Deterministic Policy Gradient)**: 연속적 파라미터 조정 (`code/ai_models/reinforcement_learning.py`)

### 4. 평가 지표
- **분류 성능**: 정확도, 정밀도, 재현율, F1 점수
- **회귀 성능**: MAE, RMSE
- **강화학습 성능**: 누적 보상
- **일반화 능력**: 10-fold 교차 검증

## 사용 방법

### 통합 훈련 및 평가 스크립트 실행
```bash
python train_evaluate_models.py
```

이 스크립트는 다음 작업을 수행합니다:
1. 신경 신호 데이터 로드 및 전처리
2. 지도 학습 모델 훈련 및 평가
3. 비지도 학습 모델 훈련 및 평가
4. 10-fold 교차 검증 수행
5. 강화학습 모델 훈련 (시뮬레이션 환경 사용)
6. 결과 시각화 및 저장

### 개별 모델 사용 예시

#### LSTM 모델
```python
from models.lstm_model import LSTMModel

# 모델 초기화
lstm_model = LSTMModel(
    sequence_length=100,
    feature_dim=16,
    output_dim=3,
    lstm_units=64,
    dropout_rate=0.3
)

# 모델 훈련
lstm_model.train(X_train, y_train, epochs=50, batch_size=32)

# 예측
predictions = lstm_model.predict(X_test)

# 모델 저장
lstm_model.save("models/saved/lstm_model.h5")
```

#### CNN-LSTM 모델
```python
from models.cnn_lstm_model import CNNLSTM

# 모델 초기화
cnn_lstm_model = CNNLSTM(
    sequence_length=100,
    feature_dim=16,
    output_dim=3,
    cnn_filters=[32, 64],
    kernel_sizes=[3, 3],
    pool_sizes=[2, 2],
    lstm_units=[64, 32],
    dropout_rate=0.3
)

# 모델 훈련
cnn_lstm_model.train(X_train, y_train, epochs=50, batch_size=32)
```

#### Transformer 모델
```python
from models.transformer_model import TransformerModel

# 모델 초기화
transformer_model = TransformerModel(
    sequence_length=100,
    feature_dim=16,
    output_dim=3,
    num_transformer_blocks=2,
    num_heads=4,
    embed_dim=64,
    ff_dim=128,
    dropout_rate=0.1
)

# 모델 훈련
transformer_model.train(X_train, y_train, epochs=50, batch_size=32)
```

#### VAE 모델
```python
from models.unsupervised_models import VariationalAutoencoder

# 모델 초기화
vae_model = VariationalAutoencoder(
    sequence_length=100,
    feature_dim=16,
    latent_dim=16,
    encoder_units=[128, 64],
    decoder_units=[64, 128],
    dropout_rate=0.2
)

# 모델 훈련
vae_model.train(X_train, epochs=50, batch_size=32)

# 잠재 공간 인코딩
latent_vectors = vae_model.encode(X_test)

# 이상치 탐지
anomalies, threshold = vae_model.detect_anomalies(X_test)
```

#### PPO 모델
```python
from models.ppo_model import PPOAgent

# 에이전트 초기화
ppo_agent = PPOAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    action_bounds=action_bounds,
    buffer_size=2048,
    batch_size=64
)

# 훈련
ppo_agent.train(env, epochs=50, steps_per_epoch=2048)

# 모델 저장
ppo_agent.save_model("models/saved/ppo_model")
```

## 결과 해석

훈련 결과는 다음 위치에 저장됩니다:
- **훈련된 모델**: `trained_models/` 디렉토리
- **평가 결과**: `results/` 디렉토리
  - 모델 비교 그래프: `supervised_model_comparison.png`
  - 교차 검증 결과: `cross_validation_results.png`
  - PCA 설명된 분산: `pca_explained_variance.png`
  - K-means 클러스터링: `kmeans_clustering.png`
  - VAE 잠재 공간: `vae_latent_space.png`
  - 메트릭 CSV: `model_metrics.csv`

## 추가 개선 방향

1. **데이터 처리 개선**:
   - 실제 신경 신호 데이터 로드 및 전처리 구현
   - 데이터 증강 기법 추가

2. **모델 최적화**:
   - 하이퍼파라미터 튜닝 자동화
   - 모델 앙상블 구현

3. **강화학습 환경**:
   - 실제 신경 자극 시뮬레이션 환경 구현
   - 현실적인 보상 함수 설계

4. **배포 및 통합**:
   - 웹 인터페이스 구현
   - 실시간 처리 시스템 구축
