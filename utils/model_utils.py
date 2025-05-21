"""
모델링 유틸리티 모듈

이 모듈은 신경 신호 데이터 분석 및 모델링을 위한 함수들을 제공합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import os


def prepare_data_for_lstm(X, y, time_steps, step=1):
    """
    데이터를 LSTM 모델에 맞게 준비하는 함수
    
    Parameters:
    -----------
    X : ndarray
        특성 배열
    y : ndarray
        레이블 배열
    time_steps : int
        시계열 윈도우 크기
    step : int
        윈도우 이동 스텝 크기
        
    Returns:
    --------
    X_lstm : ndarray
        LSTM용 3D 입력 형태 (samples, time_steps, features)
    y_lstm : ndarray
        LSTM용 레이블
    """
    n_samples, n_features = X.shape
    
    # 3D 형태로 변환 가능한 최대 샘플 수 계산
    n_valid_samples = (n_samples - time_steps) // step + 1
    
    # LSTM 입력 준비
    X_lstm = np.zeros((n_valid_samples, time_steps, n_features))
    y_lstm = np.zeros(n_valid_samples)
    
    for i in range(n_valid_samples):
        start_idx = i * step
        end_idx = start_idx + time_steps
        X_lstm[i] = X[start_idx:end_idx]
        # 레이블은 마지막 타임스텝의 것을 사용
        y_lstm[i] = y[end_idx - 1]
    
    return X_lstm, y_lstm


def build_lstm_model(input_shape, num_classes, lstm_units=64, dropout_rate=0.3):
    """
    LSTM 모델 구축 함수
    
    Parameters:
    -----------
    input_shape : tuple
        입력 형태 (time_steps, features)
    num_classes : int
        클래스 수
    lstm_units : int
        LSTM 유닛 수
    dropout_rate : float
        드롭아웃 비율
        
    Returns:
    --------
    model : Sequential
        컴파일된 LSTM 모델
    """
    model = Sequential()
    
    # 첫 번째 LSTM 층
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # 두 번째 LSTM 층
    model.add(Bidirectional(LSTM(lstm_units // 2)))
    model.add(Dropout(dropout_rate))
    
    # 출력 층
    model.add(Dense(lstm_units // 4, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # 모델 컴파일
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model


def train_lstm_model(X_train, y_train, X_val, y_val, input_shape, num_classes, 
                    lstm_units=64, dropout_rate=0.3, epochs=100, batch_size=32, 
                    patience=10, model_path=None):
    """
    LSTM 모델 훈련 함수
    
    Parameters:
    -----------
    X_train : ndarray
        훈련 특성 배열
    y_train : ndarray
        훈련 레이블 배열
    X_val : ndarray
        검증 특성 배열
    y_val : ndarray
        검증 레이블 배열
    input_shape : tuple
        입력 형태 (time_steps, features)
    num_classes : int
        클래스 수
    lstm_units : int
        LSTM 유닛 수
    dropout_rate : float
        드롭아웃 비율
    epochs : int
        훈련 에포크 수
    batch_size : int
        배치 크기
    patience : int
        조기 종료 patience
    model_path : str
        모델 저장 경로
        
    Returns:
    --------
    model : Sequential
        훈련된 LSTM 모델
    history : History
        훈련 히스토리
    """
    # 모델 구축
    model = build_lstm_model(input_shape, num_classes, lstm_units, dropout_rate)
    
    # 콜백 정의
    callbacks = []
    
    # 조기 종료
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=1,
        restore_best_weights=True
    )
    callbacks.append(early_stopping)
    
    # 모델 체크포인트
    if model_path:
        # 디렉토리 생성
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        checkpoint = ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        callbacks.append(checkpoint)
    
    # 모델 훈련
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


def evaluate_lstm_model(model, X_test, y_test, class_names=None):
    """
    LSTM 모델 평가 함수
    
    Parameters:
    -----------
    model : Sequential
        훈련된 LSTM 모델
    X_test : ndarray
        테스트 특성 배열
    y_test : ndarray
        테스트 레이블 배열
    class_names : list
        클래스 이름 리스트
        
    Returns:
    --------
    results : dict
        평가 결과
    """
    # 예측
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 정확도 계산
    accuracy = accuracy_score(y_test, y_pred)
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    
    # 클래스 이름이 없으면 인덱스로 대체
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
    
    # 분류 보고서
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # 결과 시각화
    plt.figure(figsize=(10, 8))
    
    # 혼동 행렬 시각화
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.4f})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()
    
    # 결과 반환
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return results


def plot_learning_curves(history):
    """
    모델 학습 곡선 시각화 함수
    
    Parameters:
    -----------
    history : History
        모델 훈련 히스토리
    """
    plt.figure(figsize=(12, 5))
    
    # 손실 곡선
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='훈련 손실')
    plt.plot(history.history['val_loss'], label='검증 손실')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 정확도 곡선
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='훈련 정확도')
    plt.plot(history.history['val_accuracy'], label='검증 정확도')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def predict_neural_state(model, scaler, features, class_names=None):
    """
    신경 상태 예측 함수
    
    Parameters:
    -----------
    model : Sequential or Pipeline
        훈련된 모델
    scaler : StandardScaler
        특성 스케일러
    features : ndarray
        입력 특성
    class_names : list
        클래스 이름 리스트
        
    Returns:
    --------
    result : dict
        예측 결과
    """
    # 입력 형태 확인
    is_lstm = isinstance(model, Sequential)
    
    # 특성 스케일링
    features_scaled = scaler.transform(features)
    
    # 모델 타입에 따른 예측
    if is_lstm:
        # LSTM 모델은 3D 입력 형태 필요
        if len(features_scaled.shape) == 2:
            # 단일 샘플 또는 2D 입력인 경우 3D로 변환
            features_scaled = features_scaled.reshape(1, features_scaled.shape[0], features_scaled.shape[1])
        y_pred_proba = model.predict(features_scaled)
    else:
        # 일반 모델 (RandomForest 등)
        try:
            y_pred_proba = model.predict_proba(features_scaled)
        except:
            # predict_proba를 지원하지 않는 경우
            y_pred = model.predict(features_scaled)
            y_pred_proba = np.zeros((len(y_pred), len(np.unique(y_pred))))
            for i, pred in enumerate(y_pred):
                y_pred_proba[i, int(pred)] = 1.0
    
    # 예측 클래스
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 클래스 이름이 없으면 인덱스로 대체
    if class_names is None:
        class_names = [f"Class {i}" for i in range(y_pred_proba.shape[1])]
    
    # 결과 정리
    result = {
        'predicted_class': y_pred[0],
        'predicted_class_name': class_names[y_pred[0]],
        'class_probabilities': {name: float(prob) for name, prob in zip(class_names, y_pred_proba[0])}
    }
    
    return result


def save_model(model, scaler, class_names, output_dir, model_name):
    """
    모델 저장 함수
    
    Parameters:
    -----------
    model : Sequential or Pipeline
        훈련된 모델
    scaler : StandardScaler
        특성 스케일러
    class_names : list
        클래스 이름 리스트
    output_dir : str
        출력 디렉토리 경로
    model_name : str
        모델 이름
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 타입 확인
    is_lstm = isinstance(model, Sequential)
    
    # 모델 저장
    if is_lstm:
        model_path = os.path.join(output_dir, f"{model_name}.h5")
        model.save(model_path)
    else:
        model_path = os.path.join(output_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)
    
    # 스케일러 저장
    scaler_path = os.path.join(output_dir, f"{model_name}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    # 클래스 이름 저장
    import json
    class_names_path = os.path.join(output_dir, f"{model_name}_classes.json")
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)
    
    print(f"모델이 {output_dir}에 저장되었습니다.")


def load_model_components(model_dir, model_name):
    """
    모델 로드 함수
    
    Parameters:
    -----------
    model_dir : str
        모델 디렉토리 경로
    model_name : str
        모델 이름
        
    Returns:
    --------
    components : dict
        모델 컴포넌트
    """
    import json
    
    # 모델 경로
    keras_model_path = os.path.join(model_dir, f"{model_name}.h5")
    sklearn_model_path = os.path.join(model_dir, f"{model_name}.pkl")
    scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
    class_names_path = os.path.join(model_dir, f"{model_name}_classes.json")
    
    # 모델 로드
    if os.path.exists(keras_model_path):
        model = load_model(keras_model_path)
    elif os.path.exists(sklearn_model_path):
        model = joblib.load(sklearn_model_path)
    else:
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_name}")
    
    # 스케일러 로드
    scaler = joblib.load(scaler_path)
    
    # 클래스 이름 로드
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    components = {
        'model': model,
        'scaler': scaler,
        'class_names': class_names
    }
    
    return components


def create_stimulation_recommendation(nerve_state, feature_values=None):
    """
    신경 상태에 따른 전기자극 추천 함수
    
    Parameters:
    -----------
    nerve_state : str
        신경 상태 ('normal', 'damaged', 'recovery')
    feature_values : dict
        신경 신호 특성 값 (선택적)
        
    Returns:
    --------
    recommendation : dict
        전기자극 추천 정보
    """
    # 상태별 기본 추천 파라미터
    default_params = {
        'normal': {
            'frequency': 50,    # Hz
            'amplitude': 1.0,   # mA
            'pulse_width': 200, # µs
            'duty_cycle': 25,   # %
            'duration': 30,     # 분
            'sessions_per_day': 1,
            'treatment_days': 14
        },
        'damaged': {
            'frequency': 100,
            'amplitude': 3.0,
            'pulse_width': 500,
            'duty_cycle': 75,
            'duration': 60,
            'sessions_per_day': 2,
            'treatment_days': 30
        },
        'recovery': {
            'frequency': 75,
            'amplitude': 2.0,
            'pulse_width': 300,
            'duty_cycle': 50,
            'duration': 45,
            'sessions_per_day': 1,
            'treatment_days': 21
        }
    }
    
    # 상태별 추천 설명
    descriptions = {
        'normal': [
            "일반 유지 요법으로 적용합니다.",
            "신경 활성화와 혈류 개선을 위한 중간 주파수를 적용합니다.",
            "낮은 진폭으로 과도한 자극을 방지합니다.",
            "짧은 펄스폭으로 에너지 전달을 최소화합니다.",
            "낮은 듀티 사이클로 충분한 휴식 기간을 제공합니다."
        ],
        'damaged': [
            "손상된 신경의 회복을 촉진하기 위한 강화 요법입니다.",
            "높은 주파수로 BDNF/GDNF 신경영양인자 발현을 촉진합니다.",
            "높은 진폭으로 손상된 부위까지 전기장 전달을 강화합니다.",
            "긴 펄스폭으로 총 에너지 전달량을 증가시킵니다.",
            "높은 듀티 사이클로 자극 시간을 극대화합니다.",
            "1일 2회 적용으로 회복 촉진 효과를 강화합니다."
        ],
        'recovery': [
            "회복 중인 신경의 재생을 지원하는 중간 강도 요법입니다.",
            "중간 주파수로 축삭 성장과 슈반세포 활성화를 촉진합니다.",
            "중간 진폭으로 적절한 자극 강도를 유지합니다.",
            "중간 펄스폭으로 효율적인 에너지 전달을 제공합니다.",
            "균형 잡힌 듀티 사이클로 자극과 휴식의 균형을 유지합니다."
        ]
    }
    
    # 상태별 모니터링 권장사항
    monitoring = {
        'normal': [
            "2주마다 신경 전도 속도 측정",
            "부작용 발생 시 즉시 진폭 감소",
            "생체역학적 기능 평가 병행"
        ],
        'damaged': [
            "1주마다 신경 전도 속도 측정",
            "3일마다 염증 마커 평가",
            "지속적인 통증 관리",
            "1주마다 BDNF/GDNF 발현 수준 평가",
            "부작용 발생 시 자극 중단 및 파라미터 조정"
        ],
        'recovery': [
            "2주마다 신경 전도 속도 측정",
            "기능적 회복 평가 병행",
            "1주마다 축삭 성장률 모니터링",
            "부작용 발생 시 듀티 사이클 감소"
        ]
    }
    
    # 신경 상태 확인
    if nerve_state not in default_params:
        raise ValueError(f"지원되지 않는 신경 상태: {nerve_state}")
    
    # 파라미터 커스터마이징 (특성 값이 제공된 경우)
    params = default_params[nerve_state].copy()
    
    if feature_values is not None:
        # 예: 특성 값에 따라 파라미터 조정
        # 이 부분은 실제 사용 시 특성과 파라미터 간의 관계에 맞게 구현
        pass
    
    # 추천 정보 구성
    recommendation = {
        'nerve_state': nerve_state,
        'parameters': params,
        'description': descriptions[nerve_state],
        'monitoring': monitoring[nerve_state],
        'protocol_name': f"{nerve_state.capitalize()} 상태 전기자극 프로토콜"
    }
    
    return recommendation


def build_random_forest_classifier(n_estimators=100, max_depth=None, random_state=42):
    """
    랜덤 포레스트 분류기 구축 함수
    
    Parameters:
    -----------
    n_estimators : int
        결정 트리 수
    max_depth : int
        최대 트리 깊이
    random_state : int
        랜덤 시드
        
    Returns:
    --------
    pipeline : Pipeline
        전처리와 모델이 포함된 파이프라인
    """
    # 파이프라인 구축
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=n_estimators, 
                                            max_depth=max_depth, 
                                            random_state=random_state))
    ])
    
    return pipeline


def train_random_forest(X_train, y_train, X_val=None, y_val=None, 
                       n_estimators=100, max_depth=None, random_state=42):
    """
    랜덤 포레스트 모델 훈련 함수
    
    Parameters:
    -----------
    X_train : ndarray
        훈련 특성 배열
    y_train : ndarray
        훈련 레이블 배열
    X_val : ndarray
        검증 특성 배열 (선택적)
    y_val : ndarray
        검증 레이블 배열 (선택적)
    n_estimators : int
        결정 트리 수
    max_depth : int
        최대 트리 깊이
    random_state : int
        랜덤 시드
        
    Returns:
    --------
    model : Pipeline
        훈련된 모델
    """
    # 모델 구축
    model = build_random_forest_classifier(n_estimators, max_depth, random_state)
    
    # 모델 훈련
    model.fit(X_train, y_train)
    
    # 검증 세트가 있으면 성능 평가
    if X_val is not None and y_val is not None:
        val_accuracy = model.score(X_val, y_val)
        print(f"검증 정확도: {val_accuracy:.4f}")
    
    return model


def plot_feature_importance(model, feature_names, top_n=10):
    """
    특성 중요도 시각화 함수
    
    Parameters:
    -----------
    model : Pipeline or RandomForestClassifier
        훈련된 모델
    feature_names : list
        특성 이름 리스트
    top_n : int
        표시할 상위 특성 수
    """
    # 모델에서 분류기 추출
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps['classifier']
    else:
        classifier = model
    
    # 특성 중요도 추출
    importances = classifier.feature_importances_
    
    # 중요도에 따라 인덱스 정렬
    indices = np.argsort(importances)[::-1][:top_n]
    
    # 시각화
    plt.figure(figsize=(12, 6))
    plt.title(f'상위 {top_n}개 특성 중요도')
    plt.bar(range(top_n), importances[indices], align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # 결과 출력
    print("\n상위 특성 중요도:")
    for i in range(top_n):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
