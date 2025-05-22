"""CNN-LSTM 하이브리드 모델

이 모듈은 공간-시간적 특성 추출을 위한 CNN-LSTM 하이브리드 모델을 구현합니다.
CNN 레이어로 공간적 특성을 추출하고 LSTM 레이어로 시간적 의존성을 모델링합니다.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Dense, LSTM, Conv1D, MaxPooling1D, Flatten,
                                        Dropout, Input, Concatenate, BatchNormalization,
                                        TimeDistributed)
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("CNN-LSTM 모델 사용을 위해 TensorFlow가 필요합니다.")


class CNNLSTM:
    """CNN-LSTM 하이브리드 모델 클래스"""
    
    def __init__(self, 
                 sequence_length: int, 
                 feature_dim: int,
                 output_dim: int = 1, 
                 cnn_filters: List[int] = [64, 128], 
                 kernel_sizes: List[int] = [3, 3],
                 pool_sizes: List[int] = [2, 2],
                 lstm_units: List[int] = [64, 32],
                 dense_units: List[int] = [64, 32], 
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001):
        """
        CNN-LSTM 모델 초기화
        
        매개변수:
            sequence_length (int): 입력 시퀀스 길이
            feature_dim (int): 각 시점당 특성 차원 수
            output_dim (int): 출력 차원 수
            cnn_filters (List[int]): CNN 레이어의 필터 수 리스트
            kernel_sizes (List[int]): CNN 레이어의 커널 크기 리스트
            pool_sizes (List[int]): 풀링 레이어의 풀 크기 리스트
            lstm_units (List[int]): LSTM 레이어의 유닛 수 리스트
            dense_units (List[int]): 완전 연결 레이어의 유닛 수 리스트
            dropout_rate (float): 드롭아웃 비율
            learning_rate (float): 학습률
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.cnn_filters = cnn_filters
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # 모델 빌드
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """
        CNN-LSTM 하이브리드 모델 구축
        
        반환값:
            tf.keras.Model: 구축된 모델
        """
        # 입력 레이어
        inputs = Input(shape=(self.sequence_length, self.feature_dim))
        
        # CNN 레이어 스택
        x = inputs
        for i in range(len(self.cnn_filters)):
            x = Conv1D(filters=self.cnn_filters[i], 
                       kernel_size=self.kernel_sizes[i],
                       activation='relu',
                       padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=self.pool_sizes[i])(x)
        
        # LSTM 레이어 스택
        for i in range(len(self.lstm_units)):
            return_sequences = i < len(self.lstm_units) - 1
            x = LSTM(self.lstm_units[i], return_sequences=return_sequences)(x)
            x = Dropout(self.dropout_rate)(x)
        
        # 완전 연결 레이어 스택
        for units in self.dense_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
        
        # 출력 레이어
        outputs = Dense(self.output_dim)(x)
        
        # 모델 생성
        model = Model(inputs, outputs)
        
        # 컴파일
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             epochs: int = 100, batch_size: int = 32, 
             validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
             verbose: int = 1,
             callbacks: Optional[List[tf.keras.callbacks.Callback]] = None) -> tf.keras.callbacks.History:
        """
        모델 학습
        
        매개변수:
            X_train (np.ndarray): 학습 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            y_train (np.ndarray): 학습 타겟 데이터, 형태: (샘플 수, 출력 차원)
            epochs (int): 학습 에폭 수
            batch_size (int): 배치 크기
            validation_data (Optional[Tuple[np.ndarray, np.ndarray]]): 검증 데이터
            verbose (int): 출력 상세도
            callbacks (Optional[List[tf.keras.callbacks.Callback]]): 콜백 함수 리스트
            
        반환값:
            tf.keras.callbacks.History: 학습 히스토리
        """
        return self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=verbose,
            callbacks=callbacks
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        시퀀스 예측
        
        매개변수:
            X (np.ndarray): 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            
        반환값:
            np.ndarray: 예측 결과
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """
        모델 평가
        
        매개변수:
            X_test (np.ndarray): 테스트 입력 데이터
            y_test (np.ndarray): 테스트 타겟 데이터
            
        반환값:
            Tuple[float, float]: (손실값, MAE)
        """
        return self.model.evaluate(X_test, y_test)
    
    def save(self, path: str) -> None:
        """
        모델 저장
        
        매개변수:
            path (str): 저장 경로
        """
        self.model.save(path)
        
    def load(self, path: str) -> None:
        """
        모델 로드
        
        매개변수:
            path (str): 모델 파일 경로
        """
        self.model.load_weights(path)


class ParallelCNNLSTM(CNNLSTM):
    """병렬 CNN-LSTM 하이브리드 모델 클래스
    
    다양한 커널 크기의 여러 CNN 브랜치를 병렬로 사용하여
    다양한 시간 스케일의 패턴을 포착합니다.
    """
    
    def _build_model(self) -> tf.keras.Model:
        """
        병렬 CNN-LSTM 하이브리드 모델 구축
        
        반환값:
            tf.keras.Model: 구축된 모델
        """
        # 입력 레이어
        inputs = Input(shape=(self.sequence_length, self.feature_dim))
        
        # 다양한 커널 크기를 가진 병렬 CNN 브랜치
        cnn_branches = []
        for i, kernel_size in enumerate([2, 3, 5, 7]):  # 다양한 커널 크기
            branch = Conv1D(filters=self.cnn_filters[0], 
                          kernel_size=kernel_size,
                          activation='relu',
                          padding='same')(inputs)
            branch = BatchNormalization()(branch)
            branch = MaxPooling1D(pool_size=2)(branch)
            cnn_branches.append(branch)
        
        # 병렬 브랜치 결합
        if len(cnn_branches) > 1:
            x = Concatenate(axis=1)(cnn_branches)
        else:
            x = cnn_branches[0]
        
        # 추가 CNN 레이어
        for i in range(1, len(self.cnn_filters)):
            x = Conv1D(filters=self.cnn_filters[i], 
                      kernel_size=self.kernel_sizes[i],
                      activation='relu',
                      padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=self.pool_sizes[i])(x)
        
        # LSTM 레이어 스택
        for i in range(len(self.lstm_units)):
            return_sequences = i < len(self.lstm_units) - 1
            x = LSTM(self.lstm_units[i], return_sequences=return_sequences)(x)
            x = Dropout(self.dropout_rate)(x)
        
        # 완전 연결 레이어 스택
        for units in self.dense_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
        
        # 출력 레이어
        outputs = Dense(self.output_dim)(x)
        
        # 모델 생성
        model = Model(inputs, outputs)
        
        # 컴파일
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['mae']
        )
        
        return model


class HierarchicalCNNLSTM(CNNLSTM):
    """계층적 CNN-LSTM 모델 클래스
    
    시퀀스 데이터를 하위 시퀀스로 나누고 각 하위 시퀀스에 CNN을 적용한 후
    LSTM으로 시간적 관계를 모델링합니다.
    """
    
    def __init__(self, 
                 sequence_length: int, 
                 feature_dim: int,
                 output_dim: int = 1, 
                 sub_seq_length: int = 10,
                 cnn_filters: List[int] = [64, 128], 
                 kernel_sizes: List[int] = [3, 3],
                 pool_sizes: List[int] = [2, 2],
                 lstm_units: List[int] = [64, 32],
                 dense_units: List[int] = [64, 32], 
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001):
        """
        계층적 CNN-LSTM 모델 초기화
        
        매개변수:
            sequence_length (int): 입력 시퀀스 길이
            feature_dim (int): 각 시점당 특성 차원 수
            output_dim (int): 출력 차원 수
            sub_seq_length (int): 하위 시퀀스 길이
            cnn_filters (List[int]): CNN 레이어의 필터 수 리스트
            kernel_sizes (List[int]): CNN 레이어의 커널 크기 리스트
            pool_sizes (List[int]): 풀링 레이어의 풀 크기 리스트
            lstm_units (List[int]): LSTM 레이어의 유닛 수 리스트
            dense_units (List[int]): 완전 연결 레이어의 유닛 수 리스트
            dropout_rate (float): 드롭아웃 비율
            learning_rate (float): 학습률
        """
        self.sub_seq_length = sub_seq_length
        super().__init__(sequence_length, feature_dim, output_dim, cnn_filters, 
                         kernel_sizes, pool_sizes, lstm_units, dense_units, 
                         dropout_rate, learning_rate)
    
    def _build_model(self) -> tf.keras.Model:
        """
        계층적 CNN-LSTM 모델 구축
        
        반환값:
            tf.keras.Model: 구축된 모델
        """
        # 입력 레이어 - (batch_size, n_sub_seqs, sub_seq_length, feature_dim)
        # 여기서 n_sub_seqs = sequence_length // sub_seq_length
        n_sub_seqs = self.sequence_length // self.sub_seq_length
        inputs = Input(shape=(n_sub_seqs, self.sub_seq_length, self.feature_dim))
        
        # 각 하위 시퀀스에 CNN 적용
        cnn_model = self._build_cnn_model()
        encoded_subsequences = TimeDistributed(cnn_model)(inputs)
        
        # LSTM으로 시간적 관계 모델링
        for i in range(len(self.lstm_units)):
            return_sequences = i < len(self.lstm_units) - 1
            encoded_subsequences = LSTM(self.lstm_units[i], return_sequences=return_sequences)(encoded_subsequences)
            encoded_subsequences = Dropout(self.dropout_rate)(encoded_subsequences)
        
        # 완전 연결 레이어
        x = encoded_subsequences
        for units in self.dense_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
        
        # 출력 레이어
        outputs = Dense(self.output_dim)(x)
        
        # 모델 생성
        model = Model(inputs, outputs)
        
        # 컴파일
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['mae']
        )
        
        return model
    
    def _build_cnn_model(self) -> tf.keras.Model:
        """
        하위 시퀀스용 CNN 모델 구축
        
        반환값:
            tf.keras.Model: CNN 모델
        """
        # 하위 시퀀스 입력
        inputs = Input(shape=(self.sub_seq_length, self.feature_dim))
        
        # CNN 레이어 스택
        x = inputs
        for i in range(len(self.cnn_filters)):
            x = Conv1D(filters=self.cnn_filters[i], 
                      kernel_size=self.kernel_sizes[i],
                      activation='relu',
                      padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=self.pool_sizes[i])(x)
        
        # 특성 벡터로 평탄화
        outputs = Flatten()(x)
        
        return Model(inputs, outputs)
    
    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """
        데이터를 계층적 구조로 전처리
        
        매개변수:
            X (np.ndarray): 원본 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            
        반환값:
            np.ndarray: 재구성된 데이터, 형태: (샘플 수, n_sub_seqs, sub_seq_length, 특성 차원)
        """
        n_samples = X.shape[0]
        n_sub_seqs = self.sequence_length // self.sub_seq_length
        
        # 데이터 재구성
        X_reshaped = np.zeros((n_samples, n_sub_seqs, self.sub_seq_length, self.feature_dim))
        
        for i in range(n_samples):
            for j in range(n_sub_seqs):
                start_idx = j * self.sub_seq_length
                end_idx = start_idx + self.sub_seq_length
                X_reshaped[i, j] = X[i, start_idx:end_idx]
        
        return X_reshaped
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             epochs: int = 100, batch_size: int = 32, 
             validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
             verbose: int = 1,
             callbacks: Optional[List[tf.keras.callbacks.Callback]] = None) -> tf.keras.callbacks.History:
        """
        모델 학습
        
        매개변수:
            X_train (np.ndarray): 학습 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            y_train (np.ndarray): 학습 타겟 데이터, 형태: (샘플 수, 출력 차원)
            epochs (int): 학습 에폭 수
            batch_size (int): 배치 크기
            validation_data (Optional[Tuple[np.ndarray, np.ndarray]]): 검증 데이터
            verbose (int): 출력 상세도
            callbacks (Optional[List[tf.keras.callbacks.Callback]]): 콜백 함수 리스트
            
        반환값:
            tf.keras.callbacks.History: 학습 히스토리
        """
        # 데이터 전처리
        X_train_reshaped = self.preprocess_data(X_train)
        
        # 검증 데이터 전처리
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_reshaped = self.preprocess_data(X_val)
            validation_data = (X_val_reshaped, y_val)
        
        return self.model.fit(
            X_train_reshaped, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=verbose,
            callbacks=callbacks
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        시퀀스 예측
        
        매개변수:
            X (np.ndarray): 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            
        반환값:
            np.ndarray: 예측 결과
        """
        # 데이터 전처리
        X_reshaped = self.preprocess_data(X)
        
        return self.model.predict(X_reshaped)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """
        모델 평가
        
        매개변수:
            X_test (np.ndarray): 테스트 입력 데이터
            y_test (np.ndarray): 테스트 타겟 데이터
            
        반환값:
            Tuple[float, float]: (손실값, MAE)
        """
        # 데이터 전처리
        X_test_reshaped = self.preprocess_data(X_test)
        
        return self.model.evaluate(X_test_reshaped, y_test)


# 예시 사용법
if __name__ == "__main__":
    # 더미 데이터 생성
    sequence_length = 100
    feature_dim = 10
    n_samples = 1000
    output_dim = 1
    
    X = np.random.randn(n_samples, sequence_length, feature_dim)
    y = np.random.randn(n_samples, output_dim)
    
    # 데이터 분할
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 모델 초기화
    model = CNNLSTM(
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        output_dim=output_dim,
        cnn_filters=[32, 64],
        kernel_sizes=[3, 3],
        pool_sizes=[2, 2],
        lstm_units=[64, 32],
        dense_units=[32],
        dropout_rate=0.3
    )
    
    # 모델 요약
    model.model.summary()
    
    # 모델 학습 (짧은 에폭으로 테스트)
    history = model.train(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_data=(X_test, y_test)
    )
    
    # 평가
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, MAE: {mae:.4f}")
    
    # 예측
    predictions = model.predict(X_test[:5])
    print("Predictions shape:", predictions.shape)