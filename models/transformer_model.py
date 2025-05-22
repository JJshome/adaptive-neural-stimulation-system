"""Transformer 기반 모델

이 모듈은 시퀀스 데이터 분석을 위한 Transformer 기반 모델을 구현합니다.
셀프 어텐션 메커니즘을 활용하여 장거리 의존성을 효과적으로 포착합니다.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Dense, Dropout, Input, LayerNormalization,
                                        MultiHeadAttention, GlobalAveragePooling1D,
                                        Embedding, Conv1D, BatchNormalization,
                                        Concatenate)
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("Transformer 모델 사용을 위해 TensorFlow가 필요합니다.")


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer 블록 구현"""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1):
        """
        Transformer 블록 초기화
        
        매개변수:
            embed_dim (int): 임베딩 차원
            num_heads (int): 어텐션 헤드 수
            ff_dim (int): 피드포워드 네트워크 차원
            rate (float): 드롭아웃 비율
        """
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        
    def call(self, inputs, training=False):
        """
        순방향 전파
        
        매개변수:
            inputs: 입력 텐서
            training: 학습 모드 여부
            
        반환값:
            텐서: 변환된 출력
        """
        # 멀티헤드 어텐션 레이어
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # 피드포워드 네트워크
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEncoding(tf.keras.layers.Layer):
    """위치 인코딩 레이어"""
    
    def __init__(self, position: int, d_model: int):
        """
        위치 인코딩 초기화
        
        매개변수:
            position (int): 최대 시퀀스 길이
            d_model (int): 모델 차원
        """
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
        
    def get_angles(self, position, i, d_model):
        """
        각도 계산
        
        매개변수:
            position: 위치
            i: 차원 인덱스
            d_model: 모델 차원
            
        반환값:
            계산된 각도
        """
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angle_rates
    
    def positional_encoding(self, position, d_model):
        """
        위치 인코딩 계산
        
        매개변수:
            position: 최대 위치
            d_model: 모델 차원
            
        반환값:
            위치 인코딩 텐서
        """
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        # 짝수 인덱스에는 sin 적용
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # 홀수 인덱스에는 cos 적용
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        """
        순방향 전파
        
        매개변수:
            inputs: 입력 텐서
            
        반환값:
            위치 인코딩이 적용된 텐서
        """
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class TransformerModel:
    """Transformer 기반 시계열 신호 예측 모델"""
    
    def __init__(self, 
                 sequence_length: int, 
                 feature_dim: int,
                 output_dim: int = 1, 
                 num_transformer_blocks: int = 2,
                 num_heads: int = 8,
                 embed_dim: int = 64,
                 ff_dim: int = 256,
                 dense_units: List[int] = [64, 32],
                 dropout_rate: float = 0.1,
                 learning_rate: float = 0.001):
        """
        Transformer 모델 초기화
        
        매개변수:
            sequence_length (int): 입력 시퀀스 길이
            feature_dim (int): 각 시점당 특성 차원 수
            output_dim (int): 출력 차원 수
            num_transformer_blocks (int): Transformer 블록 수
            num_heads (int): 어텐션 헤드 수
            embed_dim (int): 임베딩 차원
            ff_dim (int): 피드포워드 네트워크 차원
            dense_units (List[int]): 완전 연결 레이어 유닛 수 리스트
            dropout_rate (float): 드롭아웃 비율
            learning_rate (float): 학습률
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # 모델 빌드
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """
        Transformer 모델 구축
        
        반환값:
            tf.keras.Model: 구축된 모델
        """
        # 입력 레이어
        inputs = Input(shape=(self.sequence_length, self.feature_dim))
        
        # 특성 차원이 임베딩 차원과 다르면 선형 투영
        if self.feature_dim != self.embed_dim:
            x = Dense(self.embed_dim)(inputs)
        else:
            x = inputs
        
        # 위치 인코딩 추가
        x = PositionalEncoding(self.sequence_length, self.embed_dim)(x)
        
        # Transformer 블록 스택
        for _ in range(self.num_transformer_blocks):
            x = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate)(x)
        
        # 시퀀스 차원 처리 (평균 풀링)
        x = GlobalAveragePooling1D()(x)
        
        # 완전 연결 레이어 스택
        for units in self.dense_units:
            x = Dense(units, activation="relu")(x)
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


class ConvTransformerModel(TransformerModel):
    """CNN-Transformer 하이브리드 모델
    
    CNN 레이어를 통해 로컬 특성을 추출하고 Transformer로 장거리 의존성을 모델링합니다.
    """
    
    def __init__(self, 
                 sequence_length: int, 
                 feature_dim: int,
                 output_dim: int = 1, 
                 num_transformer_blocks: int = 2,
                 num_heads: int = 8,
                 embed_dim: int = 64,
                 ff_dim: int = 256,
                 cnn_filters: List[int] = [32, 64],
                 kernel_sizes: List[int] = [3, 3],
                 dense_units: List[int] = [64, 32],
                 dropout_rate: float = 0.1,
                 learning_rate: float = 0.001):
        """
        CNN-Transformer 모델 초기화
        
        매개변수:
            sequence_length (int): 입력 시퀀스 길이
            feature_dim (int): 각 시점당 특성 차원 수
            output_dim (int): 출력 차원 수
            num_transformer_blocks (int): Transformer 블록 수
            num_heads (int): 어텐션 헤드 수
            embed_dim (int): 임베딩 차원
            ff_dim (int): 피드포워드 네트워크 차원
            cnn_filters (List[int]): CNN 레이어 필터 수 리스트
            kernel_sizes (List[int]): CNN 레이어 커널 크기 리스트
            dense_units (List[int]): 완전 연결 레이어 유닛 수 리스트
            dropout_rate (float): 드롭아웃 비율
            learning_rate (float): 학습률
        """
        self.cnn_filters = cnn_filters
        self.kernel_sizes = kernel_sizes
        super().__init__(sequence_length, feature_dim, output_dim, 
                         num_transformer_blocks, num_heads, embed_dim, 
                         ff_dim, dense_units, dropout_rate, learning_rate)
    
    def _build_model(self) -> tf.keras.Model:
        """
        CNN-Transformer 모델 구축
        
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
        
        # 특성 차원이 임베딩 차원과 다르면 선형 투영
        if self.cnn_filters[-1] != self.embed_dim:
            x = Dense(self.embed_dim)(x)
        
        # 위치 인코딩 추가
        x = PositionalEncoding(self.sequence_length, self.embed_dim)(x)
        
        # Transformer 블록 스택
        for _ in range(self.num_transformer_blocks):
            x = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate)(x)
        
        # 시퀀스 차원 처리 (평균 풀링)
        x = GlobalAveragePooling1D()(x)
        
        # 완전 연결 레이어 스택
        for units in self.dense_units:
            x = Dense(units, activation="relu")(x)
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
    model = TransformerModel(
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        output_dim=output_dim,
        num_transformer_blocks=2,
        num_heads=4,
        embed_dim=64,
        ff_dim=128,
        dense_units=[32],
        dropout_rate=0.1
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