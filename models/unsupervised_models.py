"""비지도 학습 모델

이 모듈은 신경 신호 분석을 위한 비지도 학습 모델을 구현합니다.
VAE(Variational Autoencoder), K-means 클러스터링, PCA(Principal Component Analysis) 등
다양한 비지도 학습 알고리즘을 통해 데이터의 패턴과 구조를 파악합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Dense, Input, Lambda, Flatten, Reshape,
                                        Conv1D, Conv1DTranspose, BatchNormalization,
                                        Dropout, MaxPooling1D, UpSampling1D)
    from tensorflow.keras.losses import mse
    from tensorflow.keras import backend as K
except ImportError:
    print("VAE 모델 사용을 위해 TensorFlow가 필요합니다.")


class VariationalAutoencoder:
    """변이형 오토인코더(VAE) 모델 클래스
    
    시계열 데이터의 잠재 표현을 학습하고 특성 추출 및 이상 탐지에 활용할 수 있습니다.
    """
    
    def __init__(self, 
                 sequence_length: int, 
                 feature_dim: int,
                 latent_dim: int = 16, 
                 encoder_units: List[int] = [128, 64], 
                 decoder_units: List[int] = [64, 128],
                 dropout_rate: float = 0.2,
                 kl_weight: float = 0.001,
                 learning_rate: float = 0.001):
        """
        VAE 모델 초기화
        
        매개변수:
            sequence_length (int): 입력 시퀀스 길이
            feature_dim (int): 각 시점당 특성 차원 수
            latent_dim (int): 잠재 공간 차원
            encoder_units (List[int]): 인코더 레이어의 유닛 수 리스트
            decoder_units (List[int]): 디코더 레이어의 유닛 수 리스트
            dropout_rate (float): 드롭아웃 비율
            kl_weight (float): KL 발산 가중치
            learning_rate (float): 학습률
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.dropout_rate = dropout_rate
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate
        
        # 평탄화된 입력 차원 계산
        self.flattened_dim = sequence_length * feature_dim
        
        # 모델 빌드
        self._build_model()
    
    def _sampling(self, args):
        """
        재매개변수화 트릭을 사용한 잠재 벡터 샘플링
        
        매개변수:
            args: [평균, 로그 분산] 형태의 텐서 쌍
            
        반환값:
            샘플링된 잠재 벡터
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    def _build_model(self):
        """VAE 모델 구축"""
        # 인코더 네트워크
        encoder_inputs = Input(shape=(self.sequence_length, self.feature_dim), name='encoder_input')
        x = Flatten()(encoder_inputs)
        
        # 인코더 레이어
        for units in self.encoder_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
        
        # 잠재 공간 파라미터
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        
        # 재매개변수화를 통한 잠재 벡터 샘플링
        z = Lambda(self._sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        
        # 인코더 모델 정의
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        
        # 디코더 네트워크
        decoder_inputs = Input(shape=(self.latent_dim,), name='decoder_input')
        x = decoder_inputs
        
        # 디코더 레이어
        for units in self.decoder_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
        
        decoder_outputs = Dense(self.flattened_dim, activation='linear')(x)
        decoder_outputs = Reshape((self.sequence_length, self.feature_dim))(decoder_outputs)
        
        # 디코더 모델 정의
        self.decoder = Model(decoder_inputs, decoder_outputs, name='decoder')
        
        # VAE 모델 정의
        outputs = self.decoder(self.encoder(encoder_inputs)[2])
        self.vae = Model(encoder_inputs, outputs, name='vae')
        
        # VAE 손실 함수 정의
        reconstruction_loss = mse(K.flatten(encoder_inputs), K.flatten(outputs))
        reconstruction_loss *= self.flattened_dim
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(reconstruction_loss + self.kl_weight * kl_loss)
        
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
    
    def train(self, X_train: np.ndarray, 
             epochs: int = 100, batch_size: int = 32, 
             validation_data: Optional[np.ndarray] = None,
             verbose: int = 1,
             callbacks: Optional[List[tf.keras.callbacks.Callback]] = None) -> tf.keras.callbacks.History:
        """
        모델 학습
        
        매개변수:
            X_train (np.ndarray): 학습 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            epochs (int): 학습 에폭 수
            batch_size (int): 배치 크기
            validation_data (Optional[np.ndarray]): 검증 데이터
            verbose (int): 출력 상세도
            callbacks (Optional[List[tf.keras.callbacks.Callback]]): 콜백 함수 리스트
            
        반환값:
            tf.keras.callbacks.History: 학습 히스토리
        """
        return self.vae.fit(
            X_train, None,  # 타겟 데이터는 필요하지 않음
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(validation_data, None) if validation_data is not None else None,
            verbose=verbose,
            callbacks=callbacks
        )
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        입력 데이터를 잠재 공간으로 인코딩
        
        매개변수:
            x (np.ndarray): 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            
        반환값:
            np.ndarray: 잠재 벡터, 형태: (샘플 수, 잠재 차원)
        """
        z_mean, _, _ = self.encoder.predict(x)
        return z_mean
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        잠재 벡터를 원본 공간으로 디코딩
        
        매개변수:
            z (np.ndarray): 잠재 벡터, 형태: (샘플 수, 잠재 차원)
            
        반환값:
            np.ndarray: 재구성된 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
        """
        return self.decoder.predict(z)
    
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """
        입력 데이터 재구성
        
        매개변수:
            x (np.ndarray): 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            
        반환값:
            np.ndarray: 재구성된 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
        """
        return self.vae.predict(x)
    
    def compute_reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """
        재구성 오차 계산 (이상 탐지용)
        
        매개변수:
            x (np.ndarray): 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            
        반환값:
            np.ndarray: 각 샘플별 재구성 오차, 형태: (샘플 수,)
        """
        reconstructed = self.reconstruct(x)
        # MSE 계산
        mse_error = np.mean(np.square(x - reconstructed), axis=(1, 2))
        return mse_error
    
    def detect_anomalies(self, x: np.ndarray, threshold: Optional[float] = None, 
                       percentile: float = 95) -> Tuple[np.ndarray, float]:
        """
        이상치 탐지
        
        매개변수:
            x (np.ndarray): 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            threshold (Optional[float]): 이상치 판단 임계값
            percentile (float): 임계값 설정용 백분위수
            
        반환값:
            Tuple[np.ndarray, float]: (이상치 여부 배열, 사용된 임계값)
        """
        errors = self.compute_reconstruction_error(x)
        
        if threshold is None:
            # 지정된 백분위수에 해당하는 오차값을 임계값으로 설정
            threshold = np.percentile(errors, percentile)
        
        # 임계값을 초과하는 샘플을 이상치로 표시
        anomalies = errors > threshold
        return anomalies, threshold
    
    def save(self, encoder_path: str, decoder_path: str, vae_path: str):
        """
        모델 저장
        
        매개변수:
            encoder_path (str): 인코더 모델 저장 경로
            decoder_path (str): 디코더 모델 저장 경로
            vae_path (str): VAE 모델 저장 경로
        """
        self.encoder.save(encoder_path)
        self.decoder.save(decoder_path)
        self.vae.save(vae_path)
    
    def load(self, encoder_path: str, decoder_path: str, vae_path: str):
        """
        모델 로드
        
        매개변수:
            encoder_path (str): 인코더 모델 로드 경로
            decoder_path (str): 디코더 모델 로드 경로
            vae_path (str): VAE 모델 로드 경로
        """
        self.encoder = tf.keras.models.load_model(encoder_path)
        self.decoder = tf.keras.models.load_model(decoder_path)
        self.vae = tf.keras.models.load_model(vae_path)