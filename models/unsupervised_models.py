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


class ConvolutionalVAE(VariationalAutoencoder):
    """컨볼루션 기반 변이형 오토인코더(VAE) 모델 클래스
    
    1D 컨볼루션 레이어를 사용하여 시계열 데이터의 로컬 패턴을 효과적으로 포착합니다.
    """
    
    def __init__(self, 
                 sequence_length: int, 
                 feature_dim: int,
                 latent_dim: int = 16, 
                 filters: List[int] = [32, 64, 128], 
                 kernel_sizes: List[int] = [3, 3, 3],
                 dropout_rate: float = 0.2,
                 kl_weight: float = 0.001,
                 learning_rate: float = 0.001):
        """
        컨볼루션 VAE 모델 초기화
        
        매개변수:
            sequence_length (int): 입력 시퀀스 길이
            feature_dim (int): 각 시점당 특성 차원 수
            latent_dim (int): 잠재 공간 차원
            filters (List[int]): 컨볼루션 레이어의 필터 수 리스트
            kernel_sizes (List[int]): 컨볼루션 레이어의 커널 크기 리스트
            dropout_rate (float): 드롭아웃 비율
            kl_weight (float): KL 발산 가중치
            learning_rate (float): 학습률
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate
        
        # 인코딩 과정에서의 다운샘플링 계산
        self.downsampled_length = sequence_length
        for _ in range(len(filters)):
            self.downsampled_length = self.downsampled_length // 2
        
        self.conv_flattened_dim = self.downsampled_length * filters[-1]
        
        # 모델 빌드
        self._build_model()
    
    def _build_model(self):
        """컨볼루션 VAE 모델 구축"""
        # 인코더 네트워크
        encoder_inputs = Input(shape=(self.sequence_length, self.feature_dim), name='encoder_input')
        x = encoder_inputs
        
        # 컨볼루션 인코더 레이어
        for i in range(len(self.filters)):
            x = Conv1D(filters=self.filters[i], 
                      kernel_size=self.kernel_sizes[i],
                      activation='relu',
                      padding='same',
                      strides=1)(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(self.dropout_rate)(x)
        
        # 평탄화
        x = Flatten()(x)
        
        # 잠재 공간 파라미터
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        
        # 재매개변수화를 통한 잠재 벡터 샘플링
        z = Lambda(self._sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        
        # 인코더 모델 정의
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        
        # 디코더 네트워크
        decoder_inputs = Input(shape=(self.latent_dim,), name='decoder_input')
        
        # 잠재 벡터를 컨볼루션 특성맵 차원으로 확장
        x = Dense(self.conv_flattened_dim, activation='relu')(decoder_inputs)
        x = Reshape((self.downsampled_length, self.filters[-1]))(x)
        
        # 디컨볼루션(전치 컨볼루션) 레이어
        for i in range(len(self.filters)-1, -1, -1):
            x = UpSampling1D(size=2)(x)
            x = Conv1DTranspose(
                filters=self.filters[i] if i > 0 else self.feature_dim,
                kernel_size=self.kernel_sizes[i],
                activation='relu' if i > 0 else 'linear',
                padding='same'
            )(x)
            if i > 0:  # 마지막 레이어에는 배치 정규화 적용 안함
                x = BatchNormalization()(x)
                x = Dropout(self.dropout_rate)(x)
        
        decoder_outputs = x
        
        # 디코더 모델 정의
        self.decoder = Model(decoder_inputs, decoder_outputs, name='decoder')
        
        # VAE 모델 정의
        outputs = self.decoder(self.encoder(encoder_inputs)[2])
        self.vae = Model(encoder_inputs, outputs, name='vae')
        
        # VAE 손실 함수 정의
        reconstruction_loss = mse(K.flatten(encoder_inputs), K.flatten(outputs))
        reconstruction_loss *= self.sequence_length * self.feature_dim
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(reconstruction_loss + self.kl_weight * kl_loss)
        
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))


class TimeSeriesKMeans:
    """시계열 데이터를 위한 K-means 클러스터링 클래스
    
    신경 신호 패턴 그룹화 및 클러스터 분석을 위한 K-means 클러스터링 구현입니다.
    """
    
    def __init__(self, 
                 n_clusters: int = 3, 
                 max_iter: int = 300,
                 random_state: int = 42,
                 flatten: bool = True):
        """
        K-means 클러스터링 초기화
        
        매개변수:
            n_clusters (int): 클러스터 수
            max_iter (int): 최대 반복 횟수
            random_state (int): 랜덤 시드
            flatten (bool): 시계열 데이터 평탄화 여부
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.flatten = flatten
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=random_state,
            n_init=10
        )
        self.scaler = StandardScaler()
        
    def preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        시계열 데이터 전처리
        
        매개변수:
            X (np.ndarray): 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            
        반환값:
            np.ndarray: 전처리된 데이터
        """
        if self.flatten:
            # 시계열 데이터 평탄화
            X_flat = X.reshape(X.shape[0], -1)
        else:
            # 시계열 특성 추출 (평균, 분산, 등)
            X_flat = np.concatenate([
                np.mean(X, axis=1),
                np.std(X, axis=1),
                np.max(X, axis=1),
                np.min(X, axis=1),
                np.median(X, axis=1)
            ], axis=1)
        
        # 표준화
        X_scaled = self.scaler.fit_transform(X_flat)
        return X_scaled
    
    def fit(self, X: np.ndarray) -> 'TimeSeriesKMeans':
        """
        K-means 클러스터링 학습
        
        매개변수:
            X (np.ndarray): 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            
        반환값:
            TimeSeriesKMeans: 학습된 모델 인스턴스
        """
        X_scaled = self.preprocess(X)
        self.kmeans.fit(X_scaled)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        클러스터 레이블 예측
        
        매개변수:
            X (np.ndarray): 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            
        반환값:
            np.ndarray: 예측된 클러스터 레이블, 형태: (샘플 수,)
        """
        X_scaled = self.preprocess(X)
        return self.kmeans.predict(X_scaled)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        학습 후 클러스터 레이블 예측
        
        매개변수:
            X (np.ndarray): 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            
        반환값:
            np.ndarray: 예측된 클러스터 레이블, 형태: (샘플 수,)
        """
        self.fit(X)
        return self.predict(X)
    
    def get_cluster_centers(self, original_shape: bool = False) -> np.ndarray:
        """
        클러스터 중심점 반환
        
        매개변수:
            original_shape (bool): 원본 시계열 형태로 반환 여부
            
        반환값:
            np.ndarray: 클러스터 중심점
        """
        centers = self.kmeans.cluster_centers_
        
        if original_shape and self.flatten:
            # 원본 시계열 형태로 복원 (샘플 수, 시퀀스 길이, 특성 차원)
            original_dim = centers.shape[1]
            seq_length = original_dim // self.feature_dim if hasattr(self, 'feature_dim') else None
            
            if seq_length is not None:
                centers = centers.reshape(self.n_clusters, seq_length, self.feature_dim)
        
        return centers
    
    def evaluate(self, X: np.ndarray) -> Dict[str, float]:
        """
        클러스터링 품질 평가
        
        매개변수:
            X (np.ndarray): 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            
        반환값:
            Dict[str, float]: 평가 지표 딕셔너리
        """
        X_scaled = self.preprocess(X)
        labels = self.kmeans.predict(X_scaled)
        
        # 이너셔 (클러스터 내 거리 제곱합)
        inertia = self.kmeans.inertia_
        
        # 실루엣 점수
        silhouette = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0
        
        return {
            "inertia": inertia,
            "silhouette_score": silhouette
        }
    
    def find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10) -> Dict[str, List]:
        """
        최적의 클러스터 수 탐색 (엘보우 방법)
        
        매개변수:
            X (np.ndarray): 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            max_clusters (int): 탐색할 최대 클러스터 수
            
        반환값:
            Dict[str, List]: 각 클러스터 수에 대한 평가 지표
        """
        X_scaled = self.preprocess(X)
        
        inertias = []
        silhouettes = []
        cluster_range = range(2, max_clusters + 1)
        
        for n_clusters in cluster_range:
            kmeans = KMeans(
                n_clusters=n_clusters,
                max_iter=self.max_iter,
                random_state=self.random_state,
                n_init=10
            )
            
            labels = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)
            
            # 실루엣 점수 계산 (클러스터가 2개 이상인 경우에만)
            if len(np.unique(labels)) > 1:
                silhouettes.append(silhouette_score(X_scaled, labels))
            else:
                silhouettes.append(0)
        
        return {
            "n_clusters": list(cluster_range),
            "inertia": inertias,
            "silhouette_score": silhouettes
        }
    
    def plot_clusters(self, X: np.ndarray, labels: Optional[np.ndarray] = None, 
                    n_components: int = 2, figsize: Tuple[int, int] = (10, 8)):
        """
        클러스터 시각화 (PCA 사용)
        
        매개변수:
            X (np.ndarray): 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            labels (Optional[np.ndarray]): 클러스터 레이블
            n_components (int): PCA 차원 수
            figsize (Tuple[int, int]): 그림 크기
        """
        X_scaled = self.preprocess(X)
        
        if labels is None:
            labels = self.kmeans.predict(X_scaled)
        
        # PCA로 차원 축소
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # 클러스터 중심점 변환
        centers_pca = pca.transform(self.kmeans.cluster_centers_)
        
        # 시각화
        plt.figure(figsize=figsize)
        
        # 산점도
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
        
        # 클러스터 중심점
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200, alpha=1)
        
        plt.title(f'K-means Clustering (k={self.n_clusters})')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(label='Cluster')
        plt.grid(alpha=0.3)
        plt.show()


class TimeSeriesPCA:
    """시계열 데이터를 위한 PCA 클래스
    
    주성분 분석을 통한 차원 축소 및 특성 추출을 구현합니다.
    """
    
    def __init__(self, 
                 n_components: int = 2, 
                 flatten: bool = True,
                 standardize: bool = True):
        """
        PCA 초기화
        
        매개변수:
            n_components (int): 주성분 수
            flatten (bool): 시계열 데이터 평탄화 여부
            standardize (bool): 표준화 여부
        """
        self.n_components = n_components
        self.flatten = flatten
        self.standardize = standardize
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        
    def preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        시계열 데이터 전처리
        
        매개변수:
            X (np.ndarray): 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            
        반환값:
            np.ndarray: 전처리된 데이터
        """
        if self.flatten:
            # 시계열 데이터 평탄화
            X_flat = X.reshape(X.shape[0], -1)
        else:
            # 시계열 특성 추출 (평균, 분산, 등)
            X_flat = np.concatenate([
                np.mean(X, axis=1),
                np.std(X, axis=1),
                np.max(X, axis=1),
                np.min(X, axis=1)
            ], axis=1)
        
        # 표준화
        if self.standardize:
            X_flat = self.scaler.fit_transform(X_flat)
            
        return X_flat
    
    def fit(self, X: np.ndarray) -> 'TimeSeriesPCA':
        """
        PCA 학습
        
        매개변수:
            X (np.ndarray): 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            
        반환값:
            TimeSeriesPCA: 학습된 모델 인스턴스
        """
        X_preprocessed = self.preprocess(X)
        self.pca.fit(X_preprocessed)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        PCA 변환
        
        매개변수:
            X (np.ndarray): 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            
        반환값:
            np.ndarray: 주성분 공간으로 변환된 데이터, 형태: (샘플 수, n_components)
        """
        X_preprocessed = self.preprocess(X)
        return self.pca.transform(X_preprocessed)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        학습 후 PCA 변환
        
        매개변수:
            X (np.ndarray): 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            
        반환값:
            np.ndarray: 주성분 공간으로 변환된 데이터, 형태: (샘플 수, n_components)
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """
        PCA 역변환
        
        매개변수:
            X_pca (np.ndarray): 주성분 공간의 데이터, 형태: (샘플 수, n_components)
            
        반환값:
            np.ndarray: 역변환된 데이터
        """
        X_reconstructed = self.pca.inverse_transform(X_pca)
        
        # 표준화를 적용한 경우 역변환
        if self.standardize:
            X_reconstructed = self.scaler.inverse_transform(X_reconstructed)
        
        return X_reconstructed
    
    def explained_variance(self) -> Dict[str, np.ndarray]:
        """
        설명된 분산 정보 반환
        
        반환값:
            Dict[str, np.ndarray]: 설명된 분산 관련 지표
        """
        return {
            "explained_variance": self.pca.explained_variance_,
            "explained_variance_ratio": self.pca.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(self.pca.explained_variance_ratio_)
        }
    
    def plot_explained_variance(self, figsize: Tuple[int, int] = (10, 6)):
        """
        설명된 분산 시각화
        
        매개변수:
            figsize (Tuple[int, int]): 그림 크기
        """
        variance_info = self.explained_variance()
        
        plt.figure(figsize=figsize)
        
        # 주성분별 설명된 분산 비율
        plt.bar(
            range(1, len(variance_info["explained_variance_ratio"]) + 1),
            variance_info["explained_variance_ratio"],
            alpha=0.7,
            label='Individual Explained Variance'
        )
        
        # 누적 설명된 분산 비율
        plt.step(
            range(1, len(variance_info["cumulative_variance_ratio"]) + 1),
            variance_info["cumulative_variance_ratio"],
            where='mid',
            label='Cumulative Explained Variance',
            color='red'
        )
        
        plt.axhline(y=0.95, color='green', linestyle='--', label='95% Explained Variance')
        
        plt.title('Explained Variance by Principal Components')
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.xticks(range(1, len(variance_info["explained_variance_ratio"]) + 1))
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    
    def plot_pca_components(self, figsize: Tuple[int, int] = (12, 10)):
        """
        주성분 패턴 시각화
        
        매개변수:
            figsize (Tuple[int, int]): 그림 크기
        """
        components = self.pca.components_
        n_components = min(4, len(components))  # 최대 4개까지 시각화
        
        plt.figure(figsize=figsize)
        
        for i in range(n_components):
            plt.subplot(n_components, 1, i+1)
            plt.plot(components[i], linewidth=2)
            plt.title(f'Principal Component {i+1}')
            plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_pca_scatter(self, X: np.ndarray, labels: Optional[np.ndarray] = None, 
                       figsize: Tuple[int, int] = (10, 8)):
        """
        PCA 산점도 시각화
        
        매개변수:
            X (np.ndarray): 입력 데이터, 형태: (샘플 수, 시퀀스 길이, 특성 차원)
            labels (Optional[np.ndarray]): 데이터 레이블
            figsize (Tuple[int, int]): 그림 크기
        """
        # PCA 변환
        X_pca = self.transform(X)
        
        plt.figure(figsize=figsize)
        
        if labels is not None:
            # 레이블별 색상 구분
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Class')
        else:
            plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
        
        # 주성분 설명된 분산 비율
        variance_info = self.explained_variance()
        
        plt.title('PCA Scatter Plot')
        plt.xlabel(f'PC1 ({variance_info["explained_variance_ratio"][0]:.2%} variance)')
        plt.ylabel(f'PC2 ({variance_info["explained_variance_ratio"][1]:.2%} variance)')
        plt.grid(alpha=0.3)
        plt.show()


# 예시 사용법
if __name__ == "__main__":
    # 더미 데이터 생성
    sequence_length = 100
    feature_dim = 5
    n_samples = 1000
    
    X = np.random.randn(n_samples, sequence_length, feature_dim)
    
    # VAE 예제
    vae = VariationalAutoencoder(
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        latent_dim=16,
        encoder_units=[128, 64],
        decoder_units=[64, 128],
        dropout_rate=0.2
    )
    
    # 간단한 학습 (데모용)
    vae.train(X, epochs=5, batch_size=32)
    
    # 잠재 공간 인코딩 및 재구성
    latent_vectors = vae.encode(X[:10])
    reconstructed = vae.reconstruct(X[:10])
    
    print("Latent vectors shape:", latent_vectors.shape)
    print("Reconstructed data shape:", reconstructed.shape)
    
    # K-means 예제
    kmeans = TimeSeriesKMeans(n_clusters=3, flatten=True)
    labels = kmeans.fit_predict(X)
    
    print("Cluster labels:", np.unique(labels, return_counts=True))
    
    # PCA 예제
    pca = TimeSeriesPCA(n_components=2, flatten=True)
    X_pca = pca.fit_transform(X)
    
    print("PCA transformed shape:", X_pca.shape)
    variance_info = pca.explained_variance()
    print(f"Explained variance ratio: {variance_info['explained_variance_ratio']}")
    print(f"Total explained variance: {np.sum(variance_info['explained_variance_ratio']):.2%}")