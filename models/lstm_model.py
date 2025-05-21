"""
LSTM 기반 신경망 모델

이 모듈은 시계열 신경 신호 예측을 위한 LSTM 기반 신경망 모델을 구현합니다.
시퀀스 데이터 분석, 패턴 인식, 이상 감지 등에 활용할 수 있습니다.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Bidirectional
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("LSTM 모델 사용을 위해 TensorFlow가 필요합니다.")

class LSTMModel:
    """LSTM 기반 시계열 신호 예측 모델"""
    
    def __init__(self, sequence_length: int, feature_dim: int,
                 output_dim: int = 1, lstm_units: int = 64,
                 dropout_rate: float = 0.2, learning_rate: float = 0.001):
        """
        LSTMModel 초기화
        
        매개변수:
            sequence_length (int): 입력 시퀀스 길이
            feature_dim (int): 각 시점당 특성 차원 수
            output_dim (int): 출력 차원 수
            lstm_units (int): LSTM 유닛 수
            dropout_rate (float): 드롭아웃 비율
            learning_rate (float): 학습률
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """
        LSTM 모델 구축
        
        반환값:
            tf.keras.Model: 구축된 모델
        """
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, 
                input_shape=(self.sequence_length, self.feature_dim)),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units // 2),
            Dropout(self.dropout_rate),
            Dense(self.output_dim)
        ])
        
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
        self.model = tf.keras.models.load_model(path)

class BidirectionalLSTMModel(LSTMModel):
    """양방향 LSTM 기반 시계열 신호 예측 모델"""
    
    def _build_model(self) -> tf.keras.Model:
        """
        양방향 LSTM 모델 구축
        
        반환값:
            tf.keras.Model: 구축된 모델
        """
        model = Sequential([
            Bidirectional(LSTM(self.lstm_units, return_sequences=True),
                         input_shape=(self.sequence_length, self.feature_dim)),
            Dropout(self.dropout_rate),
            Bidirectional(LSTM(self.lstm_units // 2)),
            Dropout(self.dropout_rate),
            Dense(self.output_dim)
        ])
        
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['mae']
        )
        
        return model

class EncoderDecoderLSTM:
    """인코더-디코더 LSTM 모델 (시퀀스-투-시퀀스)"""
    
    def __init__(self, input_seq_length: int, output_seq_length: int, feature_dim: int,
                 latent_dim: int = 64, dropout_rate: float = 0.2, learning_rate: float = 0.001):
        """
        EncoderDecoderLSTM 초기화
        
        매개변수:
            input_seq_length (int): 입력 시퀀스 길이
            output_seq_length (int): 출력 시퀀스 길이
            feature_dim (int): 각 시점당 특성 차원 수
            latent_dim (int): 잠재 공간 차원 수
            dropout_rate (float): 드롭아웃 비율
            learning_rate (float): 학습률
        """
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """
        인코더-디코더 LSTM 모델 구축
        
        반환값:
            tf.keras.Model: 구축된 모델
        """
        # 인코더
        encoder_inputs = Input(shape=(self.input_seq_length, self.feature_dim))
        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        # 디코더
        decoder_inputs = Input(shape=(self.output_seq_length, self.feature_dim))
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_outputs = Dropout(self.dropout_rate)(decoder_outputs)
        decoder_dense = Dense(self.feature_dim)
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # 전체 모델
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['mae']
        )
        
        # 추론 모델 (인코더)
        self.encoder_model = Model(encoder_inputs, encoder_states)
        
        # 추론 모델 (디코더)
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        
        return model
    
    def train(self, encoder_input_data: np.ndarray, decoder_input_data: np.ndarray, 
             decoder_target_data: np.ndarray, epochs: int = 100, batch_size: int = 32, 
             validation_split: float = 0.2, verbose: int = 1,
             callbacks: Optional[List[tf.keras.callbacks.Callback]] = None) -> tf.keras.callbacks.History:
        """
        모델 학습
        
        매개변수:
            encoder_input_data (np.ndarray): 인코더 입력 데이터
            decoder_input_data (np.ndarray): 디코더 입력 데이터
            decoder_target_data (np.ndarray): 디코더 타겟 데이터
            epochs (int): 학습 에폭 수
            batch_size (int): 배치 크기
            validation_split (float): 검증 데이터 비율
            verbose (int): 출력 상세도
            callbacks (Optional[List[tf.keras.callbacks.Callback]]): 콜백 함수 리스트
            
        반환값:
            tf.keras.callbacks.History: 학습 히스토리
        """
        return self.model.fit(
            [encoder_input_data, decoder_input_data], decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=callbacks
        )
    
    def predict_sequence(self, input_seq: np.ndarray, target_seq_length: int) -> np.ndarray:
        """
        시퀀스 예측
        
        매개변수:
            input_seq (np.ndarray): 입력 시퀀스, 형태: (1, input_seq_length, feature_dim)
            target_seq_length (int): 예측할 시퀀스 길이
            
        반환값:
            np.ndarray: 예측 시퀀스
        """
        # 인코더로 상태 얻기
        states_value = self.encoder_model.predict(input_seq, verbose=0)
        
        # 디코더 입력 시퀀스 생성 (첫 타임스텝은 0으로 초기화)
        target_seq = np.zeros((1, 1, self.feature_dim))
        
        # 전체 시퀀스 예측
        decoded_seq = np.zeros((1, target_seq_length, self.feature_dim))
        
        for t in range(target_seq_length):
            # 디코더를 통한 예측 및 상태 업데이트
            outputs_and_states = self.decoder_model.predict(
                [target_seq] + states_value, verbose=0)
            output = outputs_and_states[0]
            states_value = outputs_and_states[1:]
            
            # 예측 결과 저장
            decoded_seq[0, t] = output[0, 0]
            
            # 다음 타임스텝 입력 업데이트
            target_seq = output
            
        return decoded_seq
    
    def save(self, path: str) -> None:
        """
        모델 저장
        
        매개변수:
            path (str): 저장 경로
        """
        self.model.save(path)
        self.encoder_model.save(path + "_encoder")
        self.decoder_model.save(path + "_decoder")
        
    def load(self, path: str) -> None:
        """
        모델 로드
        
        매개변수:
            path (str): 모델 파일 경로
        """
        self.model = tf.keras.models.load_model(path)
        self.encoder_model = tf.keras.models.load_model(path + "_encoder")
        self.decoder_model = tf.keras.models.load_model(path + "_decoder")
