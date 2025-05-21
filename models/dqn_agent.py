"""
DQN(Deep Q-Network) 강화학습 모델

이 모듈은 신경 자극 최적화를 위한 DQN 강화학습 모델을 구현합니다.
경험 리플레이, 타겟 네트워크, 엡실론-그리디 정책 등의 DQN 기법을 적용합니다.
"""

import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict, Any, Optional, Union

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("DQN 모델 사용을 위해 TensorFlow가 필요합니다.")

class DQNAgent:
    """DQN 강화학습 기반 자극 최적화 에이전트"""
    
    def __init__(self, state_size: int, action_size: int,
                 memory_size: int = 2000, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, learning_rate: float = 0.001):
        """
        DQNAgent 초기화
        
        매개변수:
            state_size (int): 상태 공간의 차원 수
            action_size (int): 행동 공간의 차원 수
            memory_size (int): 경험 리플레이 메모리 크기
            gamma (float): 할인 계수
            epsilon (float): 엡실론-그리디 탐색 초기 확률
            epsilon_min (float): 엡실론 최소값
            epsilon_decay (float): 엡실론 감소율
            learning_rate (float): 학습률
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # 할인 계수
        self.epsilon = epsilon  # 탐색 확률
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self) -> tf.keras.Model:
        """
        DQN 모델 구축
        
        반환값:
            tf.keras.Model: 구축된 모델
        """
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self) -> None:
        """타겟 모델 가중치를 메인 모델로 업데이트"""
        self.target_model.set_weights(self.model.get_weights())
    
    def memorize(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """
        경험 리플레이 메모리에 트랜지션 저장
        
        매개변수:
            state (np.ndarray): 현재 상태
            action (int): 수행한 행동
            reward (float): 받은 보상
            next_state (np.ndarray): 다음 상태
            done (bool): 에피소드 종료 여부
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """
        현재 상태에서 행동 선택 (엡실론-그리디 정책)
        
        매개변수:
            state (np.ndarray): 현재 상태
            
        반환값:
            int: 선택된 행동
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size: int) -> float:
        """
        경험 리플레이를 통한 학습
        
        매개변수:
            batch_size (int): 배치 크기
            
        반환값:
            float: 학습 손실값
        """
        if len(self.memory) < batch_size:
            return 0.0
            
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
        
        # 배치 처리로 예측
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Q-값 업데이트
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.gamma * np.amax(next_q_values[i])
        
        # 모델 학습
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # 엡실론 감소 (탐색 확률 줄이기)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss
    
    def load(self, name: str) -> None:
        """
        모델 가중치 로드
        
        매개변수:
            name (str): 모델 파일 경로
        """
        self.model.load_weights(name)
        self.update_target_model()
    
    def save(self, name: str) -> None:
        """
        모델 가중치 저장
        
        매개변수:
            name (str): 모델 파일 경로
        """
        self.model.save_weights(name)
        
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        모든 행동에 대한 Q-값 반환
        
        매개변수:
            state (np.ndarray): 현재 상태
            
        반환값:
            np.ndarray: 행동별 Q-값
        """
        return self.model.predict(state, verbose=0)[0]
