"""
DQN(Deep Q-Network) 강화학습 모델

이 모듈은 신경 자극 최적화를 위한 DQN 강화학습 모델을 구현합니다.
경험 리플레이, 타겟 네트워크, 엡실론-그리디 정책 등의 DQN 기법을 적용합니다.

주요 기능:
    - DQN 에이전트 구현: 딥러닝 기반 Q-학습 알고리즘
    - 경험 리플레이: 과거 경험을 저장하고 무작위로 샘플링하여 학습
    - 타겟 네트워크: 학습 안정성을 위한 별도의 타겟 모델 유지
    - 엡실론-그리디 탐색: 탐색과 활용 사이의 균형을 위한 확률적 행동 선택

사용 예시:
    ```python
    import numpy as np
    from models.dqn_agent import DQNAgent
    
    # 상태 공간 및 행동 공간 정의
    state_size = 5  # 예: 5개의 특성을 포함하는 상태
    action_size = 10  # 예: 10개의 가능한 자극 매개변수 조합
    
    # DQN 에이전트 초기화
    agent = DQNAgent(state_size, action_size, memory_size=2000)
    
    # 학습 루프
    for episode in range(100):
        state = np.random.random(state_size).reshape(1, state_size)  # 현재 상태
        total_reward = 0
        
        for step in range(50):
            # 행동 선택
            action = agent.act(state)
            
            # 가상의 환경에서 다음 상태와 보상 얻기
            # (실제로는 환경과 상호작용하여 얻어야 함)
            next_state = np.random.random(state_size).reshape(1, state_size)
            reward = np.random.random()
            done = (step == 49)  # 마지막 스텝에서 에피소드 종료
            
            # 경험 저장 및 학습
            agent.memorize(state[0], action, reward, next_state[0], done)
            
            # 충분한 경험이 쌓이면 배치 학습 수행
            if len(agent.memory) > 32:
                loss = agent.replay(batch_size=32)
                
            # 상태 업데이트
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        # 에피소드 종료 시 타겟 모델 업데이트
        agent.update_target_model()
        
        print(f"에피소드 {episode+1}, 총 보상: {total_reward:.4f}")
    
    # 학습된 모델 저장
    agent.save("dqn_model.h5")
    ```

참고 자료:
    - Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. 
      Nature, 518(7540), 529-533.
    - Barto, A. G., & Sutton, R. S. (2018). Reinforcement learning: An introduction. MIT press.
    - Li, Y. (2017). Deep Reinforcement Learning: An Overview. arXiv preprint arXiv:1701.07274.
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
    """
    DQN 강화학습 기반 자극 최적화 에이전트
    
    Deep Q-Network(DQN) 알고리즘을 구현한 강화학습 에이전트입니다.
    신경 자극 최적화 문제에서 최적의 자극 매개변수를 학습하는 데 사용됩니다.
    
    Attributes:
        state_size (int): 상태 공간의 차원 수
        action_size (int): 행동 공간의 차원 수
        memory (collections.deque): 경험 리플레이 메모리
        gamma (float): 할인 계수, 미래 보상의 중요도를 결정
        epsilon (float): 탐색 확률, 무작위 행동을 선택할 확률
        epsilon_min (float): 최소 탐색 확률, epsilon이 이 값 이하로 감소하지 않음
        epsilon_decay (float): 탐색 확률 감소율, 학습이 진행됨에 따라 epsilon 감소
        learning_rate (float): 학습률, 모델 가중치 업데이트 속도 결정
        model (tf.keras.Model): 메인 Q-네트워크 모델
        target_model (tf.keras.Model): 타겟 Q-네트워크 모델
    """
    
    def __init__(self, state_size: int, action_size: int,
                 memory_size: int = 2000, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, learning_rate: float = 0.001):
        """
        DQNAgent 초기화
        
        Args:
            state_size (int): 상태 공간의 차원 수
            action_size (int): 행동 공간의 차원 수
            memory_size (int, optional): 경험 리플레이 메모리 크기. 기본값은 2000.
            gamma (float, optional): 할인 계수. 기본값은 0.95.
            epsilon (float, optional): 초기 탐색 확률. 기본값은 1.0.
            epsilon_min (float, optional): 최소 탐색 확률. 기본값은 0.01.
            epsilon_decay (float, optional): 탐색 확률 감소율. 기본값은 0.995.
            learning_rate (float, optional): 학습률. 기본값은 0.001.
            
        Raises:
            ValueError: 상태 크기나 행동 크기가 1 미만인 경우 발생
            ValueError: 할인 계수가 0과 1 사이가 아닌 경우 발생
            ValueError: 탐색 확률이 0과 1 사이가 아닌 경우 발생
        """
        # 입력 유효성 검사
        if state_size < 1:
            raise ValueError("상태 공간의 차원 수는 1 이상이어야 합니다.")
        if action_size < 1:
            raise ValueError("행동 공간의 차원 수는 1 이상이어야 합니다.")
        if not 0 <= gamma <= 1:
            raise ValueError("할인 계수(gamma)는 0과 1 사이여야 합니다.")
        if not 0 <= epsilon <= 1 or not 0 <= epsilon_min <= 1:
            raise ValueError("탐색 확률(epsilon/epsilon_min)은 0과 1 사이여야 합니다.")
        if epsilon_decay <= 0 or epsilon_decay >= 1:
            raise ValueError("탐색 확률 감소율(epsilon_decay)은 0과 1 사이여야 합니다.")
            
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # 할인 계수
        self.epsilon = epsilon  # 탐색 확률
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        # 모델 구축
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self) -> tf.keras.Model:
        """
        DQN 모델 구축
        
        상태를 입력으로 받아 각 행동에 대한 Q-값을 출력하는 신경망 모델을 구축합니다.
        
        Returns:
            tf.keras.Model: 구축된 신경망 모델
            
        Notes:
            이 구현에서는 간단한 2층 완전연결 신경망을 사용합니다.
            더 복잡한 문제에는 더 깊거나 복잡한 구조의 신경망이 필요할 수 있습니다.
        """
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self) -> None:
        """
        타겟 모델 가중치를 메인 모델로 업데이트
        
        학습 안정성을 위해 유지되는 타겟 네트워크의 가중치를 
        메인 네트워크의 가중치로 복사합니다.
        
        Notes:
            DQN 알고리즘에서는 행동 선택에 메인 네트워크를 사용하고,
            Q-값 계산에 타겟 네트워크를 사용하여 학습의 안정성을 높입니다.
            이 함수는 주기적으로 호출되어 타겟 네트워크를 메인 네트워크와 동기화합니다.
        """
        self.target_model.set_weights(self.model.get_weights())
    
    def memorize(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """
        경험 리플레이 메모리에 트랜지션 저장
        
        에이전트가 경험한 상태 전이를 메모리에 저장하여 나중에 학습에 사용합니다.
        
        Args:
            state (np.ndarray): 현재 상태, 형태: (state_size,)
            action (int): 수행한 행동 인덱스
            reward (float): 받은 보상
            next_state (np.ndarray): 다음 상태, 형태: (state_size,)
            done (bool): 에피소드 종료 여부
            
        Raises:
            ValueError: 행동 인덱스가 유효하지 않은 경우 발생
            
        Notes:
            경험 리플레이는 연속된 샘플 간의 상관관계를 줄여 학습의 안정성을 높입니다.
            메모리가 가득 차면 오래된 경험부터 제거됩니다(deque의 maxlen 제한).
        """
        # 입력 유효성 검사
        if not 0 <= action < self.action_size:
            raise ValueError(f"행동 인덱스는 0에서 {self.action_size-1} 사이여야 합니다.")
            
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """
        현재 상태에서 행동 선택 (엡실론-그리디 정책)
        
        입력된 상태에서 엡실론-그리디 정책에 따라 행동을 선택합니다.
        epsilon 확률로 무작위 행동을, (1-epsilon) 확률로 최적 행동을 선택합니다.
        
        Args:
            state (np.ndarray): 현재 상태, 형태: (1, state_size)
            
        Returns:
            int: 선택된 행동의 인덱스
            
        Raises:
            ValueError: 상태 배열의 형태가 올바르지 않은 경우 발생
            
        Notes:
            실제 적용 시에는 탐색(exploration)과 활용(exploitation) 사이의
            균형을 위해 epsilon 값을 적절히 조정해야 합니다.
            학습 초기에는 높은 epsilon으로 다양한 행동을 탐색하고,
            학습이 진행됨에 따라 epsilon을 감소시켜 학습된 정책을 더 많이 활용합니다.
        """
        # 입력 유효성 검사
        if state.shape != (1, self.state_size):
            # 단일 샘플, 올바른 크기의 배열인지 확인
            if state.ndim == 1 and state.shape[0] == self.state_size:
                # 차원 추가
                state = state.reshape(1, -1)
            else:
                raise ValueError(f"상태 배열의 형태는 (1, {self.state_size})여야 합니다.")
        
        # 엡실론-그리디 정책에 따라 행동 선택
        if np.random.rand() <= self.epsilon:
            # 탐색: 무작위 행동 선택
            return random.randrange(self.action_size)
        
        # 활용: 현재 정책에 따른 최적 행동 선택
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size: int) -> float:
        """
        경험 리플레이를 통한 학습
        
        저장된 경험 중 무작위로 샘플링하여 Q-네트워크를 학습시킵니다.
        
        Args:
            batch_size (int): 학습에 사용할 경험 샘플 수
            
        Returns:
            float: 학습 손실값(loss)
            
        Raises:
            ValueError: 배치 크기가 1 미만인 경우 발생
            
        Notes:
            이 함수는 DQN의 핵심 학습 과정을 구현합니다.
            1. 경험 메모리에서 무작위로 배치를 샘플링
            2. 각 경험에 대해 타겟 Q-값 계산:
               - 종료 상태의 경우: Q-값 = 현재 보상
               - 비종료 상태의 경우: Q-값 = 현재 보상 + 할인된 미래 최대 Q-값
            3. 계산된 타겟 Q-값을 사용하여 메인 네트워크 학습
        """
        # 입력 유효성 검사
        if batch_size < 1:
            raise ValueError("배치 크기는 1 이상이어야 합니다.")
            
        # 메모리에 저장된 경험이 배치 크기보다 적으면 학습 건너뛰기
        if len(self.memory) < batch_size:
            return 0.0
            
        # 메모리에서 무작위로 배치 샘플링
        minibatch = random.sample(self.memory, batch_size)
        
        # 배치 처리를 위한 배열 초기화
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        
        # 상태와 다음 상태 배열 구성
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
        
        # 일괄 예측으로 계산 효율성 향상
        # 현재 상태에 대한 예측 (메인 네트워크)
        targets = self.model.predict(states, verbose=0)
        # 다음 상태에 대한 예측 (타겟 네트워크)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # 각 경험에 대해 타겟 Q-값 계산
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                # 종료 상태: 현재 보상만 사용
                targets[i, action] = reward
            else:
                # 비종료 상태: 현재 보상 + 할인된 미래 최대 Q-값
                targets[i, action] = reward + self.gamma * np.amax(next_q_values[i])
        
        # 타겟 Q-값으로 모델 학습
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # 엡실론 감소 (탐색 확률 줄이기)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss
    
    def load(self, name: str) -> None:
        """
        모델 가중치 로드
        
        저장된 모델 가중치를 로드하여 학습된 정책을 복원합니다.
        
        Args:
            name (str): 모델 파일 경로
            
        Raises:
            FileNotFoundError: 모델 파일이 존재하지 않는 경우 발생
            
        Notes:
            이 함수는 메인 모델의 가중치를 로드하고,
            타겟 모델에도 동일한 가중치를 복사합니다.
        """
        try:
            self.model.load_weights(name)
            self.update_target_model()
            print(f"모델 가중치를 '{name}'에서 성공적으로 로드했습니다.")
        except (OSError, IOError) as e:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없거나 로드하는 데 실패했습니다: {e}")
    
    def save(self, name: str) -> None:
        """
        모델 가중치 저장
        
        현재 학습된 모델의 가중치를 파일에 저장합니다.
        
        Args:
            name (str): 모델 파일 경로
            
        Raises:
            IOError: 파일 저장에 실패한 경우 발생
            
        Notes:
            이 함수는 메인 모델의 가중치만 저장합니다.
            타겟 모델의 가중치는 메인 모델과 동기화될 수 있으므로 저장할 필요가 없습니다.
        """
        try:
            self.model.save_weights(name)
            print(f"모델 가중치를 '{name}'에 성공적으로 저장했습니다.")
        except IOError as e:
            raise IOError(f"모델 가중치 저장에 실패했습니다: {e}")
        
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        모든 행동에 대한 Q-값 반환
        
        주어진 상태에서 모든 가능한 행동에 대한 Q-값을 계산합니다.
        
        Args:
            state (np.ndarray): 현재 상태, 형태: (1, state_size) 또는 (state_size,)
            
        Returns:
            np.ndarray: 행동별 Q-값, 형태: (action_size,)
            
        Raises:
            ValueError: 상태 배열의 형태가 올바르지 않은 경우 발생
            
        Notes:
            이 함수는 주로 디버깅, 분석, 시각화 목적으로 사용됩니다.
            실제 행동 선택에는 act() 메서드를 사용하세요.
        """
        # 입력 상태 형태 정규화
        if state.ndim == 1:
            state = state.reshape(1, -1)
        elif state.ndim != 2 or state.shape[0] != 1 or state.shape[1] != self.state_size:
            raise ValueError(f"상태 배열의 형태는 (1, {self.state_size}) 또는 ({self.state_size},)여야 합니다.")
        
        # Q-값 계산
        return self.model.predict(state, verbose=0)[0]
