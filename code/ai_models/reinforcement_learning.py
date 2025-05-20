"""
강화학습 기반 자극 최적화 모듈

이 모듈은 신경 전기자극 시스템에서 실시간 생체신호 피드백을 기반으로 자극 파라미터를 
최적화하기 위한 강화학습 알고리즘을 구현합니다.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any

# 실행 환경 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 경험 데이터 구조체 정의
Experience = namedtuple('Experience', 
                       field_names=['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """경험 재현 버퍼 클래스"""
    
    def __init__(self, buffer_size: int, batch_size: int):
        """초기화 함수
        
        Args:
            buffer_size (int): 버퍼 최대 크기
            batch_size (int): 배치 크기
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = Experience
        
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
           next_state: np.ndarray, done: bool):
        """경험 데이터 추가
        
        Args:
            state (np.ndarray): 현재 상태
            action (np.ndarray): 선택한 행동
            reward (float): 받은 보상
            next_state (np.ndarray): 다음 상태
            done (bool): 에피소드 종료 여부
        """
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)
        
    def sample(self) -> Tuple:
        """버퍼에서 무작위로 배치 크기만큼 경험 샘플링
        
        Returns:
            Tuple: (states, actions, rewards, next_states, dones) 배치 데이터
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(
            np.vstack([exp.state for exp in experiences if exp is not None])
        ).float().to(DEVICE)
        
        actions = torch.from_numpy(
            np.vstack([exp.action for exp in experiences if exp is not None])
        ).float().to(DEVICE)
        
        rewards = torch.from_numpy(
            np.vstack([exp.reward for exp in experiences if exp is not None])
        ).float().to(DEVICE)
        
        next_states = torch.from_numpy(
            np.vstack([exp.next_state for exp in experiences if exp is not None])
        ).float().to(DEVICE)
        
        dones = torch.from_numpy(
            np.vstack([exp.done for exp in experiences if exp is not None]).astype(np.uint8)
        ).float().to(DEVICE)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self) -> int:
        """버퍼에 저장된 경험 데이터 개수 반환
        
        Returns:
            int: 저장된 경험 개수
        """
        return len(self.memory)


class StimulationActorNetwork(nn.Module):
    """자극 파라미터를 결정하는 액터 네트워크 (정책 네트워크)"""
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int] = [256, 128], 
                learning_rate: float = 1e-4):
        """초기화 함수
        
        Args:
            state_size (int): 상태 공간 차원
            action_size (int): 행동 공간 차원 (자극 파라미터 개수)
            hidden_layers (List[int], optional): 은닉층 크기 목록. 기본값은 [256, 128].
            learning_rate (float, optional): 학습률. 기본값은 1e-4.
        """
        super(StimulationActorNetwork, self).__init__()
        
        # 네트워크 레이어 설정
        self.fc1 = nn.Linear(state_size, hidden_layers[0])
        self.bn1 = nn.BatchNorm1d(hidden_layers[0])
        
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.bn2 = nn.BatchNorm1d(hidden_layers[1])
        
        self.fc3 = nn.Linear(hidden_layers[1], action_size)
        
        # 출력 활성화 함수 (자극 파라미터 범위 제한)
        self.tanh = nn.Tanh()
        
        # 옵티마이저 설정
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """순전파 함수
        
        Args:
            state (torch.Tensor): 상태 텐서
            
        Returns:
            torch.Tensor: 정규화된 자극 파라미터 (행동)
        """
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        
        # Tanh 활성화로 출력 범위를 [-1, 1]로 제한
        return self.tanh(self.fc3(x))


class StimulationCriticNetwork(nn.Module):
    """상태-행동 가치를 평가하는 크리틱 네트워크"""
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int] = [256, 128], 
                learning_rate: float = 1e-3):
        """초기화 함수
        
        Args:
            state_size (int): 상태 공간 차원
            action_size (int): 행동 공간 차원 (자극 파라미터 개수)
            hidden_layers (List[int], optional): 은닉층 크기 목록. 기본값은 [256, 128].
            learning_rate (float, optional): 학습률. 기본값은 1e-3.
        """
        super(StimulationCriticNetwork, self).__init__()
        
        # 상태 처리 레이어
        self.fc1 = nn.Linear(state_size, hidden_layers[0])
        self.bn1 = nn.BatchNorm1d(hidden_layers[0])
        
        # 상태-행동 결합 레이어
        self.fc2 = nn.Linear(hidden_layers[0] + action_size, hidden_layers[1])
        
        # 가치 출력 레이어
        self.fc3 = nn.Linear(hidden_layers[1], 1)
        
        # 옵티마이저 설정
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """순전파 함수
        
        Args:
            state (torch.Tensor): 상태 텐서
            action (torch.Tensor): 행동 텐서 (자극 파라미터)
            
        Returns:
            torch.Tensor: 상태-행동 가치 추정치
        """
        xs = F.relu(self.bn1(self.fc1(state)))
        
        # 상태와 행동 결합
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)


class OrnsteinUhlenbeckNoise:
    """탐색을 위한 Ornstein-Uhlenbeck 노이즈 프로세스"""
    
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2, dt: float = 1e-2):
        """초기화 함수
        
        Args:
            size (int): 노이즈 벡터 크기
            mu (float, optional): 평균값. 기본값은 0.0.
            theta (float, optional): 평균 회귀 계수. 기본값은 0.15.
            sigma (float, optional): 변동성 계수. 기본값은 0.2.
            dt (float, optional): 시간 간격. 기본값은 1e-2.
        """
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()
        
    def reset(self):
        """노이즈 상태 초기화"""
        self.state = np.copy(self.mu)
        
    def sample(self) -> np.ndarray:
        """노이즈 샘플링
        
        Returns:
            np.ndarray: 생성된 노이즈 벡터
        """
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


class StimulationDDPGAgent:
    """자극 최적화를 위한 Deep Deterministic Policy Gradient (DDPG) 에이전트"""
    
    def __init__(self, state_size: int, action_size: int, action_bounds: Dict[str, Tuple[float, float]],
                buffer_size: int = 100000, batch_size: int = 64,
                gamma: float = 0.99, tau: float = 1e-3,
                actor_lr: float = 1e-4, critic_lr: float = 1e-3,
                hidden_layers: List[int] = [256, 128]):
        """초기화 함수
        
        Args:
            state_size (int): 상태 공간 차원
            action_size (int): 행동 공간 차원 (자극 파라미터 개수)
            action_bounds (Dict[str, Tuple[float, float]]): 각 자극 파라미터의 최소/최대 범위
            buffer_size (int, optional): 재현 버퍼 크기. 기본값은 100000.
            batch_size (int, optional): 배치 크기. 기본값은 64.
            gamma (float, optional): 할인율. 기본값은 0.99.
            tau (float, optional): 소프트 업데이트 계수. 기본값은 1e-3.
            actor_lr (float, optional): 액터 네트워크 학습률. 기본값은 1e-4.
            critic_lr (float, optional): 크리틱 네트워크 학습률. 기본값은 1e-3.
            hidden_layers (List[int], optional): 은닉층 크기 목록. 기본값은 [256, 128].
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_bounds = action_bounds
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        
        # 행동 파라미터 이름 및 범위 저장
        self.param_names = list(action_bounds.keys())
        self.param_ranges = np.array([(b[1] - b[0]) / 2 for b in action_bounds.values()])
        self.param_means = np.array([(b[0] + b[1]) / 2 for b in action_bounds.values()])
        
        # 액터 네트워크 (현재 & 타겟)
        self.actor_local = StimulationActorNetwork(state_size, action_size, hidden_layers, actor_lr).to(DEVICE)
        self.actor_target = StimulationActorNetwork(state_size, action_size, hidden_layers, actor_lr).to(DEVICE)
        
        # 크리틱 네트워크 (현재 & 타겟)
        self.critic_local = StimulationCriticNetwork(state_size, action_size, hidden_layers, critic_lr).to(DEVICE)
        self.critic_target = StimulationCriticNetwork(state_size, action_size, hidden_layers, critic_lr).to(DEVICE)
        
        # 타겟 네트워크 초기화 (가중치 복사)
        self.soft_update(self.actor_local, self.actor_target, 1.0)
        self.soft_update(self.critic_local, self.critic_target, 1.0)
        
        # 경험 재현 버퍼
        self.memory = ReplayBuffer(buffer_size, batch_size)
        
        # 노이즈 프로세스 (탐색용)
        self.noise = OrnsteinUhlenbeckNoise(action_size)
        
        # 학습 단계 카운터
        self.t_step = 0
        
    def denormalize_action(self, normalized_action: np.ndarray) -> Dict[str, float]:
        """정규화된 행동을 실제 자극 파라미터 값으로 변환
        
        Args:
            normalized_action (np.ndarray): [-1, 1] 범위의 정규화된 행동
            
        Returns:
            Dict[str, float]: 실제 범위의 자극 파라미터
        """
        # [-1, 1] 범위의 행동을 실제 파라미터 범위로 변환
        denorm_action = normalized_action * self.param_ranges + self.param_means
        
        # 파라미터 이름과 값을 매핑하여 딕셔너리 반환
        return {name: float(val) for name, val in zip(self.param_names, denorm_action)}
    
    def normalize_action(self, denormalized_action: Dict[str, float]) -> np.ndarray:
        """실제 자극 파라미터 값을 정규화된 행동으로 변환
        
        Args:
            denormalized_action (Dict[str, float]): 실제 범위의 자극 파라미터
            
        Returns:
            np.ndarray: [-1, 1] 범위의 정규화된 행동
        """
        # 파라미터 이름 순서대로 값 추출
        values = np.array([denormalized_action[name] for name in self.param_names])
        
        # 실제 파라미터 값을 [-1, 1] 범위로 정규화
        normalized = (values - self.param_means) / self.param_ranges
        
        # 값을 [-1, 1] 범위로 클리핑
        return np.clip(normalized, -1.0, 1.0)
    
    def step(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool):
        """에이전트의 학습 단계 수행
        
        Args:
            state (np.ndarray): 현재 상태
            action (np.ndarray): 선택한 행동
            reward (float): 받은 보상
            next_state (np.ndarray): 다음 상태
            done (bool): 에피소드 종료 여부
        """
        # 경험 메모리에 데이터 추가
        self.memory.add(state, action, reward, next_state, done)
        
        # 충분한 샘플이 모이면 학습 수행
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            
        # 타임스텝 증가
        self.t_step += 1
    
    def act(self, state: np.ndarray, add_noise: bool = True, noise_scale: float = 1.0) -> np.ndarray:
        """현재 정책에 따라 행동 선택
        
        Args:
            state (np.ndarray): 현재 상태
            add_noise (bool, optional): 탐색을 위한 노이즈 추가 여부. 기본값은 True.
            noise_scale (float, optional): 노이즈 스케일 팩터. 기본값은 1.0.
            
        Returns:
            np.ndarray: 선택된 행동 (정규화된 자극 파라미터)
        """
        # 평가 모드로 전환
        self.actor_local.eval()
        
        # 상태를 텐서로 변환
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            action = self.actor_local(state_tensor).cpu().data.numpy()
        
        # 훈련 모드로 전환
        self.actor_local.train()
        
        # 탐색을 위한 노이즈 추가
        if add_noise:
            action += noise_scale * self.noise.sample()
        
        # 행동 값을 [-1, 1] 범위로 클리핑
        return np.clip(action, -1.0, 1.0)
    
    def learn(self, experiences: Tuple):
        """경험 데이터로부터 학습 수행
        
        Args:
            experiences (Tuple): (states, actions, rewards, next_states, dones) 배치 데이터
        """
        states, actions, rewards, next_states, dones = experiences
        
        # ----- 크리틱 업데이트 -----
        # 타겟 액터를 사용하여 다음 상태에서의 행동 추정
        actions_next = self.actor_target(next_states)
        
        # 타겟 크리틱을 사용하여 다음 상태-행동 가치 추정
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Q 타겟 계산 (벨만 방정식)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # 현재 크리틱 네트워크로 Q 추정치 계산
        Q_expected = self.critic_local(states, actions)
        
        # 크리틱 손실 계산 및 업데이트
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_local.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_local.optimizer.step()
        
        # ----- 액터 업데이트 -----
        # 현재 액터의 행동 예측
        actions_pred = self.actor_local(states)
        
        # 액터 손실 계산 (정책 기울기)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # 액터 네트워크 업데이트
        self.actor_local.optimizer.zero_grad()
        actor_loss.backward()
        self.actor_local.optimizer.step()
        
        # ----- 타겟 네트워크 소프트 업데이트 -----
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        
    def soft_update(self, local_model: nn.Module, target_model: nn.Module, tau: float):
        """타겟 네트워크 소프트 업데이트
        
        Args:
            local_model (nn.Module): 소스 네트워크 (현재 학습중인 네트워크)
            target_model (nn.Module): 타겟 네트워크 (소프트 업데이트할 네트워크)
            tau (float): 업데이트 비율 (0-1)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
            
    def save_models(self, actor_path: str, critic_path: str):
        """모델 가중치 저장
        
        Args:
            actor_path (str): 액터 모델 저장 경로
            critic_path (str): 크리틱 모델 저장 경로
        """
        torch.save(self.actor_local.state_dict(), actor_path)
        torch.save(self.critic_local.state_dict(), critic_path)
        
    def load_models(self, actor_path: str, critic_path: str):
        """모델 가중치 로드
        
        Args:
            actor_path (str): 액터 모델 로드 경로
            critic_path (str): 크리틱 모델 로드 경로
        """
        self.actor_local.load_state_dict(torch.load(actor_path))
        self.critic_local.load_state_dict(torch.load(critic_path))
        
        # 타겟 네트워크에도 복사
        self.soft_update(self.actor_local, self.actor_target, 1.0)
        self.soft_update(self.critic_local, self.critic_target, 1.0)


class AdaptiveStimulationController:
    """생체신호 피드백 기반 적응형 자극 제어 시스템"""
    
    def __init__(self, state_features: List[str], stimulation_params: Dict[str, Tuple[float, float]],
                reward_weights: Dict[str, float] = None, update_interval: int = 10,
                buffer_size: int = 50000, batch_size: int = 64,
                safety_constraints: Dict[str, Tuple[float, float]] = None):
        """초기화 함수
        
        Args:
            state_features (List[str]): 상태 특징 이름 목록
            stimulation_params (Dict[str, Tuple[float, float]]): 자극 파라미터 및 범위 {이름: (최소값, 최대값)}
            reward_weights (Dict[str, float], optional): 보상 가중치 {요소: 가중치}. 기본값은 None.
            update_interval (int, optional): 자극 파라미터 업데이트 간격. 기본값은 10.
            buffer_size (int, optional): 재현 버퍼 크기. 기본값은 50000.
            batch_size (int, optional): 배치 크기. 기본값은 64.
            safety_constraints (Dict[str, Tuple[float, float]], optional): 안전 제약 조건. 기본값은 None.
        """
        self.state_features = state_features
        self.state_size = len(state_features)
        
        self.stimulation_params = stimulation_params
        self.action_size = len(stimulation_params)
        
        # 안전 제약 조건
        self.safety_constraints = safety_constraints if safety_constraints else {}
        
        # 보상 계산을 위한 가중치
        if reward_weights is None:
            # 기본 가중치: 모든 요소에 동일한 가중치
            self.reward_weights = {
                'bdnf_expression': 1.0,        # BDNF 발현 수준
                'axon_growth': 1.0,            # 축삭 성장률
                'inflammation': -1.0,          # 염증 수준 (음수 가중치)
                'pain': -1.0,                  # 통증 수준 (음수 가중치)
                'function_improvement': 2.0,   # 기능 개선 (높은 가중치)
                'side_effects': -2.0           # 부작용 (높은 음수 가중치)
            }
        else:
            self.reward_weights = reward_weights
            
        # 업데이트 간격
        self.update_interval = update_interval
        self.step_counter = 0
        
        # 현재 자극 파라미터 초기화 (중간값으로)
        self.current_params = {
            name: (bounds[0] + bounds[1]) / 2
            for name, bounds in stimulation_params.items()
        }
        
        # 상태 정규화를 위한 통계치
        self.state_mean = np.zeros(self.state_size)
        self.state_std = np.ones(self.state_size)
        self.state_count = 0
        
        # 강화학습 에이전트
        self.agent = StimulationDDPGAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            action_bounds=stimulation_params,
            buffer_size=buffer_size,
            batch_size=batch_size
        )
        
        # 학습 이력
        self.reward_history = []
        self.parameter_history = {name: [] for name in stimulation_params.keys()}
        
    def update_state_statistics(self, state: np.ndarray):
        """상태 정규화를 위한 통계치 업데이트 (이동 평균/표준편차)
        
        Args:
            state (np.ndarray): 현재 상태
        """
        if self.state_count == 0:
            self.state_mean = state
            self.state_std = np.ones_like(state)
            self.state_count = 1
        else:
            # 이동 평균 업데이트
            new_count = self.state_count + 1
            self.state_mean = (self.state_mean * self.state_count + state) / new_count
            
            # 이동 표준편차 업데이트 (온라인 알고리즘)
            old_mean = self.state_mean
            old_var = self.state_std ** 2
            
            new_mean = old_mean + (state - old_mean) / new_count
            new_var = ((self.state_count - 1) * old_var + 
                      (state - old_mean) * (state - new_mean)) / self.state_count
            
            self.state_mean = new_mean
            self.state_std = np.sqrt(new_var + 1e-8)  # 수치 안정성
            self.state_count = new_count
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """상태 정규화
        
        Args:
            state (np.ndarray): 원본 상태
            
        Returns:
            np.ndarray: 정규화된 상태
        """
        return (state - self.state_mean) / (self.state_std + 1e-8)
    
    def compute_reward(self, biosignals: Dict[str, float], prev_biosignals: Dict[str, float] = None) -> float:
        """생체신호 기반 보상 계산
        
        Args:
            biosignals (Dict[str, float]): 현재 생체신호 측정값
            prev_biosignals (Dict[str, float], optional): 이전 생체신호 측정값. 기본값은 None.
            
        Returns:
            float: 계산된 보상값
        """
        reward = 0.0
        
        # 이전 신호가 없는 경우 (첫 스텝), 변화 대신 절대값 사용
        if prev_biosignals is None:
            for key, weight in self.reward_weights.items():
                if key in biosignals:
                    reward += weight * biosignals[key]
        else:
            # 이전 대비 개선 정도로 보상 계산
            for key, weight in self.reward_weights.items():
                if key in biosignals and key in prev_biosignals:
                    # 변화율 계산
                    change = biosignals[key] - prev_biosignals[key]
                    
                    # 양수 가중치는 증가가 좋음, 음수 가중치는 감소가 좋음
                    if weight > 0:
                        reward += weight * change
                    else:
                        reward += weight * (-change)
        
        # 안전 제약 위반 시 큰 음수 보상
        for param_name, param_value in self.current_params.items():
            if param_name in self.safety_constraints:
                safe_min, safe_max = self.safety_constraints[param_name]
                if param_value < safe_min or param_value > safe_max:
                    reward -= 50.0  # 안전 위반 큰 페널티
        
        return reward
    
    def update_stimulation_parameters(self, biosignals: Dict[str, float], prev_biosignals: Dict[str, float] = None) -> Dict[str, float]:
        """생체신호 피드백에 기반하여 자극 파라미터 업데이트
        
        Args:
            biosignals (Dict[str, float]): 현재 생체신호 측정값
            prev_biosignals (Dict[str, float], optional): 이전 생체신호 측정값. 기본값은 None.
            
        Returns:
            Dict[str, float]: 업데이트된 자극 파라미터
        """
        # 스텝 카운터 증가
        self.step_counter += 1
        
        # 상태 특징 추출
        state = np.array([biosignals.get(feature, 0.0) for feature in self.state_features])
        
        # 상태 정규화
        self.update_state_statistics(state)
        normalized_state = self.normalize_state(state)
        
        # 현재 자극 파라미터 정규화
        normalized_action = self.agent.normalize_action(self.current_params)
        
        # 보상 계산
        reward = self.compute_reward(biosignals, prev_biosignals)
        self.reward_history.append(reward)
        
        # 이전 상태가 있는 경우 (두 번째 스텝부터)
        if hasattr(self, 'prev_state'):
            # 에이전트 학습
            self.agent.step(self.prev_state, normalized_action, reward, normalized_state, False)
        
        # 현재 상태 저장
        self.prev_state = normalized_state
        
        # 일정 간격마다 자극 파라미터 업데이트
        if self.step_counter % self.update_interval == 0:
            # 에이전트로부터 새 행동 (자극 파라미터) 획득
            normalized_action = self.agent.act(normalized_state, add_noise=True, 
                                            noise_scale=max(0.1, 1.0 - self.step_counter / 10000))
            
            # 정규화된 행동을 실제 자극 파라미터로 변환
            self.current_params = self.agent.denormalize_action(normalized_action[0])
            
            # 안전 제약 적용
            for param_name, param_value in self.current_params.items():
                # 자극 파라미터 범위 제약
                param_min, param_max = self.stimulation_params[param_name]
                self.current_params[param_name] = max(param_min, min(param_max, param_value))
                
                # 추가 안전 제약 (있는 경우)
                if param_name in self.safety_constraints:
                    safe_min, safe_max = self.safety_constraints[param_name]
                    self.current_params[param_name] = max(safe_min, min(safe_max, self.current_params[param_name]))
        
        # 파라미터 이력 기록
        for name, value in self.current_params.items():
            self.parameter_history[name].append(value)
            
        return self.current_params
    
    def get_optimal_protocol(self, patient_condition: str, recovery_stage: str) -> Dict[str, float]:
        """환자 상태 및 회복 단계에 맞는 최적의 자극 프로토콜 추천
        
        Args:
            patient_condition (str): 환자 상태/질환 (예: "peripheral_nerve_injury", "spinal_cord_injury")
            recovery_stage (str): 회복 단계 (예: "acute", "subacute", "regeneration", "remodeling")
            
        Returns:
            Dict[str, float]: 추천된 자극 파라미터
        """
        # 여기서는 사전 학습된 모델에 기반하여 추천
        # 실제 구현에서는 더 복잡한 추론 및 개인화가 필요함
        
        # 기본 프로토콜
        base_protocol = {name: (bounds[0] + bounds[1]) / 2 for name, bounds in self.stimulation_params.items()}
        
        # 환자 상태에 따른 조정
        if patient_condition == "peripheral_nerve_injury":
            if "frequency" in base_protocol:
                base_protocol["frequency"] = 20.0  # 20Hz (BDNF/TrkB 활성화에 효과적)
            if "pulse_width" in base_protocol:
                base_protocol["pulse_width"] = 0.3  # 300µs
                
        elif patient_condition == "spinal_cord_injury":
            if "frequency" in base_protocol:
                base_protocol["frequency"] = 40.0  # 40Hz (CPG 활성화에 효과적)
            if "amplitude" in base_protocol:
                base_protocol["amplitude"] = 3.5  # 더 높은 진폭
                
        elif patient_condition == "neuropathic_pain":
            if "frequency" in base_protocol:
                base_protocol["frequency"] = 80.0  # 80Hz (통증 게이트 조절에 효과적)
            if "pulse_width" in base_protocol:
                base_protocol["pulse_width"] = 0.2  # 200µs
        
        # 회복 단계에 따른 추가 조정
        if recovery_stage == "acute":
            # 급성기: 항염증, 신경보호 중심
            if "frequency" in base_protocol:
                base_protocol["frequency"] *= 0.8  # 주파수 감소
            if "duty_cycle" in base_protocol:
                base_protocol["duty_cycle"] = 0.3  # 낮은 듀티 사이클
                
        elif recovery_stage == "subacute":
            # 아급성기: 신경영양인자 발현 촉진
            if "burst_mode" in base_protocol:
                base_protocol["burst_mode"] = 1.0  # 버스트 모드 활성화
                
        elif recovery_stage == "regeneration":
            # 재생기: 축삭 성장 촉진
            if "amplitude" in base_protocol:
                base_protocol["amplitude"] *= 1.2  # 진폭 증가
                
        elif recovery_stage == "remodeling":
            # 재조직화: 시냅스 연결 및 기능 통합
            if "frequency" in base_protocol:
                base_protocol["frequency"] *= 1.5  # 주파수 증가
            if "burst_mode" in base_protocol:
                base_protocol["burst_mode"] = 0.0  # 버스트 모드 비활성화
        
        return base_protocol
    
    def plot_learning_progress(self, figsize: Tuple[int, int] = (12, 8)):
        """학습 진행 상황 시각화
        
        Args:
            figsize (Tuple[int, int], optional): 그림 크기. 기본값은 (12, 8).
        """
        if not self.reward_history:
            print("학습 데이터가 아직 없습니다.")
            return
        
        plt.figure(figsize=figsize)
        
        # 보상 그래프
        plt.subplot(2, 1, 1)
        plt.plot(self.reward_history)
        plt.title('Cumulative Reward')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # 자극 파라미터 그래프
        plt.subplot(2, 1, 2)
        for param_name, param_history in self.parameter_history.items():
            plt.plot(param_history, label=param_name)
        
        plt.title('Stimulation Parameters')
        plt.xlabel('Step')
        plt.ylabel('Parameter Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


# 사용 예시
if __name__ == "__main__":
    # 상태 특징 정의
    state_features = [
        'emg_amplitude',       # 근전도 진폭
        'emg_frequency',       # 근전도 주파수
        'eng_activity',        # 신경 활동
        'blood_flow',          # 혈류량
        'inflammation_marker', # 염증 마커
        'pain_level',          # 통증 수준
        'movement_accuracy'    # 움직임 정확도
    ]
    
    # 자극 파라미터 및 범위 정의
    stimulation_params = {
        'frequency': (5.0, 100.0),     # 주파수 (Hz)
        'amplitude': (0.5, 5.0),       # 진폭 (mA)
        'pulse_width': (0.1, 0.5),     # 펄스 폭 (ms)
        'burst_mode': (0.0, 1.0),      # 버스트 모드 (0-1)
        'duty_cycle': (0.2, 0.8)       # 듀티 사이클 (0-1)
    }
    
    # 보상 가중치 정의
    reward_weights = {
        'emg_amplitude': 1.0,          # 근활성도 (높을수록 좋음)
        'eng_activity': 1.0,           # 신경 활동 (높을수록 좋음)
        'blood_flow': 0.8,             # 혈류량 (높을수록 좋음)
        'inflammation_marker': -1.2,   # 염증 (낮을수록 좋음)
        'pain_level': -1.5,            # 통증 (낮을수록 좋음)
        'movement_accuracy': 2.0       # 움직임 정확도 (높을수록 좋음)
    }
    
    # 안전 제약 조건
    safety_constraints = {
        'amplitude': (0.5, 4.0),       # 안전한 진폭 범위
        'frequency': (5.0, 80.0)       # 안전한 주파수 범위
    }
    
    # 제어기 초기화
    controller = AdaptiveStimulationController(
        state_features=state_features,
        stimulation_params=stimulation_params,
        reward_weights=reward_weights,
        update_interval=5,
        safety_constraints=safety_constraints
    )
    
    # 시뮬레이션 (실제로는 실시간 생체신호 측정으로 대체)
    num_steps = 200
    
    # 초기 생체신호 (가상)
    biosignals = {
        'emg_amplitude': 0.2,
        'emg_frequency': 30.0,
        'eng_activity': 0.1,
        'blood_flow': 0.5,
        'inflammation_marker': 0.8,
        'pain_level': 0.7,
        'movement_accuracy': 0.3
    }
    
    # 시뮬레이션 실행
    for step in range(num_steps):
        # 자극 파라미터 업데이트
        prev_biosignals = biosignals.copy()
        params = controller.update_stimulation_parameters(biosignals, prev_biosignals if step > 0 else None)
        
        # 가상의 생체신호 변화 (시뮬레이션용)
        # 실제 환경에서는 실제 측정값으로 대체
        
        # 자극 효과 시뮬레이션 (단순화된 모델)
        frequency_effect = 0.01 * (params['frequency'] - 40) / 40  # 주파수 효과
        amplitude_effect = 0.02 * (params['amplitude'] - 2) / 2   # 진폭 효과
        pulse_effect = 0.015 * (params['pulse_width'] - 0.3) / 0.3  # 펄스 폭 효과
        
        # 생체신호 업데이트 (간단한 시뮬레이션)
        biosignals['emg_amplitude'] += frequency_effect + amplitude_effect + 0.002 * np.random.randn()
        biosignals['eng_activity'] += amplitude_effect + pulse_effect + 0.001 * np.random.randn()
        biosignals['blood_flow'] += 0.5 * amplitude_effect + 0.002 * np.random.randn()
        biosignals['inflammation_marker'] -= 0.001 * step + 0.005 * np.random.randn()  # 시간에 따른 자연 감소
        biosignals['pain_level'] -= frequency_effect + 0.001 * step + 0.01 * np.random.randn()
        biosignals['movement_accuracy'] += 0.002 * step + frequency_effect + 0.01 * np.random.randn()
        
        # 값 범위 제한
        for key in biosignals:
            biosignals[key] = max(0.0, min(1.0, biosignals[key]))
        
        # 진행 상황 출력 (20스텝마다)
        if step % 20 == 0:
            print(f"Step {step}, Parameters: {params}")
            print(f"Biosignals: {biosignals}")
            print("---")
    
    # 학습 진행 시각화
    controller.plot_learning_progress()
    
    # 최적 프로토콜 추천 예시
    optimal_params = controller.get_optimal_protocol("peripheral_nerve_injury", "regeneration")
    print("\nRecommended protocol for peripheral nerve injury (regeneration phase):")
    for param, value in optimal_params.items():
        print(f"  {param}: {value}")
