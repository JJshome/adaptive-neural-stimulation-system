"""PPO(Proximal Policy Optimization) 강화학습 모델

이 모듈은 전기자극 파라미터 최적화를 위한 PPO 강화학습 알고리즘을 구현합니다.
신경 상태와 회복 피드백에 기반하여 최적의 자극 파라미터를 학습합니다.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Union
import os
import pickle
from datetime import datetime
import scipy.signal

class PPOBuffer:
    """PPO 알고리즘을 위한 경험 버퍼"""
    
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int = 2048, gamma: float = 0.99, lam: float = 0.95):
        """
        PPO 버퍼 초기화
        
        매개변수:
            state_dim (int): 상태 공간 차원
            action_dim (int): 행동 공간 차원
            buffer_size (int): 버퍼 크기
            gamma (float): 할인율
            lam (float): GAE(Generalized Advantage Estimation) 람다
        """
        self.state_buf = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros(buffer_size, dtype=np.float32)
        self.value_buf = np.zeros(buffer_size, dtype=np.float32)
        self.logp_buf = np.zeros(buffer_size, dtype=np.float32)
        self.advantage_buf = np.zeros(buffer_size, dtype=np.float32)
        self.return_buf = np.zeros(buffer_size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx = 0, 0
        self.max_size = buffer_size
        
    def store(self, state: np.ndarray, action: np.ndarray, reward: float, value: float, logp: float):
        """
        트랜지션 저장
        
        매개변수:
            state (np.ndarray): 상태
            action (np.ndarray): 행동
            reward (float): 보상
            value (float): 가치 함수 추정치
            logp (float): 로그 확률
        """
        assert self.ptr < self.max_size
        self.state_buf[self.ptr] = state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.value_buf[self.ptr] = value
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
        
    def finish_path(self, last_value: float = 0):
        """
        경로 완료 처리 및 어드밴티지 계산
        
        매개변수:
            last_value (float): 마지막 상태의 가치 함수 추정치
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.reward_buf[path_slice], last_value)
        values = np.append(self.value_buf[path_slice], last_value)
        
        # GAE-Lambda 어드밴티지 계산
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantage_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        
        # 보상-투-고(rewards-to-go) 계산
        self.return_buf[path_slice] = self._discount_cumsum(rewards, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr
    
    def _discount_cumsum(self, x: np.ndarray, discount: float) -> np.ndarray:
        """
        할인된 누적합 계산
        
        매개변수:
            x (np.ndarray): 입력 배열
            discount (float): 할인율
            
        반환값:
            np.ndarray: 할인된 누적합
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    
    def get(self) -> Dict[str, np.ndarray]:
        """
        모든 버퍼 데이터 가져오기
        
        반환값:
            Dict[str, np.ndarray]: 버퍼 데이터
        """
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        
        # 어드밴티지 정규화
        adv_mean = np.mean(self.advantage_buf)
        adv_std = np.std(self.advantage_buf)
        self.advantage_buf = (self.advantage_buf - adv_mean) / (adv_std + 1e-8)
        
        return {
            "states": self.state_buf,
            "actions": self.action_buf,
            "advantages": self.advantage_buf,
            "returns": self.return_buf,
            "logps": self.logp_buf
        }


class PPOActorCritic:
    """PPO 액터-크리틱 모델"""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 action_bounds: Dict[str, Tuple[float, float]],
                 actor_hidden_sizes: List[int] = [64, 64],
                 critic_hidden_sizes: List[int] = [64, 64],
                 activation: str = 'tanh',
                 learning_rate: float = 3e-4,
                 clip_ratio: float = 0.2,
                 target_kl: float = 0.01,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        """
        PPO 액터-크리틱 모델 초기화
        
        매개변수:
            state_dim (int): 상태 공간 차원
            action_dim (int): 행동 공간 차원
            action_bounds (Dict[str, Tuple[float, float]]): 행동 제한 범위
            actor_hidden_sizes (List[int]): 액터 네트워크 은닉층 크기
            critic_hidden_sizes (List[int]): 크리틱 네트워크 은닉층 크기
            activation (str): 활성화 함수
            learning_rate (float): 학습률
            clip_ratio (float): PPO 클리핑 비율
            target_kl (float): 목표 KL 발산
            value_coef (float): 가치 손실 계수
            entropy_coef (float): 엔트로피 계수
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # 액션 범위 계산
        self.action_scale = []
        self.action_bias = []
        
        for name, bounds in action_bounds.items():
            low, high = bounds
            self.action_scale.append((high - low) / 2.0)
            self.action_bias.append((high + low) / 2.0)
        
        self.action_scale = np.array(self.action_scale, dtype=np.float32)
        self.action_bias = np.array(self.action_bias, dtype=np.float32)
        
        # 로그 표준편차 초기값 (0.5를 중심으로 작은 노이즈 추가)
        self.log_std_init = np.ones(action_dim, dtype=np.float32) * np.log(0.5) + np.random.normal(0, 0.01, action_dim).astype(np.float32)
        
        # 네트워크 구축
        self.actor_network = self._build_actor(actor_hidden_sizes, activation)
        self.critic_network = self._build_critic(critic_hidden_sizes, activation)
        
        # 옵티마이저
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
    def _build_actor(self, hidden_sizes: List[int], activation: str) -> tf.keras.Model:
        """
        액터 네트워크 구축
        
        매개변수:
            hidden_sizes (List[int]): 은닉층 크기
            activation (str): 활성화 함수
            
        반환값:
            tf.keras.Model: 구축된 액터 네트워크
        """
        # 상태 입력
        states = layers.Input(shape=(self.state_dim,))
        
        # 은닉층
        x = states
        for size in hidden_sizes:
            x = layers.Dense(size, activation=activation)(x)
        
        # 평균값 출력
        mu = layers.Dense(self.action_dim, activation='tanh')(x)
        
        # 로그 표준편차 (학습 가능한 파라미터)
        log_std = tf.Variable(self.log_std_init, name='log_std')
        
        # 모델 생성
        model = tf.keras.Model(inputs=states, outputs=[mu, log_std])
        return model
    
    def _build_critic(self, hidden_sizes: List[int], activation: str) -> tf.keras.Model:
        """
        크리틱 네트워크 구축
        
        매개변수:
            hidden_sizes (List[int]): 은닉층 크기
            activation (str): 활성화 함수
            
        반환값:
            tf.keras.Model: 구축된 크리틱 네트워크
        """
        # 상태 입력
        states = layers.Input(shape=(self.state_dim,))
        
        # 은닉층
        x = states
        for size in hidden_sizes:
            x = layers.Dense(size, activation=activation)(x)
        
        # 가치 출력
        value = layers.Dense(1)(x)
        
        # 모델 생성
        model = tf.keras.Model(inputs=states, outputs=value)
        return model
    
    def _get_policy_distribution(self, mu: tf.Tensor, log_std: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        정책 분포 계산
        
        매개변수:
            mu (tf.Tensor): 평균
            log_std (tf.Tensor): 로그 표준편차
            
        반환값:
            Dict[str, tf.Tensor]: 정책 분포 파라미터
        """
        std = tf.exp(log_std)
        return {"mu": mu, "std": std}
    
    def _sample_action(self, policy: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        정책에서 액션 샘플링
        
        매개변수:
            policy (Dict[str, tf.Tensor]): 정책 분포 파라미터
            
        반환값:
            Tuple[tf.Tensor, tf.Tensor]: (샘플링된 액션, 로그 확률)
        """
        mu, std = policy["mu"], policy["std"]
        
        # 정규 분포에서 샘플링
        noise = tf.random.normal(shape=mu.shape)
        pi = mu + noise * std
        
        # 로그 확률 계산
        logp = self._gaussian_likelihood(pi, mu, log_std)
        
        # 액션 스케일링
        pi_scaled = self._scale_action(pi)
        
        return pi_scaled, logp
    
    def _gaussian_likelihood(self, x: tf.Tensor, mu: tf.Tensor, log_std: tf.Tensor) -> tf.Tensor:
        """
        가우시안 로그 확률 밀도 계산
        
        매개변수:
            x (tf.Tensor): 샘플
            mu (tf.Tensor): 평균
            log_std (tf.Tensor): 로그 표준편차
            
        반환값:
            tf.Tensor: 로그 확률
        """
        pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + 1e-8))**2 + 2 * log_std + np.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)
    
    def _scale_action(self, action: tf.Tensor) -> tf.Tensor:
        """
        액션 스케일링 ([-1, 1] -> [low, high])
        
        매개변수:
            action (tf.Tensor): 원본 액션 ([-1, 1] 범위)
            
        반환값:
            tf.Tensor: 스케일링된 액션
        """
        return action * self.action_scale + self.action_bias
    
    def _unscale_action(self, action: tf.Tensor) -> tf.Tensor:
        """
        액션 언스케일링 ([low, high] -> [-1, 1])
        
        매개변수:
            action (tf.Tensor): 스케일링된 액션
            
        반환값:
            tf.Tensor: 원본 액션 ([-1, 1] 범위)
        """
        return (action - self.action_bias) / self.action_scale
    
    def act(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        상태에 기반하여 액션 선택
        
        매개변수:
            state (np.ndarray): 현재 상태
            
        반환값:
            Tuple[np.ndarray, float, float]: (선택된 액션, 로그 확률, 가치 추정치)
        """
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        
        # 액터 네트워크 추론
        mu, log_std = self.actor_network(state_tensor)
        policy = self._get_policy_distribution(mu, log_std)
        
        # 액션 샘플링
        action, logp = self._sample_action(policy)
        
        # 크리틱 네트워크 추론
        value = self.critic_network(state_tensor)
        
        return action.numpy()[0], logp.numpy()[0], value.numpy()[0, 0]
    
    def predict_value(self, state: np.ndarray) -> float:
        """
        상태의 가치 예측
        
        매개변수:
            state (np.ndarray): 상태
            
        반환값:
            float: 가치 추정치
        """
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        value = self.critic_network(state_tensor)
        return value.numpy()[0, 0]
    
    def update(self, buffer_data: Dict[str, np.ndarray], train_iters: int = 80) -> Dict[str, float]:
        """
        PPO 업데이트 수행
        
        매개변수:
            buffer_data (Dict[str, np.ndarray]): 버퍼 데이터
            train_iters (int): 학습 반복 횟수
            
        반환값:
            Dict[str, float]: 학습 지표
        """
        states = buffer_data["states"]
        actions = buffer_data["actions"]
        advantages = buffer_data["advantages"]
        returns = buffer_data["returns"]
        old_logps = buffer_data["logps"]
        
        # 메트릭 추적
        metrics = {
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "kl": []
        }
        
        # 학습 반복
        for i in range(train_iters):
            # 미니배치 처리
            with tf.GradientTape() as tape:
                # 액터 네트워크 추론
                mu, log_std = self.actor_network(states)
                policy = self._get_policy_distribution(mu, log_std)
                
                # 현재 로그 확률 계산
                unscaled_actions = self._unscale_action(actions)
                logp = self._gaussian_likelihood(unscaled_actions, mu, log_std)
                
                # 비율 계산 (현재 정책 / 이전 정책)
                ratio = tf.exp(logp - old_logps)
                
                # 클리핑된 목적 함수
                clip_adv = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clip_adv))
                
                # 크리틱 네트워크 추론
                values = self.critic_network(states)
                critic_loss = tf.reduce_mean((returns - values)**2)
                
                # 엔트로피 계산
                std = tf.exp(log_std)
                entropy = tf.reduce_mean(0.5 + 0.5 * np.log(2 * np.pi) + log_std)
                
                # 전체 손실
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # KL 발산 계산
            approx_kl = tf.reduce_mean(old_logps - logp)
            
            # 조기 종료 (KL 발산이 너무 큰 경우)
            if approx_kl > 1.5 * self.target_kl:
                break
            
            # 그래디언트 계산 및 적용
            variables = self.actor_network.trainable_variables + self.critic_network.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            
            # 메트릭 저장
            metrics["actor_loss"].append(actor_loss.numpy())
            metrics["critic_loss"].append(critic_loss.numpy())
            metrics["entropy"].append(entropy.numpy())
            metrics["kl"].append(approx_kl.numpy())
        
        # 평균 메트릭 계산
        for key in metrics:
            metrics[key] = np.mean(metrics[key])
        
        return metrics
    
    def save_model(self, path: str):
        """
        모델 저장
        
        매개변수:
            path (str): 저장 경로
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        self.actor_network.save_weights(os.path.join(path, "actor_weights.h5"))
        self.critic_network.save_weights(os.path.join(path, "critic_weights.h5"))
        
        # 추가 설정 저장
        model_config = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_bounds": self.action_bounds,
            "action_scale": self.action_scale,
            "action_bias": self.action_bias,
            "log_std_init": self.log_std_init,
            "clip_ratio": self.clip_ratio,
            "target_kl": self.target_kl,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef
        }
        
        with open(os.path.join(path, "model_config.pkl"), 'wb') as f:
            pickle.dump(model_config, f)
    
    def load_model(self, path: str):
        """
        모델 로드
        
        매개변수:
            path (str): 로드 경로
        """
        self.actor_network.load_weights(os.path.join(path, "actor_weights.h5"))
        self.critic_network.load_weights(os.path.join(path, "critic_weights.h5"))
        
        # 추가 설정 로드
        with open(os.path.join(path, "model_config.pkl"), 'rb') as f:
            model_config = pickle.load(f)
            
        self.action_scale = model_config["action_scale"]
        self.action_bias = model_config["action_bias"]
        self.log_std_init = model_config["log_std_init"]
        self.clip_ratio = model_config["clip_ratio"]
        self.target_kl = model_config["target_kl"]
        self.value_coef = model_config["value_coef"]
        self.entropy_coef = model_config["entropy_coef"]


class PPOAgent:
    """PPO 에이전트"""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 action_bounds: Dict[str, Tuple[float, float]],
                 buffer_size: int = 2048,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 actor_hidden_sizes: List[int] = [64, 64],
                 critic_hidden_sizes: List[int] = [64, 64],
                 activation: str = 'tanh',
                 learning_rate: float = 3e-4,
                 clip_ratio: float = 0.2,
                 target_kl: float = 0.01,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 train_iters: int = 80,
                 log_dir: Optional[str] = None):
        """
        PPO 에이전트 초기화
        
        매개변수:
            state_dim (int): 상태 공간 차원
            action_dim (int): 행동 공간 차원
            action_bounds (Dict[str, Tuple[float, float]]): 행동 제한 범위
            buffer_size (int): 버퍼 크기
            batch_size (int): 배치 크기
            gamma (float): 할인율
            lam (float): GAE 람다
            actor_hidden_sizes (List[int]): 액터 네트워크 은닉층 크기
            critic_hidden_sizes (List[int]): 크리틱 네트워크 은닉층 크기
            activation (str): 활성화 함수
            learning_rate (float): 학습률
            clip_ratio (float): PPO 클리핑 비율
            target_kl (float): 목표 KL 발산
            value_coef (float): 가치 손실 계수
            entropy_coef (float): 엔트로피 계수
            train_iters (int): 학습 반복 횟수
            log_dir (str): 로그 디렉토리
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.train_iters = train_iters
        
        # 경험 버퍼
        self.buffer = PPOBuffer(state_dim, action_dim, buffer_size, gamma, lam)
        
        # 액터-크리틱 모델
        self.ac_model = PPOActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            action_bounds=action_bounds,
            actor_hidden_sizes=actor_hidden_sizes,
            critic_hidden_sizes=critic_hidden_sizes,
            activation=activation,
            learning_rate=learning_rate,
            clip_ratio=clip_ratio,
            target_kl=target_kl,
            value_coef=value_coef,
            entropy_coef=entropy_coef
        )
        
        # 로깅
        self.log_dir = log_dir if log_dir else f"./logs/ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # 학습 메트릭
        self.metrics = {
            "episode_returns": [],
            "episode_lengths": [],
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "kl": []
        }
    
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        상태에 기반한 액션 선택
        
        매개변수:
            state (np.ndarray): 현재 상태
            
        반환값:
            Tuple[np.ndarray, float, float]: (선택된 액션, 로그 확률, 가치 추정치)
        """
        return self.ac_model.act(state)
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, value: float, logp: float):
        """
        트랜지션 저장
        
        매개변수:
            state (np.ndarray): 상태
            action (np.ndarray): 행동
            reward (float): 보상
            value (float): 가치 함수 추정치
            logp (float): 로그 확률
        """
        self.buffer.store(state, action, reward, value, logp)
    
    def finish_episode(self, last_value: float = 0):
        """
        에피소드 완료 처리
        
        매개변수:
            last_value (float): 마지막 상태의 가치 함수 추정치
        """
        self.buffer.finish_path(last_value)
    
    def update(self) -> Dict[str, float]:
        """
        정책 업데이트
        
        반환값:
            Dict[str, float]: 학습 지표
        """
        # 버퍼에서 데이터 가져오기
        buffer_data = self.buffer.get()
        
        # 모델 업데이트
        metrics = self.ac_model.update(buffer_data, self.train_iters)
        
        # 메트릭 저장
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        return metrics
    
    def train(self, env, epochs: int = 100, steps_per_epoch: int = 2048, max_ep_len: int = 1000, 
             save_freq: int = 10, log_freq: int = 1, render: bool = False) -> Dict[str, List[float]]:
        """
        PPO 학습 수행
        
        매개변수:
            env: 환경 인스턴스
            epochs (int): 학습 에폭 수
            steps_per_epoch (int): 에폭당 환경 스텝 수
            max_ep_len (int): 최대 에피소드 길이
            save_freq (int): 모델 저장 주기
            log_freq (int): 로깅 주기
            render (bool): 렌더링 여부
            
        반환값:
            Dict[str, List[float]]: 학습 메트릭
        """
        # 에폭 반복
        for epoch in range(epochs):
            # 배치 데이터 수집
            episode_returns = []
            episode_lengths = []
            
            state, _ = env.reset()
            episode_return = 0
            episode_length = 0
            
            for t in range(steps_per_epoch):
                if render:
                    env.render()
                
                # 액션 선택
                action, logp, value = self.select_action(state)
                
                # 환경에서 한 스텝 진행
                next_state, reward, done, truncated, info = env.step(action)
                
                # 트랜지션 저장
                self.store_transition(state, action, reward, value, logp)
                
                # 상태 업데이트
                state = next_state
                episode_return += reward
                episode_length += 1
                
                # 종료 처리
                timeout = episode_length == max_ep_len
                terminal = done or timeout
                epoch_ended = t == steps_per_epoch - 1
                
                if terminal or epoch_ended:
                    # 에피소드 완료 또는 중단됨
                    if epoch_ended and not terminal:
                        # 에폭 종료, 에피소드는 계속
                        # 다음 상태의 가치 추정
                        _, _, last_value = self.select_action(state)
                    else:
                        # 에피소드 종료
                        last_value = 0
                    
                    # 경로 완료 처리
                    self.finish_episode(last_value)
                    
                    if terminal:
                        # 에피소드 완료
                        episode_returns.append(episode_return)
                        episode_lengths.append(episode_length)
                    
                    # 새 에피소드 시작
                    state, _ = env.reset()
                    episode_return = 0
                    episode_length = 0
            
            # 메트릭 저장
            self.metrics["episode_returns"].append(np.mean(episode_returns) if episode_returns else 0)
            self.metrics["episode_lengths"].append(np.mean(episode_lengths) if episode_lengths else 0)
            
            # 정책 업데이트
            update_metrics = self.update()
            
            # 로깅
            if (epoch + 1) % log_freq == 0:
                print(f"Epoch: {epoch + 1}/{epochs}")
                print(f"Mean Return: {self.metrics['episode_returns'][-1]:.2f}")
                print(f"Mean Length: {self.metrics['episode_lengths'][-1]:.2f}")
                print(f"Actor Loss: {update_metrics['actor_loss']:.4f}")
                print(f"Critic Loss: {update_metrics['critic_loss']:.4f}")
                print(f"Entropy: {update_metrics['entropy']:.4f}")
                print(f"KL Divergence: {update_metrics['kl']:.4f}")
                print("-" * 50)
            
            # 모델 저장
            if (epoch + 1) % save_freq == 0:
                save_path = os.path.join(self.log_dir, f"model_epoch_{epoch + 1}")
                self.save_model(save_path)
        
        # 최종 모델 저장
        self.save_model(os.path.join(self.log_dir, "model_final"))
        
        return self.metrics
    
    def plot_training_curves(self, figsize: Tuple[int, int] = (15, 10)):
        """
        학습 곡선 시각화
        
        매개변수:
            figsize (Tuple[int, int]): 그림 크기
        """
        plt.figure(figsize=figsize)
        
        # 에피소드 리턴 그래프
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics["episode_returns"])
        plt.title('Episode Returns')
        plt.xlabel('Epoch')
        plt.ylabel('Return')
        plt.grid(True, alpha=0.3)
        
        # 에피소드 길이 그래프
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics["episode_lengths"])
        plt.title('Episode Lengths')
        plt.xlabel('Epoch')
        plt.ylabel('Length')
        plt.grid(True, alpha=0.3)
        
        # 액터/크리틱 손실 그래프
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics["actor_loss"], label='Actor Loss')
        plt.plot(self.metrics["critic_loss"], label='Critic Loss')
        plt.title('Network Losses')
        plt.xlabel('Update')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 엔트로피/KL 그래프
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics["entropy"], label='Entropy')
        plt.plot(self.metrics["kl"], label='KL Divergence')
        plt.title('Entropy & KL Divergence')
        plt.xlabel('Update')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "training_curves.png"))
        plt.show()
    
    def save_model(self, path: str):
        """
        모델 저장
        
        매개변수:
            path (str): 저장 경로
        """
        self.ac_model.save_model(path)
        
        # 에이전트 설정 저장
        agent_config = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_bounds": self.action_bounds,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "train_iters": self.train_iters
        }
        
        with open(os.path.join(path, "agent_config.pkl"), 'wb') as f:
            pickle.dump(agent_config, f)
        
        # 메트릭 저장
        with open(os.path.join(path, "metrics.pkl"), 'wb') as f:
            pickle.dump(self.metrics, f)
    
    def load_model(self, path: str):
        """
        모델 로드
        
        매개변수:
            path (str): 로드 경로
        """
        self.ac_model.load_model(path)
        
        # 에이전트 설정 로드
        with open(os.path.join(path, "agent_config.pkl"), 'rb') as f:
            agent_config = pickle.load(f)
            
        self.state_dim = agent_config["state_dim"]
        self.action_dim = agent_config["action_dim"]
        self.action_bounds = agent_config["action_bounds"]
        self.buffer_size = agent_config["buffer_size"]
        self.batch_size = agent_config["batch_size"]
        self.train_iters = agent_config["train_iters"]
        
        # 메트릭 로드 (있는 경우)
        metrics_path = os.path.join(path, "metrics.pkl")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                self.metrics = pickle.load(f)


# 예시 사용법
if __name__ == "__main__":
    # 간단한 환경 생성 (OpenAI Gym)
    import gym
    
    # 환경 생성
    env = gym.make('Pendulum-v1')
    
    # 환경 정보
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 액션 범위
    action_bounds = {
        "action": (env.action_space.low[0], env.action_space.high[0])
    }
    
    # PPO 에이전트 생성
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bounds=action_bounds,
        buffer_size=2048,
        batch_size=64,
        gamma=0.99,
        lam=0.95,
        actor_hidden_sizes=[64, 64],
        critic_hidden_sizes=[64, 64],
        learning_rate=3e-4,
        clip_ratio=0.2,
        target_kl=0.01,
        value_coef=0.5,
        entropy_coef=0.01,
        train_iters=80,
        log_dir="./logs/ppo_pendulum"
    )
    
    # 학습 (간략한 설정)
    metrics = agent.train(
        env=env,
        epochs=10,
        steps_per_epoch=2048,
        max_ep_len=200,
        save_freq=5,
        log_freq=1,
        render=False
    )
    
    # 학습 곡선 시각화
    agent.plot_training_curves()
    
    # 학습된 모델 저장
    agent.save_model("./models/ppo_pendulum_final")
    
    # 환경 종료
    env.close()