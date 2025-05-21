"""
신경자극 환경 모듈

이 모듈은 강화학습을 위한 신경자극 환경을 구현합니다.
상태 공간, 행동 공간, 보상 함수 등을 정의하고 에이전트와 환경 간의 상호작용을 관리합니다.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

class StimulationEnvironment:
    """강화학습 기반 신경자극 최적화를 위한 환경"""
    
    def __init__(self, state_size: int = 5, 
                amplitude_levels: int = 5, 
                frequency_levels: int = 5, 
                pulse_width_levels: int = 5):
        """
        StimulationEnvironment 초기화
        
        매개변수:
            state_size (int): 상태 공간의 차원 수
            amplitude_levels (int): 자극 진폭 수준 수
            frequency_levels (int): 자극 주파수 수준 수
            pulse_width_levels (int): 자극 펄스 폭 수준 수
        """
        self.state_size = state_size
        self.amplitude_levels = amplitude_levels
        self.frequency_levels = frequency_levels
        self.pulse_width_levels = pulse_width_levels
        
        # 전체 행동 공간 크기
        self.action_size = amplitude_levels * frequency_levels * pulse_width_levels
        
        # 각 매개변수의 경계값 설정
        self.amplitude_range = (0.1, 10.0)  # mA
        self.frequency_range = (1.0, 300.0)  # Hz
        self.pulse_width_range = (10.0, 500.0)  # μs
        
        # 현재 상태 및 자극 매개변수 초기화
        self.reset()
        
        # 환경 내 피드백 함수 (기본적으로는 가상 함수)
        self.feedback_function = self._default_feedback
        
        # 보상 히스토리
        self.reward_history = []
        
        # 목표 응답 (시뮬레이션 또는 실제 측정값)
        self.target_response = None
    
    def reset(self) -> np.ndarray:
        """
        환경 초기화 및 초기 상태 반환
        
        반환값:
            np.ndarray: 초기 상태
        """
        # 임의의 생리학적 상태 초기화
        self.current_state = np.random.normal(0, 1, size=self.state_size)
        
        # 자극 매개변수 초기화
        self.current_amplitude_idx = 0
        self.current_frequency_idx = 0
        self.current_pulse_width_idx = 0
        
        # 에피소드 진행 상태 초기화
        self.steps = 0
        self.done = False
        self.total_reward = 0.0
        
        return self.current_state.reshape(1, -1)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        환경에서 한 스텝 진행
        
        매개변수:
            action (int): 수행할 행동 (자극 매개변수 조합의 인덱스)
            
        반환값:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: 
                (다음 상태, 보상, 완료 여부, 추가 정보)
        """
        # 행동을 자극 매개변수 인덱스로 변환
        self._decode_action(action)
        
        # 자극 매개변수 값 계산
        amplitude = self._idx_to_value(self.current_amplitude_idx, 
                                       self.amplitude_levels, 
                                       self.amplitude_range)
        frequency = self._idx_to_value(self.current_frequency_idx, 
                                      self.frequency_levels, 
                                      self.frequency_range)
        pulse_width = self._idx_to_value(self.current_pulse_width_idx, 
                                        self.pulse_width_levels, 
                                        self.pulse_width_range)
        
        # 자극에 대한 가상 응답 생성 (실제로는 실시간 측정 데이터로 대체)
        self._apply_stimulation(amplitude, frequency, pulse_width)
        
        # 보상 계산
        reward = self._calculate_reward()
        
        # 히스토리 업데이트
        self.total_reward += reward
        self.steps += 1
        
        # 에피소드 종료 조건 확인
        if self.steps >= 100:  # 최대 스텝 수
            self.done = True
            
        # 보상 히스토리 업데이트
        self.reward_history.append(reward)
        
        # 추가 정보 딕셔너리
        info = {
            'amplitude': amplitude,
            'frequency': frequency,
            'pulse_width': pulse_width,
            'steps': self.steps,
            'total_reward': self.total_reward
        }
        
        return self.current_state.reshape(1, -1), reward, self.done, info
    
    def _decode_action(self, action: int) -> None:
        """
        행동 인덱스를 자극 매개변수 인덱스로 변환
        
        매개변수:
            action (int): 행동 인덱스
        """
        # 3차원 행동 공간에서의 인덱스 디코딩
        pw_levels = self.pulse_width_levels
        freq_levels = self.frequency_levels
        
        self.current_pulse_width_idx = action % pw_levels
        action //= pw_levels
        
        self.current_frequency_idx = action % freq_levels
        action //= freq_levels
        
        self.current_amplitude_idx = action
    
    def _idx_to_value(self, idx: int, levels: int, value_range: Tuple[float, float]) -> float:
        """
        인덱스를 실제 값으로 변환
        
        매개변수:
            idx (int): 매개변수 인덱스
            levels (int): 가능한 수준 수
            value_range (Tuple[float, float]): 값 범위
            
        반환값:
            float: 실제 매개변수 값
        """
        min_val, max_val = value_range
        return min_val + (idx / (levels - 1)) * (max_val - min_val)
    
    def _apply_stimulation(self, amplitude: float, frequency: float, pulse_width: float) -> None:
        """
        자극 적용 및 신경 상태 업데이트 (시뮬레이션)
        
        매개변수:
            amplitude (float): 자극 진폭 (mA)
            frequency (float): 자극 주파수 (Hz)
            pulse_width (float): 자극 펄스 폭 (μs)
        """
        # 자극 효과 시뮬레이션 (실제로는 실시간 신경 반응 측정으로 대체)
        
        # 간단한 상태 전이 모델
        transition_effect = np.zeros(self.state_size)
        
        # 자극 진폭 효과 (첫 번째 상태 값에 영향)
        transition_effect[0] = 0.2 * amplitude / self.amplitude_range[1]
        
        # 자극 주파수 효과 (두 번째 상태 값에 영향)
        transition_effect[1] = 0.15 * frequency / self.frequency_range[1]
        
        # 자극 펄스 폭 효과 (세 번째 상태 값에 영향)
        transition_effect[2] = 0.1 * pulse_width / self.pulse_width_range[1]
        
        # 시너지 효과 (네 번째 상태 값에 영향)
        synergy = 0.05 * (amplitude * frequency * pulse_width) / (
            self.amplitude_range[1] * self.frequency_range[1] * self.pulse_width_range[1])
        transition_effect[3] = synergy
        
        # 전체 에너지 (마지막 상태 값에 영향)
        energy = 0.1 * amplitude * pulse_width * frequency / 1000
        transition_effect[4 % self.state_size] = energy
        
        # 약간의 랜덤성 추가 (측정 노이즈 또는 생물학적 변동성 모사)
        noise = np.random.normal(0, 0.05, size=self.state_size)
        
        # 상태 업데이트
        self.current_state = self.current_state * 0.8 + transition_effect + noise
    
    def _calculate_reward(self) -> float:
        """
        현재 상태에 대한 보상 계산
        
        반환값:
            float: 보상 값
        """
        # 기본적으로는 피드백 함수 사용
        return self.feedback_function(self.current_state)
    
    def _default_feedback(self, state: np.ndarray) -> float:
        """
        기본 피드백 함수 (정규 분포 기반)
        
        매개변수:
            state (np.ndarray): 현재 상태
            
        반환값:
            float: 피드백 값 (보상)
        """
        # 단순한 가우시안 보상 함수 (상태[0]와 상태[1]은 적당한 값을 가질 때 최대 보상)
        target = np.array([0.5, 0.4, 0.3, 0.2, 0.1])[:self.state_size]
        
        # 타겟과의 거리 계산
        distance = np.linalg.norm(state - target)
        
        # 거리가 작을수록 보상이 큼
        reward = np.exp(-distance**2)
        
        return reward
    
    def set_feedback_function(self, function):
        """
        사용자 정의 피드백 함수 설정
        
        매개변수:
            function: 상태를 입력으로 하고 보상을 반환하는 함수
        """
        self.feedback_function = function
    
    def set_target_response(self, target: np.ndarray) -> None:
        """
        목표 응답 설정
        
        매개변수:
            target (np.ndarray): 목표 신경 응답
        """
        self.target_response = target
        
        # 목표 기반 피드백 함수로 업데이트
        self.set_feedback_function(lambda state: -np.linalg.norm(state - self.target_response))
    
    def get_reward_history(self) -> List[float]:
        """
        보상 히스토리 반환
        
        반환값:
            List[float]: 보상 히스토리
        """
        return self.reward_history
    
    def get_state_description(self) -> Dict[str, str]:
        """
        상태 공간 설명 반환
        
        반환값:
            Dict[str, str]: 각 상태 차원의 설명
        """
        descriptions = {
            0: "진폭 반응 (amplitude response)",
            1: "주파수 반응 (frequency response)",
            2: "펄스 폭 반응 (pulse width response)",
            3: "시너지 효과 (synergy effect)",
            4: "에너지 소비 (energy consumption)"
        }
        
        return {i: descriptions.get(i, f"상태 {i}") for i in range(self.state_size)}
    
    def get_action_description(self, action: int) -> Dict[str, float]:
        """
        행동에 대한 설명 반환
        
        매개변수:
            action (int): 행동 인덱스
            
        반환값:
            Dict[str, float]: 행동에 해당하는 자극 매개변수
        """
        # 행동 디코딩
        self._decode_action(action)
        
        # 매개변수 값 계산
        amplitude = self._idx_to_value(self.current_amplitude_idx, 
                                     self.amplitude_levels, 
                                     self.amplitude_range)
        frequency = self._idx_to_value(self.current_frequency_idx, 
                                      self.frequency_levels, 
                                      self.frequency_range)
        pulse_width = self._idx_to_value(self.current_pulse_width_idx, 
                                        self.pulse_width_levels, 
                                        self.pulse_width_range)
        
        return {
            'amplitude': amplitude,
            'frequency': frequency,
            'pulse_width': pulse_width
        }
