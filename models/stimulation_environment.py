"""
신경자극 환경 모듈

이 모듈은 강화학습을 위한 신경자극 환경을 구현합니다.
상태 공간, 행동 공간, 보상 함수 등을 정의하고 에이전트와 환경 간의 상호작용을 관리합니다.
신경재생 단계별로 최적화된 파라미터 범위를 지원합니다.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class StimulationEnvironment:
    """강화학습 기반 신경자극 최적화를 위한 환경"""
    
    def __init__(self, state_size: int = 5, 
                amplitude_levels: int = 5, 
                frequency_levels: int = 5, 
                pulse_width_levels: int = 5,
                frequency_range: Optional[Tuple[float, float]] = None,
                amplitude_range: Optional[Tuple[float, float]] = None,
                regeneration_stage: str = 'acute'):
        """
        StimulationEnvironment 초기화
        
        매개변수:
            state_size (int): 상태 공간의 차원 수
            amplitude_levels (int): 자극 진폭 수준 수
            frequency_levels (int): 자극 주파수 수준 수
            pulse_width_levels (int): 자극 펄스 폭 수준 수
            frequency_range (Optional[Tuple[float, float]]): 주파수 범위 (Hz)
            amplitude_range (Optional[Tuple[float, float]]): 진폭 범위 (mA)
            regeneration_stage (str): 신경재생 단계 ('acute', 'subacute', 'regeneration', 'remodeling')
        """
        self.state_size = state_size
        self.amplitude_levels = amplitude_levels
        self.frequency_levels = frequency_levels
        self.pulse_width_levels = pulse_width_levels
        self.regeneration_stage = regeneration_stage
        
        # 전체 행동 공간 크기
        self.action_size = amplitude_levels * frequency_levels * pulse_width_levels
        
        # 신경재생 단계별 기본 파라미터 범위
        self.stage_defaults = {
            'acute': {
                'amplitude_range': (0.1, 1.0),  # mA
                'frequency_range': (10.0, 50.0),  # Hz
                'pulse_width_range': (100.0, 300.0)  # μs
            },
            'subacute': {
                'amplitude_range': (0.5, 2.0),  # mA
                'frequency_range': (20.0, 100.0),  # Hz (BDNF 발현 최적화)
                'pulse_width_range': (200.0, 400.0)  # μs
            },
            'regeneration': {
                'amplitude_range': (1.0, 3.0),  # mA
                'frequency_range': (50.0, 100.0),  # Hz (cAMP/PKA/CREB 경로)
                'pulse_width_range': (300.0, 500.0)  # μs
            },
            'remodeling': {
                'amplitude_range': (0.5, 2.5),  # mA
                'frequency_range': (20.0, 200.0),  # Hz (시냅스 가소성)
                'pulse_width_range': (250.0, 450.0)  # μs
            }
        }
        
        # 각 매개변수의 경계값 설정
        stage_params = self.stage_defaults.get(regeneration_stage, self.stage_defaults['acute'])
        self.amplitude_range = amplitude_range or stage_params['amplitude_range']
        self.frequency_range = frequency_range or stage_params['frequency_range']
        self.pulse_width_range = stage_params['pulse_width_range']
        
        # 신경재생 메커니즘별 가중치
        self.mechanism_weights = {
            'acute': {
                'anti_inflammatory': 0.4,
                'neuroprotection': 0.3,
                'blood_flow': 0.2,
                'pain_reduction': 0.1
            },
            'subacute': {
                'neurotrophic_factors': 0.35,  # BDNF, GDNF
                'schwann_cell_activation': 0.25,
                'angiogenesis': 0.2,
                'debris_clearance': 0.2
            },
            'regeneration': {
                'axon_growth': 0.3,
                'myelination': 0.25,
                'camp_signaling': 0.25,  # cAMP/PKA/CREB
                'gap43_expression': 0.2
            },
            'remodeling': {
                'synaptic_plasticity': 0.3,
                'functional_integration': 0.25,
                'sensory_motor_coupling': 0.25,
                'circuit_refinement': 0.2
            }
        }
        
        # 현재 상태 및 자극 매개변수 초기화
        self.reset()
        
        # 환경 내 피드백 함수 (기본적으로는 가상 함수)
        self.feedback_function = self._default_feedback
        
        # 보상 히스토리
        self.reward_history = []
        
        # 목표 응답 (시뮬레이션 또는 실제 측정값)
        self.target_response = None
        
        logger.info(f"StimulationEnvironment 초기화 완료 - 재생 단계: {regeneration_stage}")
    
    def reset(self) -> np.ndarray:
        """
        환경 초기화 및 초기 상태 반환
        
        반환값:
            np.ndarray: 초기 상태
        """
        # 재생 단계에 맞는 생리학적 상태 초기화
        if self.regeneration_stage == 'acute':
            # 급성기: 높은 염증 지표, 낮은 신경 활성
            self.current_state = np.array([
                np.random.uniform(0.7, 0.9),  # 염증 지표
                np.random.uniform(0.1, 0.3),  # 신경 활성
                np.random.uniform(0.3, 0.5),  # 혈류
                np.random.uniform(0.6, 0.8),  # 통증 수준
                np.random.uniform(0.2, 0.4)   # 조직 손상
            ])[:self.state_size]
        elif self.regeneration_stage == 'subacute':
            # 아급성기: 중간 염증, 신경영양인자 활성 시작
            self.current_state = np.array([
                np.random.uniform(0.4, 0.6),  # 염증 지표
                np.random.uniform(0.3, 0.5),  # 신경영양인자
                np.random.uniform(0.2, 0.4),  # 슈반세포 활성
                np.random.uniform(0.4, 0.6),  # 혈관신생
                np.random.uniform(0.3, 0.5)   # 조직 재생
            ])[:self.state_size]
        elif self.regeneration_stage == 'regeneration':
            # 재생기: 낮은 염증, 높은 축삭 성장
            self.current_state = np.array([
                np.random.uniform(0.1, 0.3),  # 염증 지표
                np.random.uniform(0.5, 0.7),  # 축삭 성장
                np.random.uniform(0.4, 0.6),  # 수초화
                np.random.uniform(0.5, 0.7),  # cAMP 수준
                np.random.uniform(0.4, 0.6)   # GAP-43 발현
            ])[:self.state_size]
        else:  # remodeling
            # 재조직화기: 시냅스 형성, 기능 통합
            self.current_state = np.array([
                np.random.uniform(0.0, 0.2),  # 염증 지표
                np.random.uniform(0.6, 0.8),  # 시냅스 밀도
                np.random.uniform(0.5, 0.7),  # 기능 연결성
                np.random.uniform(0.4, 0.6),  # 감각운동 협응
                np.random.uniform(0.5, 0.7)   # 회로 성숙도
            ])[:self.state_size]
        
        # 자극 매개변수 초기화
        self.current_amplitude_idx = self.amplitude_levels // 2
        self.current_frequency_idx = self.frequency_levels // 2
        self.current_pulse_width_idx = self.pulse_width_levels // 2
        
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
        
        # 자극에 대한 응답 생성 (재생 단계별 메커니즘 고려)
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
            'total_reward': self.total_reward,
            'regeneration_stage': self.regeneration_stage
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
        if levels == 1:
            return (min_val + max_val) / 2
        return min_val + (idx / (levels - 1)) * (max_val - min_val)
    
    def _apply_stimulation(self, amplitude: float, frequency: float, pulse_width: float) -> None:
        """
        자극 적용 및 신경 상태 업데이트 (신경재생 메커니즘 기반)
        
        매개변수:
            amplitude (float): 자극 진폭 (mA)
            frequency (float): 자극 주파수 (Hz)
            pulse_width (float): 자극 펄스 폭 (μs)
        """
        # 재생 단계별 메커니즘 가중치 가져오기
        weights = self.mechanism_weights[self.regeneration_stage]
        
        # 자극 효과 계산 (재생 단계별로 다른 메커니즘)
        transition_effect = np.zeros(self.state_size)
        
        if self.regeneration_stage == 'acute':
            # 급성기: 항염증, 신경보호, 혈류 개선 중심
            transition_effect[0] = -0.3 * weights['anti_inflammatory'] * amplitude / self.amplitude_range[1]
            transition_effect[1] = 0.2 * weights['neuroprotection'] * frequency / 50  # 50Hz 근처 최적
            transition_effect[2] = 0.25 * weights['blood_flow'] * pulse_width / 300
            transition_effect[3] = -0.2 * weights['pain_reduction'] * frequency / 100  # 고주파 통증 억제
            
        elif self.regeneration_stage == 'subacute':
            # 아급성기: 신경영양인자, 슈반세포 활성화 중심
            # BDNF는 20Hz에서 최적
            bdnf_effect = np.exp(-((frequency - 20) / 30)**2)
            transition_effect[0] = -0.15 * amplitude / self.amplitude_range[1]  # 염증 감소
            transition_effect[1] = 0.4 * weights['neurotrophic_factors'] * bdnf_effect
            transition_effect[2] = 0.3 * weights['schwann_cell_activation'] * (frequency / 10)  # 저주파 선호
            transition_effect[3] = 0.25 * weights['angiogenesis'] * amplitude / self.amplitude_range[1]
            
        elif self.regeneration_stage == 'regeneration':
            # 재생기: 축삭 성장, cAMP 신호전달 중심
            # cAMP는 50Hz에서 최적
            camp_effect = np.exp(-((frequency - 50) / 40)**2)
            transition_effect[0] = -0.05 * amplitude / self.amplitude_range[1]  # 염증 최소
            transition_effect[1] = 0.35 * weights['axon_growth'] * camp_effect
            transition_effect[2] = 0.3 * weights['myelination'] * amplitude / self.amplitude_range[1]
            transition_effect[3] = 0.35 * weights['camp_signaling'] * camp_effect
            transition_effect[4 % self.state_size] = 0.25 * weights['gap43_expression'] * frequency / 100
            
        else:  # remodeling
            # 재조직화기: 시냅스 가소성, 기능 통합 중심
            # 다양한 주파수 필요
            plasticity_effect = 0.3 + 0.7 * np.sin(frequency / 50)  # 주파수 변화 중요
            transition_effect[1] = 0.3 * weights['synaptic_plasticity'] * plasticity_effect
            transition_effect[2] = 0.25 * weights['functional_integration'] * amplitude / self.amplitude_range[1]
            transition_effect[3] = 0.25 * weights['sensory_motor_coupling'] * pulse_width / 400
            transition_effect[4 % self.state_size] = 0.2 * weights['circuit_refinement'] * frequency / 100
        
        # 에너지 소비 고려 (과도한 자극 방지)
        energy = amplitude * pulse_width * frequency / 10000
        energy_penalty = -0.1 * np.tanh(energy - 0.5)  # 에너지가 너무 높으면 패널티
        transition_effect = transition_effect * (1 + energy_penalty)
        
        # 약간의 랜덤성 추가 (생물학적 변동성)
        noise = np.random.normal(0, 0.02, size=self.state_size)
        
        # 상태 업데이트
        self.current_state = np.clip(
            self.current_state * 0.85 + transition_effect + noise,
            0.0, 1.0  # 상태를 0-1 범위로 제한
        )
    
    def _calculate_reward(self) -> float:
        """
        현재 상태에 대한 보상 계산 (재생 단계별 목표 고려)
        
        반환값:
            float: 보상 값
        """
        # 기본적으로는 피드백 함수 사용
        base_reward = self.feedback_function(self.current_state)
        
        # 재생 단계별 추가 보상/패널티
        stage_bonus = 0.0
        
        if self.regeneration_stage == 'acute':
            # 급성기: 낮은 염증, 높은 신경보호가 목표
            if self.current_state[0] < 0.3:  # 낮은 염증
                stage_bonus += 0.2
            if self.current_state[1] > 0.5:  # 높은 신경 활성
                stage_bonus += 0.1
                
        elif self.regeneration_stage == 'subacute':
            # 아급성기: 신경영양인자와 슈반세포 활성화가 목표
            if self.current_state[1] > 0.6:  # 높은 신경영양인자
                stage_bonus += 0.25
            if self.current_state[2] > 0.5:  # 슈반세포 활성
                stage_bonus += 0.15
                
        elif self.regeneration_stage == 'regeneration':
            # 재생기: 축삭 성장과 수초화가 목표
            if self.current_state[1] > 0.7:  # 높은 축삭 성장
                stage_bonus += 0.3
            if self.current_state[2] > 0.6:  # 수초화
                stage_bonus += 0.2
                
        else:  # remodeling
            # 재조직화기: 시냅스 형성과 기능 통합이 목표
            if self.current_state[1] > 0.7:  # 높은 시냅스 밀도
                stage_bonus += 0.25
            if self.current_state[2] > 0.6:  # 기능 연결성
                stage_bonus += 0.25
        
        return base_reward + stage_bonus
    
    def _default_feedback(self, state: np.ndarray) -> float:
        """
        기본 피드백 함수 (재생 단계별 목표 상태 기반)
        
        매개변수:
            state (np.ndarray): 현재 상태
            
        반환값:
            float: 피드백 값 (보상)
        """
        # 재생 단계별 목표 상태
        target_states = {
            'acute': np.array([0.2, 0.6, 0.7, 0.3, 0.3]),  # 낮은 염증, 높은 보호
            'subacute': np.array([0.3, 0.7, 0.6, 0.6, 0.5]),  # 신경영양인자 활성
            'regeneration': np.array([0.1, 0.8, 0.7, 0.7, 0.6]),  # 축삭 성장
            'remodeling': np.array([0.1, 0.8, 0.7, 0.6, 0.7])  # 시냅스 형성
        }
        
        target = target_states[self.regeneration_stage][:self.state_size]
        
        # 타겟과의 거리 계산
        distance = np.linalg.norm(state - target)
        
        # 거리가 작을수록 보상이 큼
        reward = np.exp(-2 * distance**2)
        
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
        상태 공간 설명 반환 (재생 단계별)
        
        반환값:
            Dict[str, str]: 각 상태 차원의 설명
        """
        descriptions = {
            'acute': {
                0: "염증 지표 (Inflammation level)",
                1: "신경 활성도 (Neural activity)",
                2: "혈류량 (Blood flow)",
                3: "통증 수준 (Pain level)",
                4: "조직 손상도 (Tissue damage)"
            },
            'subacute': {
                0: "염증 지표 (Inflammation level)",
                1: "신경영양인자 (Neurotrophic factors)",
                2: "슈반세포 활성 (Schwann cell activity)",
                3: "혈관신생 (Angiogenesis)",
                4: "조직 재생 (Tissue regeneration)"
            },
            'regeneration': {
                0: "염증 지표 (Inflammation level)",
                1: "축삭 성장 (Axon growth)",
                2: "수초화 정도 (Myelination)",
                3: "cAMP 수준 (cAMP level)",
                4: "GAP-43 발현 (GAP-43 expression)"
            },
            'remodeling': {
                0: "염증 지표 (Inflammation level)",
                1: "시냅스 밀도 (Synaptic density)",
                2: "기능 연결성 (Functional connectivity)",
                3: "감각운동 협응 (Sensorimotor coupling)",
                4: "회로 성숙도 (Circuit maturity)"
            }
        }
        
        stage_desc = descriptions[self.regeneration_stage]
        return {i: stage_desc.get(i, f"상태 {i}") for i in range(self.state_size)}
    
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
    
    def set_regeneration_stage(self, stage: str) -> None:
        """
        신경재생 단계 변경
        
        매개변수:
            stage (str): 재생 단계 ('acute', 'subacute', 'regeneration', 'remodeling')
        """
        if stage not in self.stage_defaults:
            raise ValueError(f"유효하지 않은 재생 단계: {stage}")
        
        self.regeneration_stage = stage
        
        # 파라미터 범위 업데이트
        stage_params = self.stage_defaults[stage]
        self.amplitude_range = stage_params['amplitude_range']
        self.frequency_range = stage_params['frequency_range']
        self.pulse_width_range = stage_params['pulse_width_range']
        
        logger.info(f"재생 단계가 '{stage}'로 변경되었습니다.")
