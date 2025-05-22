"""
적응형 신경 자극 시스템 (Adaptive Neural Stimulation System)

이 모듈은 적응형 신경 자극 시스템의 메인 진입점입니다.
강화학습 기반 자극 최적화, 신경 신호 처리, 시각화 등의 기능을 통합합니다.

주요 기능:
    - 신경 신호 데이터 처리: 필터링, 특성 추출, 분석
    - 강화학습 기반 자극 최적화: DQN을 통한 최적 자극 매개변수 학습
    - LSTM 기반 신호 예측: 시계열 신경 신호 패턴 예측
    - 자극 패턴 생성 및 제어: 다양한 자극 파형 생성 및 매개변수 제어
    - 데이터 시각화: 신호, 특성, 최적화 과정 시각화

사용 예시:
    ```python
    # 적응형 신경 자극 시스템 기본 사용법
    
    # 1. 시스템 인스턴스 생성
    system = AdaptiveStimulationSystem()
    
    # 2. 신경 신호 데이터 로드
    data, sr = system.load_data('neural_data.csv')
    
    # 3. 데이터 전처리
    processed_data = system.preprocess_data(data)
    
    # 4. DQN 에이전트 학습
    rewards = system.train_dqn(num_episodes=100)
    
    # 5. 적응형 자극 수행
    result = system.adaptive_stimulation(processed_data, duration=5.0)
    
    # 6. 결과 시각화
    system.visualize_results({
        'signal': processed_data,
        'stimulation_waveform': result['stimulation_waveform'],
        'rewards': rewards
    })
    ```

참고 자료:
    - Gordon, T., & English, A. W. (2016). Strategies to promote peripheral nerve regeneration:
      electrical stimulation and/or exercise. European Journal of Neuroscience, 43(3), 336-350.
    - Yao, L., et al. (2018). Electrical stimulation optimizes the polarization of cortical
      neurons and enhances motor function recovery after spinal cord injury. Neural
      Regeneration Research, 13(12), 2112-2119.
    - Chen, L., et al. (2020). Deep learning based online predictions of neural responses
      to electrical stimulation for data-driven neuromodulation. Scientific Reports, 10(1), 1-10.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import logging
import warnings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TensorFlow import 처리
try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow를 찾을 수 없습니다. LSTM 기능이 제한될 수 있습니다.")
    # 더미 History 클래스 정의
    class History:
        pass

import sys
sys.path.append('.')

from utils.signal_processor import SignalProcessor
from utils.data_handler import DataLoader, DataTransformer
from utils.stimulation_controller import StimulationController
from utils.parameter_optimizer import ParameterOptimizer
from utils.visualizer import Visualizer

from models.dqn_agent import DQNAgent
from models.lstm_model import LSTMModel, BidirectionalLSTMModel
from models.stimulation_environment import StimulationEnvironment

class AdaptiveStimulationSystem:
    """
    적응형 신경 자극 시스템의 핵심 클래스
    
    이 클래스는 신경 재생을 위한 전기자극 시스템의 모든 구성 요소를 통합합니다.
    신호 처리, 자극 최적화, 시각화 등의 기능을 제공합니다.
    
    Attributes:
        config (Dict[str, Any]): 시스템 설정 매개변수
        signal_processor (SignalProcessor): 신호 처리 유틸리티
        data_loader (DataLoader): 데이터 로드 유틸리티
        data_transformer (DataTransformer): 데이터 변환 유틸리티
        stimulation_controller (StimulationController): 자극 제어 유틸리티
        parameter_optimizer (ParameterOptimizer): 매개변수 최적화 유틸리티
        visualizer (Visualizer): 데이터 시각화 유틸리티
        environment (Optional[StimulationEnvironment]): 강화학습 환경
        agent (Optional[DQNAgent]): DQN 강화학습 에이전트
        lstm_model (Optional[LSTMModel]): LSTM 기반 시계열 예측 모델
        training_data (Dict[str, List]): 학습 데이터 저장
        current_state (Optional[np.ndarray]): 현재 시스템 상태
        is_learning (bool): 학습 모드 여부
        episode_count (int): 현재 에피소드 카운트
        step_count (int): 현재 스텝 카운트
        neural_regeneration_params (Dict[str, Any]): 신경재생 관련 파라미터
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        AdaptiveStimulationSystem 초기화
        
        Args:
            config (Dict[str, Any], optional): 시스템 설정 매개변수. 기본값은 None (기본 설정 사용).
                
                가능한 설정 키:
                - 'sampling_rate': 샘플링 레이트 (Hz)
                - 'sequence_length': LSTM 시퀀스 길이
                - 'feature_dim': 특성 차원 수
                - 'use_lstm': LSTM 모델 사용 여부
                - 'use_reinforcement_learning': 강화학습 사용 여부
                - 'save_path': 결과 저장 경로
                - 'model_path': 모델 저장 경로
                - 'regeneration_stage': 신경재생 단계 ('acute', 'subacute', 'regeneration', 'remodeling')
        
        Raises:
            ValueError: 설정 값에 오류가 있는 경우 발생
            OSError: 디렉토리 생성 오류 시 발생
        """
        # 기본 설정 값
        self.config = {
            'sampling_rate': 1000.0,  # Hz
            'sequence_length': 100,
            'feature_dim': 5,
            'use_lstm': True,
            'use_reinforcement_learning': True,
            'save_path': 'results',
            'model_path': 'models/saved',
            'regeneration_stage': 'acute'  # 신경재생 단계
        }
        
        # 신경재생 메커니즘 관련 파라미터
        self.neural_regeneration_params = {
            'acute': {  # 급성기 (0-3일)
                'frequency_range': (10, 50),  # Hz
                'amplitude_range': (0.1, 1.0),  # mA
                'pulse_width': 200,  # μs
                'primary_mechanism': 'anti_inflammatory',
                'target_factors': ['TNF-α 억제', 'IL-10 증가']
            },
            'subacute': {  # 아급성기 (4-14일)
                'frequency_range': (20, 100),  # Hz, BDNF 발현 최적화
                'amplitude_range': (0.5, 2.0),  # mA
                'pulse_width': 300,  # μs
                'primary_mechanism': 'neurotrophic_factor_induction',
                'target_factors': ['BDNF 발현', 'GDNF 발현', '슈반세포 활성화']
            },
            'regeneration': {  # 재생기 (14-60일)
                'frequency_range': (50, 100),  # Hz, cAMP/PKA/CREB 경로 활성화
                'amplitude_range': (1.0, 3.0),  # mA
                'pulse_width': 400,  # μs
                'primary_mechanism': 'axon_growth_acceleration',
                'target_factors': ['GAP-43 발현', 'cAMP 증가', '축삭 성장']
            },
            'remodeling': {  # 재조직화기 (2-6개월)
                'frequency_range': (20, 200),  # Hz, 시냅스 가소성
                'amplitude_range': (0.5, 2.5),  # mA
                'pulse_width': 350,  # μs
                'primary_mechanism': 'synaptic_plasticity',
                'target_factors': ['시냅스 형성', '기능적 통합', '감각운동 협응']
            }
        }
        
        # 사용자 설정으로 업데이트
        if config:
            # 설정 유효성 검사
            self._validate_config(config)
            self.config.update(config)
            
        # 유틸리티 인스턴스 생성
        self.signal_processor = SignalProcessor(sampling_rate=self.config['sampling_rate'])
        self.data_loader = DataLoader()
        self.data_transformer = DataTransformer()
        self.stimulation_controller = StimulationController(sampling_rate=self.config['sampling_rate'])
        self.parameter_optimizer = ParameterOptimizer()
        self.visualizer = Visualizer()
        
        # 결과 저장 디렉토리 생성
        try:
            os.makedirs(self.config['save_path'], exist_ok=True)
            os.makedirs(self.config['model_path'], exist_ok=True)
        except OSError as e:
            raise OSError(f"디렉토리 생성 중 오류 발생: {e}")
        
        # 강화학습 환경 및 에이전트
        self.environment = None
        self.agent = None
        if self.config['use_reinforcement_learning']:
            self._initialize_reinforcement_learning()
            
        # LSTM 모델
        self.lstm_model = None
        if self.config['use_lstm'] and TF_AVAILABLE:
            self._initialize_lstm()
            
        # 학습 데이터
        self.training_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        
        # 시스템 상태
        self.current_state = None
        self.is_learning = False
        self.episode_count = 0
        self.step_count = 0
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """설정 값 유효성 검사"""
        if 'sampling_rate' in config and config['sampling_rate'] <= 0:
            raise ValueError("샘플링 레이트는 양수여야 합니다.")
        if 'sequence_length' in config and config['sequence_length'] <= 0:
            raise ValueError("시퀀스 길이는 양수여야 합니다.")
        if 'feature_dim' in config and config['feature_dim'] <= 0:
            raise ValueError("특성 차원 수는 양수여야 합니다.")
        if 'regeneration_stage' in config:
            valid_stages = ['acute', 'subacute', 'regeneration', 'remodeling']
            if config['regeneration_stage'] not in valid_stages:
                raise ValueError(f"유효하지 않은 재생 단계: {config['regeneration_stage']}. "
                               f"다음 중 하나를 선택하세요: {valid_stages}")
    
    def _initialize_reinforcement_learning(self) -> None:
        """강화학습 환경 및 에이전트 초기화"""
        # 재생 단계에 따른 파라미터 범위 설정
        stage_params = self.neural_regeneration_params[self.config['regeneration_stage']]
        
        self.environment = StimulationEnvironment(
            state_size=self.config['feature_dim'],
            amplitude_levels=5,
            frequency_levels=5,
            pulse_width_levels=5,
            frequency_range=stage_params['frequency_range'],
            amplitude_range=stage_params['amplitude_range']
        )
        self.agent = DQNAgent(
            state_size=self.config['feature_dim'],
            action_size=self.environment.action_size
        )
    
    def _initialize_lstm(self) -> None:
        """LSTM 모델 초기화"""
        self.lstm_model = LSTMModel(
            sequence_length=self.config['sequence_length'],
            feature_dim=self.config['feature_dim']
        )
        
    def load_data(self, file_path: str, **kwargs) -> Tuple[np.ndarray, float]:
        """
        신경 신호 데이터 로드
        
        다양한 형식의 신호 데이터 파일을 로드합니다.
        
        Args:
            file_path (str): 데이터 파일 경로
            **kwargs: 추가 매개변수. 파일 형식에 따라 필요한 매개변수가 다를 수 있음.
                - 'var_name': .mat 파일에서 로드할 변수 이름
                - 'dataset_name': HDF5 파일에서 로드할 데이터셋 이름
            
        Returns:
            Tuple[np.ndarray, float]: 로드된 데이터와 샘플링 레이트
            
        Raises:
            ValueError: 지원되지 않는 파일 형식이거나 파일 로드 중 오류 발생 시
            FileNotFoundError: 파일이 존재하지 않는 경우
        """
        # 파일 존재 확인
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
            
        data = None
        sampling_rate = self.config['sampling_rate']
        
        # 파일 확장자에 따라 적절한 로더 선택
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.csv':
                data, sampling_rate = self.data_loader.load_csv(file_path)
            elif ext == '.mat':
                data, sampling_rate = self.data_loader.load_mat(file_path, **kwargs)
            elif ext == '.npy':
                data, sampling_rate = self.data_loader.load_npy(file_path, sampling_rate)
            elif ext in ['.h5', '.hdf5']:
                data, sampling_rate = self.data_loader.load_hdf5(file_path, **kwargs)
            else:
                raise ValueError(f"지원되지 않는 파일 형식: {ext}")
        except Exception as e:
            raise ValueError(f"데이터 로드 중 오류 발생: {e}")
            
        # 설정 업데이트
        self.config['sampling_rate'] = sampling_rate
        self.signal_processor.sampling_rate = sampling_rate
        self.stimulation_controller.sampling_rate = sampling_rate
        
        logger.info(f"데이터 로드 완료: {data.shape}, 샘플링 레이트: {sampling_rate}Hz")
        
        return data, sampling_rate
    
    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        데이터 전처리
        
        원시 신호 데이터를 필터링, 노이즈 제거, 정규화하여 분석에 적합한 형태로 변환합니다.
        
        Args:
            data (np.ndarray): 전처리할 원본 데이터
            
        Returns:
            np.ndarray: 전처리된 데이터
            
        Raises:
            ValueError: 입력 데이터가 비어있거나 잘못된 형식인 경우
        """
        if data is None or data.size == 0:
            raise ValueError("전처리할 데이터가 비어 있습니다.")
            
        try:
            # 노이즈 필터링 (밴드패스 필터)
            filtered_data = self.signal_processor.bandpass_filter(data)
            
            # 노치 필터 적용 (60Hz)
            filtered_data = self.signal_processor.notch_filter(filtered_data)
            
            # 정규화
            normalized_data = self.data_transformer.normalize(filtered_data)
            
            return normalized_data
            
        except Exception as e:
            raise ValueError(f"데이터 전처리 중 오류 발생: {e}")
    
    def extract_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        신호에서 특성 추출
        
        전처리된 신호 데이터에서 중요한 특성을 추출합니다.
        
        Args:
            data (np.ndarray): 특성을 추출할 신호 데이터
            
        Returns:
            Dict[str, float]: 추출된 특성 딕셔너리
            
        Raises:
            ValueError: 입력 데이터가 비어있거나 잘못된 형식인 경우
        """
        if data is None or data.size == 0:
            raise ValueError("특성을 추출할 데이터가 비어 있습니다.")
            
        try:
            return self.signal_processor.extract_features(data)
        except Exception as e:
            raise ValueError(f"특성 추출 중 오류 발생: {e}")
    
    def generate_stimulation(self, duration: float, **params) -> np.ndarray:
        """
        자극 파형 생성
        
        지정된 매개변수를 사용하여 자극 파형을 생성합니다.
        신경재생 단계에 따라 최적화된 파라미터를 사용합니다.
        
        Args:
            duration (float): 자극 지속 시간 (초)
            **params: 자극 매개변수
                - 'amplitude': 자극 진폭 (mA)
                - 'frequency': 자극 주파수 (Hz)
                - 'pulse_width': 펄스 폭 (μs)
                - 'waveform': 파형 유형 ('monophasic', 'biphasic', 'triphasic', 'burst')
                - 'burst_frequency': 버스트 자극 주파수 (Hz, 'burst' 파형 시 사용)
                - 'pulses_per_burst': 버스트당 펄스 수 ('burst' 파형 시 사용)
            
        Returns:
            np.ndarray: 생성된 자극 파형
            
        Raises:
            ValueError: 매개변수 값이 잘못된 경우
        """
        if duration <= 0:
            raise ValueError("자극 지속 시간은 0보다 커야 합니다.")
            
        # 재생 단계에 따른 기본 파라미터 설정
        stage_params = self.neural_regeneration_params[self.config['regeneration_stage']]
        default_params = {
            'frequency': np.mean(stage_params['frequency_range']),
            'amplitude': np.mean(stage_params['amplitude_range']),
            'pulse_width': stage_params['pulse_width'],
            'waveform': 'biphasic'  # 기본적으로 biphasic 파형 사용
        }
        
        # 사용자 파라미터로 업데이트
        final_params = {**default_params, **params}
        
        try:
            return self.stimulation_controller.generate_stimulation_waveform(duration, **final_params)
        except Exception as e:
            raise ValueError(f"자극 파형 생성 중 오류 발생: {e}")
    
    def optimize_parameters(self, objective_function: Callable, method: str = 'grid',
                           **kwargs) -> Dict[str, Any]:
        """
        자극 매개변수 최적화
        
        다양한 최적화 알고리즘을 사용하여 목적 함수를 최대화하는 자극 매개변수를 찾습니다.
        
        Args:
            objective_function (Callable): 최적화할 목적 함수
            method (str, optional): 최적화 방법. 기본값은 'grid'.
                가능한 값: 'grid' (그리드 탐색), 'pso' (입자 군집 최적화), 'bayesian' (베이지안 최적화)
            **kwargs: 최적화 방법에 따른 추가 매개변수
                - 'grid': 'parameter_ranges' (Dict[str, List[float]]) - 탐색할 매개변수 범위
                - 'pso': 'num_particles' (int), 'num_iterations' (int) - 입자 수와 반복 횟수
                - 'bayesian': 'num_initial_points' (int), 'num_iterations' (int) - 초기 포인트 수와 반복 횟수
            
        Returns:
            Dict[str, Any]: 최적화 결과
                - 'parameters': 최적 매개변수
                - 'score': 최적 목적 함수 값
            
        Raises:
            ValueError: 지원되지 않는 최적화 방법이나 매개변수 오류 시 발생
        """
        if method not in ['grid', 'pso', 'bayesian']:
            raise ValueError(f"지원되지 않는 최적화 방법: {method}. 'grid', 'pso', 'bayesian' 중 하나를 사용하세요.")
            
        # 재생 단계에 따른 파라미터 범위 설정
        stage_params = self.neural_regeneration_params[self.config['regeneration_stage']]
        if method == 'grid' and 'parameter_ranges' not in kwargs:
            kwargs['parameter_ranges'] = {
                'frequency': np.linspace(*stage_params['frequency_range'], 10),
                'amplitude': np.linspace(*stage_params['amplitude_range'], 10),
                'pulse_width': [stage_params['pulse_width']]
            }
            
        try:
            if method == 'grid':
                if 'parameter_ranges' not in kwargs:
                    raise ValueError("그리드 탐색에는 'parameter_ranges' 매개변수가 필요합니다.")
                    
                result = self.parameter_optimizer.grid_search(
                    objective_function, 
                    kwargs.get('parameter_ranges', {})
                )
            elif method == 'pso':
                result = self.parameter_optimizer.particle_swarm_optimization(
                    objective_function,
                    kwargs.get('num_particles', 10),
                    kwargs.get('num_iterations', 50)
                )
            elif method == 'bayesian':
                result = self.parameter_optimizer.bayesian_optimization(
                    objective_function,
                    kwargs.get('num_initial_points', 5),
                    kwargs.get('num_iterations', 20)
                )
                
            logger.info(f"최적화 완료 - 최적 점수: {result['score']:.4f}")
            return result
            
        except Exception as e:
            raise ValueError(f"매개변수 최적화 중 오류 발생: {e}")
    
    def train_dqn(self, num_episodes: int = 100, batch_size: int = 32,
                 max_steps: int = 100) -> List[float]:
        """
        DQN 에이전트 학습
        
        강화학습 환경에서 DQN 에이전트를 학습시킵니다.
        
        Args:
            num_episodes (int, optional): 학습할 에피소드 수. 기본값은 100.
            batch_size (int, optional): 경험 리플레이 배치 크기. 기본값은 32.
            max_steps (int, optional): 에피소드당 최대 스텝 수. 기본값은 100.
            
        Returns:
            List[float]: 에피소드별 총 보상을 포함하는 리스트
            
        Raises:
            ValueError: 강화학습이 비활성화되어 있거나 매개변수 오류 시 발생
        """
        if not self.config['use_reinforcement_learning']:
            raise ValueError("강화학습 기능이 비활성화되어 있습니다. 'use_reinforcement_learning' 설정을 True로 변경하세요.")
            
        if num_episodes <= 0 or batch_size <= 0 or max_steps <= 0:
            raise ValueError("에피소드 수, 배치 크기, 최대 스텝 수는 모두 양수여야 합니다.")
            
        total_rewards = []
        
        # 학습 모드 활성화
        self.is_learning = True
        
        try:
            for episode in range(num_episodes):
                # 환경 초기화
                state = self.environment.reset()
                total_reward = 0
                
                for step in range(max_steps):
                    # 행동 선택
                    action = self.agent.act(state)
                    
                    # 환경에서 한 스텝 진행
                    next_state, reward, done, info = self.environment.step(action)
                    
                    # 경험 저장
                    self.agent.memorize(state[0], action, reward, next_state[0], done)
                    
                    # 메모리에 충분한 경험이 쌓이면 학습 수행
                    if len(self.agent.memory) > batch_size:
                        self.agent.replay(batch_size)
                    
                    # 학습 데이터 기록
                    self.training_data['states'].append(state[0])
                    self.training_data['actions'].append(action)
                    self.training_data['rewards'].append(reward)
                    self.training_data['next_states'].append(next_state[0])
                    self.training_data['dones'].append(done)
                    
                    # 상태 업데이트
                    state = next_state
                    total_reward += reward
                    self.step_count += 1
                    
                    # 에피소드 종료 확인
                    if done:
                        break
                        
                # 타겟 모델 주기적 업데이트
                self.agent.update_target_model()
                
                # 총 보상 기록
                total_rewards.append(total_reward)
                self.episode_count += 1
                
                # 진행 상황 출력 (10 에피소드마다)
                if (episode + 1) % 10 == 0:
                    avg_reward = np.mean(total_rewards[-10:])
                    logger.info(f"에피소드 {episode + 1}/{num_episodes}, "
                               f"평균 보상: {avg_reward:.4f}, "
                               f"재생 단계: {self.config['regeneration_stage']}")
            
            # 학습된 모델 저장
            model_path = os.path.join(self.config['model_path'], 'dqn_model.h5')
            self.agent.save(model_path)
            logger.info(f"DQN 모델이 '{model_path}'에 저장되었습니다.")
            
            # 학습 모드 비활성화
            self.is_learning = False
            
            return total_rewards
            
        except Exception as e:
            self.is_learning = False  # 예외 발생 시에도 학습 모드 비활성화
            raise ValueError(f"DQN 학습 중 오류 발생: {e}")
    
    def train_lstm(self, data: np.ndarray, epochs: int = 100, batch_size: int = 32,
                  validation_split: float = 0.2) -> History:
        """
        LSTM 모델 학습
        
        시계열 신경 신호 데이터를 사용하여 LSTM 모델을 학습시킵니다.
        
        Args:
            data (np.ndarray): 학습 데이터
            epochs (int, optional): 학습 에폭 수. 기본값은 100.
            batch_size (int, optional): 배치 크기. 기본값은 32.
            validation_split (float, optional): 검증 데이터 비율. 기본값은 0.2.
            
        Returns:
            History: 학습 과정에 대한 히스토리 객체
            
        Raises:
            ValueError: LSTM이 비활성화되어 있거나 입력 데이터에 문제가 있는 경우 발생
        """
        if not self.config['use_lstm']:
            raise ValueError("LSTM 기능이 비활성화되어 있습니다. 'use_lstm' 설정을 True로 변경하세요.")
            
        if not TF_AVAILABLE:
            raise ValueError("TensorFlow가 설치되지 않았습니다. LSTM 모델을 사용하려면 TensorFlow를 설치하세요.")
            
        if data is None or data.size == 0:
            raise ValueError("학습 데이터가 비어 있습니다.")
            
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("에폭 수와 배치 크기는 양수여야 합니다.")
            
        if not 0 <= validation_split < 1:
            raise ValueError("검증 데이터 비율은 0 이상 1 미만이어야 합니다.")
            
        try:
            # 시퀀스 데이터 준비
            X, y = self.data_transformer.prepare_sequence_data(
                data, 
                self.config['sequence_length']
            )
            
            # 입력 형태 변환 (샘플 수, 시퀀스 길이, 특성 차원)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # 콜백 함수 설정
            callbacks = []
            if TF_AVAILABLE:
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True,
                        verbose=1
                    ),
                    ModelCheckpoint(
                        os.path.join(self.config['model_path'], 'lstm_model.h5'),
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=1
                    )
                ]
            
            # 모델 학습
            history = self.lstm_model.train(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks
            )
            
            logger.info(f"LSTM 모델이 '{self.config['model_path']}/lstm_model.h5'에 저장되었습니다.")
            
            return history
            
        except Exception as e:
            raise ValueError(f"LSTM 학습 중 오류 발생: {e}")
    
    def predict_lstm(self, data: np.ndarray) -> float:
        """
        LSTM으로 신호 예측
        
        학습된 LSTM 모델을 사용하여 신경 신호의 다음 값을 예측합니다.
        
        Args:
            data (np.ndarray): 입력 신호 데이터
            
        Returns:
            float: 예측된 다음 시점의 값
            
        Raises:
            ValueError: LSTM이 비활성화되어 있거나 입력 데이터가 너무 짧은 경우 발생
        """
        if not self.config['use_lstm']:
            raise ValueError("LSTM 기능이 비활성화되어 있습니다. 'use_lstm' 설정을 True로 변경하세요.")
            
        if not TF_AVAILABLE:
            raise ValueError("TensorFlow가 설치되지 않았습니다. LSTM 모델을 사용하려면 TensorFlow를 설치하세요.")
            
        if data is None or data.size == 0:
            raise ValueError("예측할 데이터가 비어 있습니다.")
            
        if len(data) < self.config['sequence_length']:
            raise ValueError(f"입력 데이터 길이({len(data)})가 요구되는 시퀀스 길이({self.config['sequence_length']})보다 짧습니다.")
            
        try:
            # 시퀀스 데이터 준비 (가장 최근 데이터 사용)
            X = data[-self.config['sequence_length']:].reshape(1, self.config['sequence_length'], 1)
            
            # 예측
            prediction = self.lstm_model.predict(X)
            
            return float(prediction[0, 0])
            
        except Exception as e:
            raise ValueError(f"LSTM 예측 중 오류 발생: {e}")
    
    def adaptive_stimulation(self, data: np.ndarray, duration: float, 
                            target_response: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        적응형 자극 수행
        
        신경 신호 데이터를 분석하고 강화학습을 통해 최적화된 자극 매개변수를 적용합니다.
        신경재생 단계에 따라 적절한 자극 프로토콜을 적용합니다.
        
        Args:
            data (np.ndarray): 신호 데이터
            duration (float): 자극 지속 시간 (초)
            target_response (Optional[np.ndarray], optional): 목표 신경 응답. 기본값은 None.
            
        Returns:
            Dict[str, Any]: 자극 결과 딕셔너리
                - 'stimulation_parameters': 사용된 자극 매개변수
                - 'stimulation_waveform': 생성된 자극 파형
                - 'predicted_response': 예측된 신경 반응 (LSTM 활성화 시)
                - 'features': 추출된 신호 특성
                - 'regeneration_stage': 현재 신경재생 단계
                - 'target_mechanisms': 목표 메커니즘
            
        Raises:
            ValueError: 데이터가 비어있거나 처리 중 오류 발생 시
        """
        if data is None or data.size == 0:
            raise ValueError("자극을 적용할 데이터가 비어 있습니다.")
            
        if duration <= 0:
            raise ValueError("자극 지속 시간은 0보다 커야 합니다.")
            
        try:
            # 데이터 전처리
            processed_data = self.preprocess_data(data)
            
            # 특성 추출
            features = self.signal_processor.extract_features(processed_data)
            
            # 현재 상태 구성
            feature_values = np.array(list(features.values()))
            
            # 상태 정규화
            state = self.data_transformer.normalize(feature_values)
            self.current_state = state
            
            # 환경에 목표 응답 설정 (제공된 경우)
            if target_response is not None and self.environment is not None:
                self.environment.set_target_response(target_response)
            
            # 재생 단계 정보 가져오기
            stage_params = self.neural_regeneration_params[self.config['regeneration_stage']]
            
            # 자극 매개변수 선택
            if self.config['use_reinforcement_learning'] and self.agent is not None:
                # 학습된 DQN 에이전트로 최적 행동 선택
                action = self.agent.act(state.reshape(1, -1))
                stim_params = self.environment.get_action_description(action)
            else:
                # 재생 단계에 따른 기본 자극 매개변수
                stim_params = {
                    'amplitude': np.mean(stage_params['amplitude_range']),
                    'frequency': np.mean(stage_params['frequency_range']),
                    'pulse_width': stage_params['pulse_width']
                }
            
            # 자극 제어기 매개변수 업데이트
            self.stimulation_controller.update_parameters(**stim_params)
            
            # 자극 파형 생성
            stimulation = self.stimulation_controller.generate_stimulation_waveform(
                duration, **stim_params)
            
            # 자극 효과 예측 (LSTM 사용 시)
            predicted_response = None
            if self.config['use_lstm'] and self.lstm_model is not None and TF_AVAILABLE:
                try:
                    predicted_response = self.predict_lstm(processed_data)
                except Exception as e:
                    logger.warning(f"LSTM 예측 중 오류 발생: {e}")
            
            # 결과 반환
            result = {
                'stimulation_parameters': stim_params,
                'stimulation_waveform': stimulation,
                'predicted_response': predicted_response,
                'features': features,
                'regeneration_stage': self.config['regeneration_stage'],
                'target_mechanisms': stage_params['target_factors'],
                'primary_mechanism': stage_params['primary_mechanism']
            }
            
            logger.info(f"적응형 자극 적용 완료 - 단계: {self.config['regeneration_stage']}, "
                       f"주파수: {stim_params['frequency']}Hz, "
                       f"진폭: {stim_params['amplitude']}mA")
            
            return result
            
        except Exception as e:
            raise ValueError(f"적응형 자극 적용 중 오류 발생: {e}")
    
    def set_regeneration_stage(self, stage: str) -> None:
        """
        신경재생 단계 설정
        
        Args:
            stage (str): 재생 단계 ('acute', 'subacute', 'regeneration', 'remodeling')
            
        Raises:
            ValueError: 유효하지 않은 재생 단계인 경우
        """
        valid_stages = ['acute', 'subacute', 'regeneration', 'remodeling']
        if stage not in valid_stages:
            raise ValueError(f"유효하지 않은 재생 단계: {stage}. "
                           f"다음 중 하나를 선택하세요: {valid_stages}")
        
        self.config['regeneration_stage'] = stage
        
        # 강화학습 환경 재초기화 (활성화된 경우)
        if self.config['use_reinforcement_learning']:
            self._initialize_reinforcement_learning()
        
        logger.info(f"신경재생 단계가 '{stage}'로 변경되었습니다.")
    
    def visualize_results(self, data: dict, save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        결과 시각화
        
        시스템에서 생성된 다양한 데이터를 시각화합니다.
        
        Args:
            data (dict): 시각화할 데이터 딕셔너리
                가능한 키:
                - 'signal': 신호 데이터
                - 'stimulation_waveform': 자극 파형
                - 'rewards': 학습 보상 히스토리
                - 'optimization_history': 매개변수 최적화 히스토리
            save_path (Optional[str], optional): 그림 저장 경로. 기본값은 None (저장 안 함).
            
        Returns:
            Dict[str, plt.Figure]: 생성된 그림 객체들을 포함하는 딕셔너리
            
        Raises:
            ValueError: 시각화 중 오류 발생 시
        """
        figures = {}
        
        # 저장 경로 설정
        if save_path is None:
            save_path = self.config['save_path']
            
        try:
            os.makedirs(save_path, exist_ok=True)
        except OSError as e:
            raise ValueError(f"저장 경로 생성 중 오류 발생: {e}")
            
        try:
            # 신호 플롯
            if 'signal' in data:
                fig = self.visualizer.plot_signal(
                    data['signal'], 
                    sampling_rate=self.config['sampling_rate'],
                    title="신경 신호",
                    save_path=save_path and os.path.join(save_path, "signal.png")
                )
                figures['signal'] = fig
            
            # 스펙트로그램 플롯
            if 'signal' in data:
                fig = self.visualizer.plot_spectrogram(
                    data['signal'], 
                    sampling_rate=self.config['sampling_rate'],
                    title="신호 스펙트로그램",
                    save_path=save_path and os.path.join(save_path, "spectrogram.png")
                )
                figures['spectrogram'] = fig
            
            # 자극 파형 플롯
            if 'stimulation_waveform' in data:
                fig = self.visualizer.plot_stimulation_waveform(
                    data['stimulation_waveform'], 
                    sampling_rate=self.config['sampling_rate'],
                    title=f"자극 파형 - {self.config['regeneration_stage']} 단계",
                    save_path=save_path and os.path.join(save_path, "stimulation.png")
                )
                figures['stimulation'] = fig
            
            # 보상 히스토리 플롯
            if 'rewards' in data:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(data['rewards'])
                ax.set_xlabel('에피소드')
                ax.set_ylabel('총 보상')
                ax.set_title('학습 보상 히스토리')
                ax.grid(True)
                
                if save_path:
                    plt.savefig(os.path.join(save_path, "rewards.png"), dpi=300, bbox_inches='tight')
                    
                figures['rewards'] = fig
                
            # 매개변수 최적화 진행 상황 플롯
            if 'optimization_history' in data:
                fig = self.visualizer.plot_optimization_progress(
                    data['optimization_history'],
                    parameter='score',
                    title="매개변수 최적화 진행 상황",
                    save_path=save_path and os.path.join(save_path, "optimization.png")
                )
                figures['optimization'] = fig
                
            return figures
            
        except Exception as e:
            raise ValueError(f"결과 시각화 중 오류 발생: {e}")
    
    def get_regeneration_info(self) -> Dict[str, Any]:
        """
        현재 신경재생 단계 정보 반환
        
        Returns:
            Dict[str, Any]: 현재 재생 단계에 대한 상세 정보
        """
        stage = self.config['regeneration_stage']
        return {
            'current_stage': stage,
            'parameters': self.neural_regeneration_params[stage],
            'description': {
                'acute': '급성기 (0-3일): 염증 억제 및 신경보호',
                'subacute': '아급성기 (4-14일): 신경영양인자 유도 및 슈반세포 활성화',
                'regeneration': '재생기 (14-60일): 축삭 성장 가속화 및 수초화',
                'remodeling': '재조직화기 (2-6개월): 시냅스 가소성 및 기능적 통합'
            }[stage]
        }


# 메인 실행 코드
def main():
    """
    적응형 신경 자극 시스템의 메인 실행 함수
    
    이 함수는 시스템의 기본 동작을 시연합니다.
    실제 응용에서는 특정 요구사항에 맞게 수정하여 사용하세요.
    """
    # 시스템 설정
    config = {
        'sampling_rate': 1000.0,
        'sequence_length': 100,
        'feature_dim': 5,
        'use_lstm': True,
        'use_reinforcement_learning': True,
        'save_path': 'results',
        'model_path': 'models/saved',
        'regeneration_stage': 'acute'  # 초기 단계 설정
    }
    
    # 시스템 인스턴스 생성
    system = AdaptiveStimulationSystem(config)
    
    # 가상 신호 생성 (실제 데이터 대신)
    print("가상 신호 생성 중...")
    time = np.arange(0, 60, 1/config['sampling_rate'])
    signal = np.sin(2 * np.pi * 5 * time) + 0.5 * np.sin(2 * np.pi * 10 * time) + 0.2 * np.random.randn(len(time))
    
    # 데이터 전처리
    print("신호 전처리 중...")
    processed_signal = system.preprocess_data(signal)
    
    # 현재 재생 단계 정보 출력
    regen_info = system.get_regeneration_info()
    print(f"\n현재 신경재생 단계: {regen_info['description']}")
    
    # DQN 에이전트 학습
    print("\nDQN 에이전트 학습 중...")
    rewards = system.train_dqn(num_episodes=50, batch_size=32, max_steps=100)
    
    # 적응형 자극 수행
    print("\n적응형 자극 적용 중...")
    result = system.adaptive_stimulation(processed_signal, duration=5.0)
    
    # 결과 시각화
    print("\n결과 시각화 중...")
    data = {
        'signal': processed_signal,
        'stimulation_waveform': result['stimulation_waveform'],
        'rewards': rewards
    }
    
    figures = system.visualize_results(data, save_path=config['save_path'])
    
    # 결과 출력
    print("\n자극 매개변수:", result['stimulation_parameters'])
    print("특성:", result['features'])
    print("목표 메커니즘:", result['target_mechanisms'])
    
    print(f"\n결과가 '{config['save_path']}' 폴더에 저장되었습니다.")
    
    # 그래프 표시
    plt.show()

if __name__ == "__main__":
    main()
