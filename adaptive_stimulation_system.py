"""
적응형 신경 자극 시스템

이 모듈은 적응형 신경 자극 시스템의 메인 진입점입니다.
강화학습 기반 자극 최적화, 신경 신호 처리, 시각화 등의 기능을 통합합니다.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Union

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
    """적응형 신경 자극 시스템의 핵심 클래스"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        AdaptiveStimulationSystem 초기화
        
        매개변수:
            config (Dict[str, Any]): 시스템 설정
        """
        # 기본 설정 값
        self.config = {
            'sampling_rate': 1000.0,  # Hz
            'sequence_length': 100,
            'feature_dim': 5,
            'use_lstm': True,
            'use_reinforcement_learning': True,
            'save_path': 'results',
            'model_path': 'models/saved'
        }
        
        # 사용자 설정으로 업데이트
        if config:
            self.config.update(config)
            
        # 유틸리티 인스턴스 생성
        self.signal_processor = SignalProcessor(sampling_rate=self.config['sampling_rate'])
        self.data_loader = DataLoader()
        self.data_transformer = DataTransformer()
        self.stimulation_controller = StimulationController(sampling_rate=self.config['sampling_rate'])
        self.parameter_optimizer = ParameterOptimizer()
        self.visualizer = Visualizer()
        
        # 결과 저장 디렉토리 생성
        os.makedirs(self.config['save_path'], exist_ok=True)
        os.makedirs(self.config['model_path'], exist_ok=True)
        
        # 강화학습 환경 및 에이전트
        if self.config['use_reinforcement_learning']:
            self.environment = StimulationEnvironment(
                state_size=self.config['feature_dim'],
                amplitude_levels=5,
                frequency_levels=5,
                pulse_width_levels=5
            )
            self.agent = DQNAgent(
                state_size=self.config['feature_dim'],
                action_size=self.environment.action_size
            )
            
        # LSTM 모델
        if self.config['use_lstm']:
            self.lstm_model = LSTMModel(
                sequence_length=self.config['sequence_length'],
                feature_dim=self.config['feature_dim']
            )
            
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
        
    def load_data(self, file_path: str, **kwargs) -> Tuple[np.ndarray, float]:
        """
        신경 신호 데이터 로드
        
        매개변수:
            file_path (str): 데이터 파일 경로
            **kwargs: 추가 매개변수
            
        반환값:
            Tuple[np.ndarray, float]: (데이터, 샘플링 레이트)
        """
        data = None
        sampling_rate = self.config['sampling_rate']
        
        # 파일 확장자에 따라 적절한 로더 선택
        ext = os.path.splitext(file_path)[1].lower()
        
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
            
        # 설정 업데이트
        self.config['sampling_rate'] = sampling_rate
        self.signal_processor.sampling_rate = sampling_rate
        self.stimulation_controller.sampling_rate = sampling_rate
        
        return data, sampling_rate
    
    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        데이터 전처리
        
        매개변수:
            data (np.ndarray): 원본 데이터
            
        반환값:
            np.ndarray: 전처리된 데이터
        """
        # 노이즈 필터링
        filtered_data = self.signal_processor.bandpass_filter(data)
        
        # 노치 필터 적용 (60Hz)
        filtered_data = self.signal_processor.notch_filter(filtered_data)
        
        # 정규화
        normalized_data = self.data_transformer.normalize(filtered_data)
        
        return normalized_data
    
    def extract_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        신호에서 특성 추출
        
        매개변수:
            data (np.ndarray): 데이터
            
        반환값:
            Dict[str, float]: 추출된 특성
        """
        return self.signal_processor.extract_features(data)
    
    def generate_stimulation(self, duration: float, **params) -> np.ndarray:
        """
        자극 파형 생성
        
        매개변수:
            duration (float): 자극 지속 시간 (초)
            **params: 자극 매개변수
            
        반환값:
            np.ndarray: 생성된 자극 파형
        """
        return self.stimulation_controller.generate_stimulation_waveform(duration, **params)
    
    def optimize_parameters(self, objective_function, method: str = 'grid',
                           **kwargs) -> Dict[str, Any]:
        """
        자극 매개변수 최적화
        
        매개변수:
            objective_function: 최적화할 목적 함수
            method (str): 최적화 방법 ('grid', 'pso', 'bayesian')
            **kwargs: 추가 매개변수
            
        반환값:
            Dict[str, Any]: 최적화 결과
        """
        if method == 'grid':
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
        else:
            raise ValueError(f"지원되지 않는 최적화 방법: {method}")
            
        return result
    
    def train_dqn(self, num_episodes: int = 100, batch_size: int = 32,
                 max_steps: int = 100) -> List[float]:
        """
        DQN 에이전트 학습
        
        매개변수:
            num_episodes (int): 학습 에피소드 수
            batch_size (int): 배치 크기
            max_steps (int): 에피소드당 최대 스텝 수
            
        반환값:
            List[float]: 에피소드별 총 보상
        """
        if not self.config['use_reinforcement_learning']:
            raise ValueError("강화학습 기능이 비활성화되어 있습니다")
            
        total_rewards = []
        
        for episode in range(num_episodes):
            state = self.environment.reset()
            total_reward = 0
            
            for step in range(max_steps):
                # 행동 선택
                action = self.agent.act(state)
                
                # 환경에서 한 스텝 진행
                next_state, reward, done, info = self.environment.step(action)
                
                # 메모리에 저장
                self.agent.memorize(state[0], action, reward, next_state[0], done)
                
                # 배치 학습
                if len(self.agent.memory) > batch_size:
                    self.agent.replay(batch_size)
                
                # 다음 상태로 이동
                state = next_state
                total_reward += reward
                
                # 에피소드 종료 확인
                if done:
                    break
                    
            # 에피소드마다 타겟 모델 업데이트
            self.agent.update_target_model()
            
            # 총 보상 기록
            total_rewards.append(total_reward)
            
            # 진행 상황 출력
            if (episode + 1) % 10 == 0:
                print(f"에피소드 {episode + 1}/{num_episodes}, 총 보상: {total_reward:.4f}")
        
        # 학습된 모델 저장
        self.agent.save(os.path.join(self.config['model_path'], 'dqn_model.h5'))
        
        return total_rewards
    
    def train_lstm(self, data: np.ndarray, epochs: int = 100, batch_size: int = 32,
                  validation_split: float = 0.2) -> tf.keras.callbacks.History:
        """
        LSTM 모델 학습
        
        매개변수:
            data (np.ndarray): 학습 데이터
            epochs (int): 학습 에폭 수
            batch_size (int): 배치 크기
            validation_split (float): 검증 데이터 비율
            
        반환값:
            tf.keras.callbacks.History: 학습 히스토리
        """
        if not self.config['use_lstm']:
            raise ValueError("LSTM 기능이 비활성화되어 있습니다")
            
        # 시퀀스 데이터 준비
        X, y = self.data_transformer.prepare_sequence_data(
            data, 
            self.config['sequence_length']
        )
        
        # 입력 형태 변환
        X = X.reshape(X.shape[0], X.shape[1], 1)  # (샘플 수, 시퀀스 길이, 특성 차원)
        
        # 콜백 함수 설정
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                os.path.join(self.config['model_path'], 'lstm_model.h5'),
                monitor='val_loss',
                save_best_only=True
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
        
        return history
    
    def predict_lstm(self, data: np.ndarray) -> np.ndarray:
        """
        LSTM으로 신호 예측
        
        매개변수:
            data (np.ndarray): 입력 데이터
            
        반환값:
            np.ndarray: 예측 결과
        """
        if not self.config['use_lstm']:
            raise ValueError("LSTM 기능이 비활성화되어 있습니다")
            
        # 시퀀스 데이터 준비
        X = data[-self.config['sequence_length']:].reshape(1, self.config['sequence_length'], 1)
        
        # 예측
        prediction = self.lstm_model.predict(X)
        
        return prediction[0, 0]
    
    def adaptive_stimulation(self, data: np.ndarray, duration: float, 
                            target_response: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        적응형 자극 수행
        
        매개변수:
            data (np.ndarray): 신호 데이터
            duration (float): 자극 지속 시간 (초)
            target_response (Optional[np.ndarray]): 목표 신경 응답
            
        반환값:
            Dict[str, Any]: 자극 결과
        """
        # 데이터 전처리
        processed_data = self.preprocess_data(data)
        
        # 특성 추출
        features = self.signal_processor.extract_features(processed_data)
        
        # 현재 상태 구성
        feature_values = np.array(list(features.values()))
        
        # 상태 정규화
        state = self.data_transformer.normalize(feature_values)
        
        # 환경에 목표 응답 설정
        if target_response is not None:
            self.environment.set_target_response(target_response)
        
        # 학습된 DQN 에이전트로 최적 행동 선택
        if self.config['use_reinforcement_learning']:
            action = self.agent.act(state.reshape(1, -1))
            stim_params = self.environment.get_action_description(action)
        else:
            # 기본 자극 매개변수
            stim_params = {
                'amplitude': 1.0,  # mA
                'frequency': 130.0,  # Hz
                'pulse_width': 60.0  # μs
            }
        
        # 자극 제어기 매개변수 업데이트
        self.stimulation_controller.update_parameters(**stim_params)
        
        # 자극 파형 생성
        stimulation = self.stimulation_controller.generate_stimulation_waveform(
            duration, **stim_params)
        
        # 자극 효과 예측 (LSTM 사용 시)
        predicted_response = None
        if self.config['use_lstm']:
            predicted_response = self.predict_lstm(processed_data)
        
        # 결과 반환
        return {
            'stimulation_parameters': stim_params,
            'stimulation_waveform': stimulation,
            'predicted_response': predicted_response,
            'features': features
        }
    
    def visualize_results(self, data: dict, save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        결과 시각화
        
        매개변수:
            data (dict): 시각화할 데이터
            save_path (Optional[str]): 저장 경로
            
        반환값:
            Dict[str, plt.Figure]: 생성된 그림
        """
        figures = {}
        
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
                title="자극 파형",
                save_path=save_path and os.path.join(save_path, "stimulation.png")
            )
            figures['stimulation'] = fig
        
        # 보상 히스토리 플롯
        if 'rewards' in data:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data['rewards'])
            ax.set_xlabel('스텝')
            ax.set_ylabel('보상')
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


# 메인 실행 코드
def main():
    # 시스템 설정
    config = {
        'sampling_rate': 1000.0,
        'sequence_length': 100,
        'feature_dim': 5,
        'use_lstm': True,
        'use_reinforcement_learning': True,
        'save_path': 'results',
        'model_path': 'models/saved'
    }
    
    # 시스템 인스턴스 생성
    system = AdaptiveStimulationSystem(config)
    
    # 가상 신호 생성 (실제 데이터 대신)
    time = np.arange(0, 60, 1/config['sampling_rate'])
    signal = np.sin(2 * np.pi * 5 * time) + 0.5 * np.sin(2 * np.pi * 10 * time) + 0.2 * np.random.randn(len(time))
    
    # 데이터 전처리
    processed_signal = system.preprocess_data(signal)
    
    # DQN 에이전트 학습
    rewards = system.train_dqn(num_episodes=50, batch_size=32, max_steps=100)
    
    # 적응형 자극 수행
    result = system.adaptive_stimulation(processed_signal, duration=5.0)
    
    # 결과 시각화
    data = {
        'signal': processed_signal,
        'stimulation_waveform': result['stimulation_waveform'],
        'rewards': rewards
    }
    
    figures = system.visualize_results(data, save_path=config['save_path'])
    
    # 결과 출력
    print("자극 매개변수:", result['stimulation_parameters'])
    print("특성:", result['features'])
    
    plt.show()

if __name__ == "__main__":
    main()
