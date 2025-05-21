"""
AdaptiveStimulationSystem 클래스에 대한 통합 테스트

이 모듈은 전체 시스템 통합 기능을 테스트합니다.
"""

import unittest
import numpy as np
import os
import tempfile
import shutil

# 필요한 경로 추가
import sys
sys.path.append('.')

# tensorflow 임포트 실패 시 모의 객체 생성을 위한 준비
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # 모의 tf 모듈 생성
    import types
    tf = types.ModuleType('tf')
    tf.keras = types.ModuleType('tf.keras')
    tf.keras.Model = object
    tf.keras.callbacks = types.ModuleType('tf.keras.callbacks')
    tf.keras.callbacks.History = object
    sys.modules['tensorflow'] = tf
    sys.modules['tensorflow.keras'] = tf.keras

from adaptive_stimulation_system import AdaptiveStimulationSystem

# TensorFlow가 설치되지 않은 경우 스킵할 테스트 데코레이터
def requires_tensorflow(func):
    """TensorFlow가 필요한 테스트를 위한 데코레이터"""
    def wrapper(*args, **kwargs):
        if not TENSORFLOW_AVAILABLE:
            raise unittest.SkipTest("TensorFlow가 설치되어 있지 않습니다")
        return func(*args, **kwargs)
    return wrapper

class TestAdaptiveStimulationSystem(unittest.TestCase):
    """
    AdaptiveStimulationSystem 클래스의 통합 테스트
    
    이 테스트 클래스는 전체 시스템의 주요 기능을 검증합니다:
    - 시스템 초기화 및 설정
    - 데이터 로드 및 전처리
    - 특성 추출
    - 자극 파형 생성
    - DQN 학습
    - 적응형 자극 적용
    - 결과 시각화
    """
    
    @requires_tensorflow
    def setUp(self):
        """
        각 테스트 전에 실행되는 설정 메서드
        """
        # 임시 디렉토리 생성
        self.test_dir = tempfile.mkdtemp()
        self.save_path = os.path.join(self.test_dir, 'results')
        self.model_path = os.path.join(self.test_dir, 'models')
        
        # 시스템 설정
        self.config = {
            'sampling_rate': 1000.0,
            'sequence_length': 10,  # 작은 값으로 설정 (테스트용)
            'feature_dim': 5,
            'use_lstm': True,
            'use_reinforcement_learning': True,
            'save_path': self.save_path,
            'model_path': self.model_path
        }
        
        # 시스템 인스턴스 생성
        self.system = AdaptiveStimulationSystem(self.config)
        
        # 테스트용 신호 생성
        self.time = np.arange(0, 1, 1/self.config['sampling_rate'])
        self.signal = np.sin(2 * np.pi * 5 * self.time) + 0.5 * np.sin(2 * np.pi * 10 * self.time) + 0.2 * np.random.randn(len(self.time))
        
        # 테스트용 CSV 파일 생성
        self.csv_path = os.path.join(self.test_dir, 'test_signal.csv')
        import pandas as pd
        df = pd.DataFrame({
            'time': self.time,
            'signal': self.signal
        })
        df.to_csv(self.csv_path, index=False)
    
    def tearDown(self):
        """
        각 테스트 후에 실행되는 정리 메서드
        """
        # 임시 디렉토리 및 파일 삭제
        shutil.rmtree(self.test_dir)
    
    @requires_tensorflow
    def test_initialization(self):
        """초기화 테스트"""
        # 기본 설정 확인
        self.assertEqual(self.system.config['sampling_rate'], self.config['sampling_rate'])
        self.assertEqual(self.system.config['sequence_length'], self.config['sequence_length'])
        self.assertEqual(self.system.config['feature_dim'], self.config['feature_dim'])
        self.assertEqual(self.system.config['use_lstm'], self.config['use_lstm'])
        self.assertEqual(self.system.config['use_reinforcement_learning'], self.config['use_reinforcement_learning'])
        self.assertEqual(self.system.config['save_path'], self.config['save_path'])
        self.assertEqual(self.system.config['model_path'], self.config['model_path'])
        
        # 유틸리티 인스턴스 확인
        self.assertIsNotNone(self.system.signal_processor)
        self.assertIsNotNone(self.system.data_loader)
        self.assertIsNotNone(self.system.data_transformer)
        self.assertIsNotNone(self.system.stimulation_controller)
        self.assertIsNotNone(self.system.parameter_optimizer)
        self.assertIsNotNone(self.system.visualizer)
        
        # 모델 인스턴스 확인
        self.assertIsNotNone(self.system.environment)
        self.assertIsNotNone(self.system.agent)
        self.assertIsNotNone(self.system.lstm_model)
        
        # 잘못된 설정으로 초기화 시도
        with self.assertRaises(ValueError):
            AdaptiveStimulationSystem({'sampling_rate': 0})
        with self.assertRaises(ValueError):
            AdaptiveStimulationSystem({'sequence_length': 0})
        with self.assertRaises(ValueError):
            AdaptiveStimulationSystem({'feature_dim': 0})
    
    @requires_tensorflow
    def test_load_data(self):
        """데이터 로드 테스트"""
        # CSV 파일 로드
        data, sr = self.system.load_data(self.csv_path)
        
        # 로드된 데이터 확인
        self.assertEqual(len(data), len(self.signal))
        self.assertEqual(sr, self.config['sampling_rate'])
        
        # 존재하지 않는 파일 로드 시도
        with self.assertRaises(FileNotFoundError):
            self.system.load_data('non_existent_file.csv')
        
        # 지원되지 않는 파일 형식 로드 시도
        invalid_path = os.path.join(self.test_dir, 'invalid.xyz')
        with open(invalid_path, 'w') as f:
            f.write('test')
        with self.assertRaises(ValueError):
            self.system.load_data(invalid_path)
    
    @requires_tensorflow
    def test_preprocess_data(self):
        """데이터 전처리 테스트"""
        # 데이터 전처리
        processed_data = self.system.preprocess_data(self.signal)
        
        # 전처리 결과 확인
        self.assertEqual(len(processed_data), len(self.signal))
        
        # 정규화 확인 (값 범위: 0~1 또는 -1~1)
        self.assertLessEqual(np.max(processed_data), 1.0)
        self.assertGreaterEqual(np.min(processed_data), -1.0)
        
        # 빈 배열 전처리 시도
        with self.assertRaises(ValueError):
            self.system.preprocess_data(np.array([]))
    
    @requires_tensorflow
    def test_extract_features(self):
        """특성 추출 테스트"""
        # 특성 추출
        features = self.system.extract_features(self.signal)
        
        # 필요한 특성이 모두 추출되었는지 확인
        expected_features = [
            'mean', 'std', 'max', 'min', 'range', 'rms', 'energy', 
            'dominant_frequency', 'spectral_entropy'
        ]
        for feature in expected_features:
            self.assertIn(feature, features)
        
        # 특성 값 타입 확인
        for value in features.values():
            self.assertIsInstance(value, float)
        
        # 빈 배열에서 특성 추출 시도
        with self.assertRaises(ValueError):
            self.system.extract_features(np.array([]))
    
    @requires_tensorflow
    def test_generate_stimulation(self):
        """자극 파형 생성 테스트"""
        # 자극 지속 시간
        duration = 0.5  # 초
        
        # 기본 매개변수로 자극 파형 생성
        stim = self.system.generate_stimulation(duration)
        
        # 파형 길이 확인
        expected_length = int(duration * self.config['sampling_rate'])
        self.assertEqual(len(stim), expected_length)
        
        # 다양한 파형 유형 테스트
        for waveform in ['monophasic', 'biphasic', 'triphasic', 'burst']:
            stim = self.system.generate_stimulation(
                duration=0.1,
                amplitude=1.5,
                frequency=50.0,
                pulse_width=100.0,
                waveform=waveform
            )
            self.assertIsNotNone(stim)
            
        # 유효하지 않은 지속 시간으로 생성 시도
        with self.assertRaises(ValueError):
            self.system.generate_stimulation(0)
        with self.assertRaises(ValueError):
            self.system.generate_stimulation(-1)
    
    @requires_tensorflow
    def test_adaptive_stimulation(self):
        """적응형 자극 테스트"""
        # 전처리된 신호 준비
        processed_data = self.system.preprocess_data(self.signal)
        
        # 적응형 자극 적용
        result = self.system.adaptive_stimulation(processed_data, duration=0.1)
        
        # 결과 확인
        self.assertIn('stimulation_parameters', result)
        self.assertIn('stimulation_waveform', result)
        self.assertIn('features', result)
        
        # LSTM이 활성화된 경우 predicted_response 확인
        if self.system.config['use_lstm']:
            # LSTM 모델이 학습되지 않았으므로 예측은 실패할 수 있음
            # 이 시점에서는 예측 결과가 None이 될 수 있음
            self.assertIn('predicted_response', result)
        
        # 자극 매개변수 확인
        params = result['stimulation_parameters']
        self.assertIn('amplitude', params)
        self.assertIn('frequency', params)
        self.assertIn('pulse_width', params)
        
        # 빈 배열로 적응형 자극 시도
        with self.assertRaises(ValueError):
            self.system.adaptive_stimulation(np.array([]), duration=0.1)
    
    @requires_tensorflow
    def test_visualize_results(self):
        """결과 시각화 테스트"""
        # 가상의 데이터 준비
        data = {
            'signal': self.signal,
            'stimulation_waveform': self.system.generate_stimulation(0.5),
            'rewards': [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
        # 결과 시각화
        figures = self.system.visualize_results(data)
        
        # 그림 확인
        self.assertIn('signal', figures)
        self.assertIn('spectrogram', figures)
        self.assertIn('stimulation', figures)
        self.assertIn('rewards', figures)
        
        # 지정된 경로에 이미지가 저장되는지 확인
        save_path = os.path.join(self.test_dir, 'images')
        figures = self.system.visualize_results(data, save_path=save_path)
        
        # 저장된 파일 확인
        self.assertTrue(os.path.exists(os.path.join(save_path, 'signal.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'spectrogram.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'stimulation.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'rewards.png')))

if __name__ == '__main__':
    unittest.main()
