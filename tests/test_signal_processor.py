"""
SignalProcessor 클래스에 대한 단위 테스트

이 모듈은 utils.signal_processor.SignalProcessor 클래스의 기능을 테스트합니다.
"""

import unittest
import numpy as np
from utils.signal_processor import SignalProcessor

class TestSignalProcessor(unittest.TestCase):
    """
    SignalProcessor 클래스의 단위 테스트
    
    이 테스트 클래스는 SignalProcessor 클래스의 주요 메서드를 검증합니다:
    - 밴드패스 필터링
    - 노치 필터링
    - 특성 추출
    - 스파이크 검출
    - 발화율 계산
    """
    
    def setUp(self):
        """
        각 테스트 전에 실행되는 설정 메서드
        """
        self.sampling_rate = 1000.0  # Hz
        self.processor = SignalProcessor(sampling_rate=self.sampling_rate)
        
        # 테스트용 신호 생성
        self.time = np.arange(0, 1, 1/self.sampling_rate)
        
        # 5Hz와 50Hz 사인파의 합
        self.signal = np.sin(2 * np.pi * 5 * self.time) + 0.5 * np.sin(2 * np.pi * 50 * self.time)
        
        # 60Hz 노이즈 추가
        self.noisy_signal = self.signal + 0.3 * np.sin(2 * np.pi * 60 * self.time)
        
        # 스파이크가 있는 신호 생성
        self.spike_signal = np.zeros(1000)
        spike_positions = [100, 300, 500, 700, 900]
        for pos in spike_positions:
            self.spike_signal[pos] = 5.0  # 스파이크 추가
    
    def test_initialization(self):
        """샘플링 레이트 초기화 테스트"""
        self.assertEqual(self.processor.sampling_rate, self.sampling_rate)
        
        # 잘못된 샘플링 레이트로 초기화 시도
        with self.assertRaises(ValueError):
            SignalProcessor(sampling_rate=0)
        with self.assertRaises(ValueError):
            SignalProcessor(sampling_rate=-100)
    
    def test_bandpass_filter(self):
        """밴드패스 필터링 테스트"""
        filtered_signal = self.processor.bandpass_filter(
            self.signal, lowcut=1.0, highcut=10.0, order=4
        )
        
        # 필터링 후 길이 확인
        self.assertEqual(len(filtered_signal), len(self.signal))
        
        # 필터링 효과 확인 (주파수 도메인)
        from scipy import signal as sp_signal
        freqs, psd = sp_signal.welch(filtered_signal, fs=self.sampling_rate)
        
        # 5Hz 성분은 남아있고, 50Hz 성분은 감쇠되었는지 확인
        f5_idx = np.argmin(np.abs(freqs - 5))
        f50_idx = np.argmin(np.abs(freqs - 50))
        
        # 5Hz 파워는 유지되고, 50Hz 파워는 감소해야 함
        self.assertGreater(psd[f5_idx], psd[f50_idx] * 10)  # 5Hz 파워가 50Hz 파워보다 10배 이상 큰지 확인
        
        # 잘못된 입력에 대한 예외 처리 테스트
        with self.assertRaises(ValueError):
            self.processor.bandpass_filter(self.signal, lowcut=-1.0, highcut=10.0)
        with self.assertRaises(ValueError):
            self.processor.bandpass_filter(self.signal, lowcut=10.0, highcut=5.0)
        with self.assertRaises(ValueError):
            self.processor.bandpass_filter(self.signal, lowcut=1.0, 
                                          highcut=self.sampling_rate)
    
    def test_notch_filter(self):
        """노치 필터링 테스트"""
        filtered_signal = self.processor.notch_filter(
            self.noisy_signal, freq=60.0, q=30.0
        )
        
        # 필터링 후 길이 확인
        self.assertEqual(len(filtered_signal), len(self.noisy_signal))
        
        # 필터링 효과 확인 (주파수 도메인)
        from scipy import signal as sp_signal
        freqs, psd_original = sp_signal.welch(self.noisy_signal, fs=self.sampling_rate)
        freqs, psd_filtered = sp_signal.welch(filtered_signal, fs=self.sampling_rate)
        
        # 60Hz 성분이 감쇠되었는지 확인
        f60_idx = np.argmin(np.abs(freqs - 60))
        
        # 필터링 후 60Hz 성분의 파워가 원본보다 작아야 함
        self.assertLess(psd_filtered[f60_idx], psd_original[f60_idx])
        
        # 잘못된 입력에 대한 예외 처리 테스트
        with self.assertRaises(ValueError):
            self.processor.notch_filter(self.noisy_signal, freq=-60.0)
        with self.assertRaises(ValueError):
            self.processor.notch_filter(self.noisy_signal, freq=self.sampling_rate)
    
    def test_extract_features(self):
        """특성 추출 테스트"""
        features = self.processor.extract_features(self.signal)
        
        # 필요한 특성이 모두 추출되었는지 확인
        expected_features = [
            'mean', 'std', 'max', 'min', 'range', 'rms', 'energy', 
            'dominant_frequency', 'spectral_entropy'
        ]
        for feature in expected_features:
            self.assertIn(feature, features)
            
        # 특성 값의 타입과 범위 확인
        self.assertIsInstance(features['mean'], float)
        self.assertIsInstance(features['std'], float)
        self.assertGreaterEqual(features['std'], 0)  # 표준편차는 음수가 될 수 없음
        self.assertGreaterEqual(features['energy'], 0)  # 에너지는 음수가 될 수 없음
        
        # 5Hz 신호의 주파수 특성 확인
        self.assertAlmostEqual(features['dominant_frequency'], 5.0, delta=1.0)
        
        # 빈 배열 입력에 대한 예외 처리 테스트
        with self.assertRaises(ValueError):
            self.processor.extract_features(np.array([]))
    
    def test_detect_spikes(self):
        """스파이크 검출 테스트"""
        spike_indices = self.processor.detect_spikes(self.spike_signal, threshold=3.0)
        
        # 예상되는 스파이크 위치와 비교
        expected_spikes = [100, 300, 500, 700, 900]
        self.assertEqual(len(spike_indices), len(expected_spikes))
        for expected, actual in zip(expected_spikes, spike_indices):
            self.assertEqual(expected, actual)
        
        # 임계값을 높여서 스파이크 검출이 줄어드는지 확인
        higher_threshold_spikes = self.processor.detect_spikes(self.spike_signal, threshold=10.0)
        self.assertEqual(len(higher_threshold_spikes), 0)  # 검출된 스파이크가 없어야 함
        
        # 빈 배열 입력에 대한 예외 처리 테스트
        with self.assertRaises(ValueError):
            self.processor.detect_spikes(np.array([]), threshold=3.0)
    
    def test_calculate_firing_rate(self):
        """발화율 계산 테스트"""
        # 스파이크 인덱스 생성 (100ms 간격)
        spike_indices = [100, 200, 300, 400, 500, 600, 700, 800, 900]
        
        # 1초 윈도우 (1000 샘플)로 발화율 계산
        firing_rate = self.processor.calculate_firing_rate(spike_indices, window_size=1000)
        
        # 1초에 9개의 스파이크가 있으므로 발화율은 9Hz여야 함
        self.assertEqual(len(firing_rate), 1)  # 1초 윈도우 1개
        self.assertAlmostEqual(firing_rate[0], 9.0, delta=0.1)  # 발화율 ~9Hz
        
        # 0.5초 윈도우 (500 샘플)로 발화율 계산
        firing_rate = self.processor.calculate_firing_rate(spike_indices, window_size=500)
        
        # 0.5초 윈도우가 2개이며, 각각 4개, 5개의 스파이크를 포함
        # 따라서 발화율은 각각 8Hz, 10Hz여야 함
        self.assertEqual(len(firing_rate), 2)  # 0.5초 윈도우 2개
        self.assertAlmostEqual(firing_rate[0], 8.0, delta=0.1)  # 첫 번째 윈도우 ~8Hz
        self.assertAlmostEqual(firing_rate[1], 10.0, delta=0.1)  # 두 번째 윈도우 ~10Hz
        
        # 빈 스파이크 리스트에 대한 처리 확인
        empty_firing_rate = self.processor.calculate_firing_rate([], window_size=1000)
        self.assertEqual(len(empty_firing_rate), 0)

if __name__ == '__main__':
    unittest.main()
