"""
신경 신호 데이터 처리 유틸리티

이 모듈은 신경 신호 데이터를 전처리하고 분석하는 기능을 제공합니다.
전기생리학적 신호의 필터링, 특성 추출, 노이즈 제거 등의 알고리즘을 구현합니다.
"""

import numpy as np
from scipy import signal
from typing import List, Tuple, Dict, Any, Optional, Union

class SignalProcessor:
    """신경 신호 처리를 위한 클래스"""
    
    def __init__(self, sampling_rate: float = 1000.0):
        """
        SignalProcessor 초기화
        
        매개변수:
            sampling_rate (float): 신호의 샘플링 레이트 (Hz)
        """
        self.sampling_rate = sampling_rate
        
    def bandpass_filter(self, data: np.ndarray, lowcut: float = 5.0, highcut: float = 100.0, order: int = 4) -> np.ndarray:
        """
        신호에 밴드패스 필터 적용
        
        매개변수:
            data (np.ndarray): 처리할 신호 데이터
            lowcut (float): 저역 차단 주파수 (Hz)
            highcut (float): 고역 차단 주파수 (Hz)
            order (int): 필터 차수
            
        반환값:
            np.ndarray: 필터링된 신호
        """
        nyq = 0.5 * self.sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
    
    def notch_filter(self, data: np.ndarray, freq: float = 60.0, q: float = 30.0) -> np.ndarray:
        """
        노치 필터로 특정 주파수(예: 60Hz 전원 노이즈) 제거
        
        매개변수:
            data (np.ndarray): 처리할 신호 데이터
            freq (float): 제거할 주파수 (Hz)
            q (float): 필터의 품질 계수
            
        반환값:
            np.ndarray: 필터링된 신호
        """
        nyq = 0.5 * self.sampling_rate
        w0 = freq / nyq
        b, a = signal.iirnotch(w0, q)
        return signal.filtfilt(b, a, data)
    
    def extract_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        신호에서 주요 특성 추출
        
        매개변수:
            data (np.ndarray): 분석할 신호 데이터
            
        반환값:
            Dict[str, float]: 추출된 특성들의 딕셔너리
        """
        features = {
            'mean': np.mean(data),
            'std': np.std(data),
            'max': np.max(data),
            'min': np.min(data),
            'range': np.ptp(data),
            'rms': np.sqrt(np.mean(np.square(data))),
            'energy': np.sum(np.square(data))
        }
        
        # 주파수 영역 특성
        freqs, psd = signal.welch(data, self.sampling_rate)
        dominant_freq_idx = np.argmax(psd)
        
        features.update({
            'dominant_frequency': freqs[dominant_freq_idx],
            'spectral_entropy': self._spectral_entropy(psd)
        })
        
        return features
    
    def _spectral_entropy(self, psd: np.ndarray) -> float:
        """
        전력 스펙트럼 밀도에서 스펙트럼 엔트로피 계산
        
        매개변수:
            psd (np.ndarray): 전력 스펙트럼 밀도
            
        반환값:
            float: 스펙트럼 엔트로피
        """
        psd_norm = psd / np.sum(psd)
        psd_norm = psd_norm[psd_norm > 0]  # 로그에 0이 들어가는 것 방지
        return -np.sum(psd_norm * np.log2(psd_norm))
    
    def detect_spikes(self, data: np.ndarray, threshold: float = 3.0) -> List[int]:
        """
        역치 기반 스파이크 검출
        
        매개변수:
            data (np.ndarray): 분석할 신호 데이터
            threshold (float): 표준편차 단위의 역치
            
        반환값:
            List[int]: 검출된 스파이크의 인덱스
        """
        mean = np.mean(data)
        std = np.std(data)
        threshold_value = mean + threshold * std
        
        # 스파이크 위치 찾기
        spike_indices = []
        for i in range(1, len(data) - 1):
            if data[i] > threshold_value and data[i] > data[i-1] and data[i] > data[i+1]:
                spike_indices.append(i)
                
        return spike_indices
    
    def calculate_firing_rate(self, spike_indices: List[int], window_size: int = 1000) -> np.ndarray:
        """
        스파이크 인덱스에서 발화율 계산
        
        매개변수:
            spike_indices (List[int]): 스파이크 인덱스 리스트
            window_size (int): 발화율 계산 윈도우 크기 (샘플 단위)
            
        반환값:
            np.ndarray: 시간에 따른 발화율
        """
        if not spike_indices:
            return np.array([])
            
        max_time = max(spike_indices)
        bins = np.arange(0, max_time + window_size, window_size)
        
        # 히스토그램으로 각 빈에 들어가는 스파이크 수 계산
        hist, _ = np.histogram(spike_indices, bins=bins)
        
        # 샘플 단위에서 초 단위로 변환
        firing_rate = hist * (self.sampling_rate / window_size)
        
        return firing_rate
