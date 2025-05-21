"""
신경 신호 데이터 처리 유틸리티

이 모듈은 신경 신호 데이터를 전처리하고 분석하는 기능을 제공합니다.
전기생리학적 신호의 필터링, 특성 추출, 노이즈 제거 등의 알고리즘을 구현합니다.

주요 기능:
    - 밴드패스 필터링: 특정 주파수 대역의 신호만 통과시키는 필터링
    - 노치 필터링: 특정 주파수(예: 60Hz 전원 노이즈)를 제거하는 필터링
    - 특성 추출: 신호에서 중요한 특성(평균, 표준편차, RMS 등) 추출
    - 스파이크 검출: 신경 활동 전위를 검출하는 알고리즘
    - 발화율 계산: 검출된 스파이크를 기반으로 신경 발화율 계산

사용 예시:
    ```python
    import numpy as np
    from utils.signal_processor import SignalProcessor
    
    # 1kHz 샘플링 레이트로 신호 처리기 초기화
    processor = SignalProcessor(sampling_rate=1000.0)
    
    # 임의의 신호 생성
    time = np.arange(0, 5, 1/1000)
    signal = np.sin(2 * np.pi * 5 * time) + 0.5 * np.random.randn(len(time))
    
    # 5-100Hz 밴드패스 필터 적용
    filtered_signal = processor.bandpass_filter(signal, 5.0, 100.0)
    
    # 60Hz 노치 필터 적용
    filtered_signal = processor.notch_filter(filtered_signal)
    
    # 신호 특성 추출
    features = processor.extract_features(filtered_signal)
    print(f"신호 RMS: {features['rms']:.4f}")
    
    # 스파이크 검출
    spike_indices = processor.detect_spikes(filtered_signal, threshold=3.0)
    print(f"검출된 스파이크 수: {len(spike_indices)}")
    
    # 발화율 계산 (1초 윈도우)
    firing_rate = processor.calculate_firing_rate(spike_indices, window_size=1000)
    ```

참고 자료:
    - Quiroga, R. Q., & Panzeri, S. (2013). Extracting information from neuronal populations: 
      information theory and decoding approaches. Nature Reviews Neuroscience, 14(8), 584-600.
    - Lewicki, M. S. (1998). A review of methods for spike sorting: the detection and classification
      of neural action potentials. Network: Computation in Neural Systems, 9(4), R53-R78.
"""

import numpy as np
from scipy import signal
from typing import List, Tuple, Dict, Any, Optional, Union

class SignalProcessor:
    """
    신경 신호 처리를 위한 클래스
    
    이 클래스는 전기생리학적 신경 신호를 처리하기 위한 다양한 방법을 제공합니다.
    필터링, 특성 추출, 스파이크 검출, 발화율 계산 등의 기능을 포함합니다.
    
    Attributes:
        sampling_rate (float): 신호의 샘플링 레이트 (Hz)
    """
    
    def __init__(self, sampling_rate: float = 1000.0):
        """
        SignalProcessor 클래스 초기화
        
        Args:
            sampling_rate (float, optional): 신호의 샘플링 레이트 (Hz). 기본값은 1000.0 Hz.
            
        Raises:
            ValueError: 샘플링 레이트가 0 이하인 경우 발생
        """
        if sampling_rate <= 0:
            raise ValueError("샘플링 레이트는 양수여야 합니다.")
        
        self.sampling_rate = sampling_rate
        
    def bandpass_filter(self, data: np.ndarray, 
                        lowcut: float = 5.0, 
                        highcut: float = 100.0, 
                        order: int = 4) -> np.ndarray:
        """
        신호에 밴드패스 필터 적용
        
        특정 주파수 대역(lowcut ~ highcut)의 신호만 통과시키고 나머지는 감쇠시키는 필터를 적용합니다.
        
        Args:
            data (np.ndarray): 처리할 신호 데이터, 1D 배열 형태
            lowcut (float, optional): 저역 차단 주파수 (Hz). 기본값은 5.0 Hz.
            highcut (float, optional): 고역 차단 주파수 (Hz). 기본값은 100.0 Hz.
            order (int, optional): 필터 차수. 값이 클수록 필터링 효과가 강함. 기본값은 4.
            
        Returns:
            np.ndarray: 필터링된 신호
            
        Raises:
            ValueError: lowcut이 0 미만이거나 highcut이 Nyquist 주파수 초과시 발생
            ValueError: highcut이 lowcut보다 작을 경우 발생
            
        Notes:
            이 함수는 Butterworth 필터를 사용하며, 위상 왜곡을 최소화하기 위해 
            forward-backward 필터링(filtfilt)을 적용합니다.
        """
        # 입력 유효성 검사
        if data.ndim != 1:
            if data.shape[0] == 1 or data.shape[1] == 1:
                data = data.flatten()  # 단일 차원 배열로 평탄화
            else:
                raise ValueError("입력 데이터는 1차원 배열이어야 합니다.")
        
        nyq = 0.5 * self.sampling_rate
        
        if lowcut < 0:
            raise ValueError("저역 차단 주파수는 0 이상이어야 합니다.")
        if highcut > nyq:
            raise ValueError(f"고역 차단 주파수는 Nyquist 주파수({nyq} Hz) 이하여야 합니다.")
        if highcut <= lowcut:
            raise ValueError("고역 차단 주파수는 저역 차단 주파수보다 커야 합니다.")
        
        # 필터 계수 계산
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        
        # 필터 적용
        return signal.filtfilt(b, a, data)
    
    def notch_filter(self, data: np.ndarray, 
                    freq: float = 60.0, 
                    q: float = 30.0) -> np.ndarray:
        """
        노치 필터로 특정 주파수(예: 60Hz 전원 노이즈) 제거
        
        전원선 간섭이나 특정 주파수의 노이즈를 제거하는 데 유용합니다.
        
        Args:
            data (np.ndarray): 처리할 신호 데이터, 1D 배열 형태
            freq (float, optional): 제거할 주파수 (Hz). 기본값은 60.0 Hz (북미 전원 주파수).
            q (float, optional): 필터의 품질 계수. 값이 클수록 더 좁은 대역을 제거함. 기본값은 30.0.
            
        Returns:
            np.ndarray: 필터링된 신호
            
        Raises:
            ValueError: 제거 주파수가 0 이하이거나 Nyquist 주파수 이상인 경우 발생
            
        Notes:
            이 함수는 IIR 노치 필터를 사용하여 특정 주파수를 제거합니다.
            위상 왜곡을 최소화하기 위해 forward-backward 필터링(filtfilt)을 적용합니다.
        """
        # 입력 유효성 검사
        if data.ndim != 1:
            if data.shape[0] == 1 or data.shape[1] == 1:
                data = data.flatten()  # 단일 차원 배열로 평탄화
            else:
                raise ValueError("입력 데이터는 1차원 배열이어야 합니다.")
        
        nyq = 0.5 * self.sampling_rate
        
        if freq <= 0:
            raise ValueError("제거할 주파수는 0보다 커야 합니다.")
        if freq >= nyq:
            raise ValueError(f"제거할 주파수는 Nyquist 주파수({nyq} Hz) 미만이어야 합니다.")
        
        # 필터 계수 계산
        w0 = freq / nyq
        b, a = signal.iirnotch(w0, q)
        
        # 필터 적용
        return signal.filtfilt(b, a, data)
    
    def extract_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        신호에서 주요 특성 추출
        
        시간 및 주파수 도메인에서의 다양한 특성을 계산합니다.
        
        Args:
            data (np.ndarray): 분석할 신호 데이터, 1D 배열 형태
            
        Returns:
            Dict[str, float]: 추출된 특성들을 포함한 딕셔너리
                - 'mean': 신호의 평균값
                - 'std': 신호의 표준편차
                - 'max': 신호의 최대값
                - 'min': 신호의 최소값
                - 'range': 신호의 범위 (최대값 - 최소값)
                - 'rms': 신호의 RMS(Root Mean Square) 값
                - 'energy': 신호의 에너지 (제곱합)
                - 'dominant_frequency': 가장 강한 주파수 성분
                - 'spectral_entropy': 스펙트럼 엔트로피 (신호의 복잡도 측정)
                
        Raises:
            ValueError: 입력 데이터가 비어있거나 1차원 배열이 아닌 경우 발생
            
        Notes:
            스펙트럼 엔트로피는 신호의 주파수 구성이 얼마나 불규칙한지를 측정합니다.
            값이 높을수록 더 불규칙한(무작위에 가까운) 신호를 나타냅니다.
        """
        # 입력 유효성 검사
        if data.size == 0:
            raise ValueError("입력 데이터가 비어 있습니다.")
        
        if data.ndim != 1:
            if data.shape[0] == 1 or data.shape[1] == 1:
                data = data.flatten()  # 단일 차원 배열로 평탄화
            else:
                raise ValueError("입력 데이터는 1차원 배열이어야 합니다.")
        
        # 시간 도메인 특성
        features = {
            'mean': np.mean(data),
            'std': np.std(data),
            'max': np.max(data),
            'min': np.min(data),
            'range': np.ptp(data),
            'rms': np.sqrt(np.mean(np.square(data))),
            'energy': np.sum(np.square(data))
        }
        
        # 주파수 도메인 특성
        try:
            freqs, psd = signal.welch(data, self.sampling_rate)
            dominant_freq_idx = np.argmax(psd)
            
            features.update({
                'dominant_frequency': freqs[dominant_freq_idx],
                'spectral_entropy': self._spectral_entropy(psd)
            })
        except Exception as e:
            # 주파수 도메인 분석에 실패한 경우, 해당 특성은 NaN으로 설정
            features.update({
                'dominant_frequency': float('nan'),
                'spectral_entropy': float('nan')
            })
            print(f"주파수 도메인 분석 중 오류 발생: {e}")
        
        return features
    
    def _spectral_entropy(self, psd: np.ndarray) -> float:
        """
        전력 스펙트럼 밀도에서 스펙트럼 엔트로피 계산
        
        Shannon 엔트로피 방법을 사용하여 신호의 주파수 구성의 불규칙성을 측정합니다.
        
        Args:
            psd (np.ndarray): 전력 스펙트럼 밀도
            
        Returns:
            float: 스펙트럼 엔트로피 값. 값이 높을수록 더 불규칙한 신호.
            
        Notes:
            엔트로피 계산 시 로그는 밑이 2인 로그를 사용하므로, 결과는 비트(bits) 단위입니다.
        """
        # 총 전력으로 정규화
        psd_sum = np.sum(psd)
        if psd_sum == 0:
            return 0.0  # 신호가 없는 경우
        
        psd_norm = psd / psd_sum
        
        # 0이 아닌 값만 고려 (로그에 0이 들어가는 것 방지)
        psd_norm = psd_norm[psd_norm > 0]
        
        # Shannon 엔트로피 계산
        return -np.sum(psd_norm * np.log2(psd_norm))
    
    def detect_spikes(self, data: np.ndarray, threshold: float = 3.0) -> List[int]:
        """
        역치 기반 스파이크 검출
        
        표준편차 기반 역치를 사용하여 신경 활동 전위(스파이크)를 검출합니다.
        
        Args:
            data (np.ndarray): 분석할 신호 데이터, 1D 배열 형태
            threshold (float, optional): 표준편차 단위의 역치. 기본값은 3.0.
                                        값이 클수록 더 큰 스파이크만 검출됨.
            
        Returns:
            List[int]: 검출된 스파이크의 인덱스 목록
            
        Raises:
            ValueError: 입력 데이터가 비어있거나 1차원 배열이 아닌 경우 발생
            
        Notes:
            이 함수는 해당 지점의 값이 역치를 초과하고, 이웃 값들보다 큰 경우
            해당 지점을 스파이크로 식별합니다(국소 최대값 방식).
            이는 간단한 스파이크 검출 방법으로, 실제 신경 신호에서는 더 복잡한
            알고리즘이 필요할 수 있습니다.
        """
        # 입력 유효성 검사
        if data.size == 0:
            raise ValueError("입력 데이터가 비어 있습니다.")
        
        if data.ndim != 1:
            if data.shape[0] == 1 or data.shape[1] == 1:
                data = data.flatten()  # 단일 차원 배열로 평탄화
            else:
                raise ValueError("입력 데이터는 1차원 배열이어야 합니다.")
        
        # 역치 계산
        mean = np.mean(data)
        std = np.std(data)
        threshold_value = mean + threshold * std
        
        # 스파이크 위치 찾기 (국소 최대값 조건 추가)
        spike_indices = []
        
        for i in range(1, len(data) - 1):
            if (data[i] > threshold_value and 
                data[i] > data[i-1] and 
                data[i] > data[i+1]):
                spike_indices.append(i)
                
        return spike_indices
    
    def calculate_firing_rate(self, spike_indices: List[int], 
                             window_size: int = 1000) -> np.ndarray:
        """
        스파이크 인덱스에서 발화율 계산
        
        주어진 스파이크의 시간적 분포를 기반으로 발화율을 계산합니다.
        
        Args:
            spike_indices (List[int]): 스파이크 인덱스 리스트
            window_size (int, optional): 발화율 계산 윈도우 크기 (샘플 단위). 기본값은 1000 (1초).
            
        Returns:
            np.ndarray: 시간에 따른 발화율 (Hz, 초당 스파이크 수)
            
        Notes:
            이 함수는 전체 신호를 window_size 간격의 빈으로 분할하고,
            각 빈에 포함된 스파이크 수를 계산한 후, 샘플링 레이트를 고려하여
            초당 스파이크 수(Hz)로 변환합니다.
            빈 크기가 작을수록 시간 해상도는 높아지지만 발화율 추정의 
            변동성이 증가합니다.
        """
        if not spike_indices:
            return np.array([])
            
        # 최대 시간 (마지막 스파이크 인덱스)
        max_time = max(spike_indices)
        
        # 빈 정의 (윈도우 크기 간격)
        bins = np.arange(0, max_time + window_size, window_size)
        
        # 히스토그램으로 각 빈에 들어가는 스파이크 수 계산
        hist, _ = np.histogram(spike_indices, bins=bins)
        
        # 샘플 단위에서 초 단위로 변환
        # (window_size 샘플 당 스파이크 수 → 초당 스파이크 수)
        firing_rate = hist * (self.sampling_rate / window_size)
        
        return firing_rate
