"""
생체신호 전처리 모듈

이 모듈은 다양한 생체신호(EMG, EEG, ENG, 혈류 등)의 전처리를 위한 기능을 제공합니다.
노이즈 제거, 필터링, 신호 정규화 등의 기능이 포함됩니다.
"""

import numpy as np
import scipy.signal as signal
from scipy.signal import butter, filtfilt, iirnotch, welch, coherence
import pywt
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Dict, Optional, Any
import pandas as pd
from sklearn.decomposition import PCA, FastICA


class BiosignalPreprocessor:
    """다양한 생체신호 전처리를 위한 클래스"""
    
    def __init__(self, sampling_rate: float):
        """초기화 함수
        
        Args:
            sampling_rate (float): 신호의 샘플링 레이트 (Hz)
        """
        self.fs = sampling_rate
        
    def bandpass_filter(self, data: np.ndarray, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
        """버터워스 밴드패스 필터 적용
        
        Args:
            data (np.ndarray): 입력 신호
            lowcut (float): 저역 차단 주파수 (Hz)
            highcut (float): 고역 차단 주파수 (Hz)
            order (int, optional): 필터 차수. 기본값은 4.
            
        Returns:
            np.ndarray: 필터링된 신호
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    def notch_filter(self, data: np.ndarray, notch_freq: float, quality_factor: float = 30.0) -> np.ndarray:
        """노치 필터 적용 (전원 노이즈 제거용)
        
        Args:
            data (np.ndarray): 입력 신호
            notch_freq (float): 제거할 주파수 (Hz)
            quality_factor (float, optional): 품질 계수. 기본값은 30.0.
            
        Returns:
            np.ndarray: 필터링된 신호
        """
        nyq = 0.5 * self.fs
        notch_freq_normalized = notch_freq / nyq
        
        b, a = iirnotch(notch_freq_normalized, quality_factor)
        return filtfilt(b, a, data)
    
    def wavelet_denoising(self, data: np.ndarray, wavelet: str = 'db4', 
                         level: int = 3, threshold_mode: str = 'soft') -> np.ndarray:
        """웨이블릿 변환을 이용한 신호 잡음 제거
        
        Args:
            data (np.ndarray): 입력 신호
            wavelet (str, optional): 웨이블릿 유형. 기본값은 'db4'.
            level (int, optional): 분해 레벨. 기본값은 3.
            threshold_mode (str, optional): 역치 모드, 'soft' 또는 'hard'. 기본값은 'soft'.
            
        Returns:
            np.ndarray: 잡음이 제거된 신호
        """
        # 웨이블릿 변환 수행
        coeffs = pywt.wavedec(data, wavelet, level=level)
        
        # 각 레벨에 대한 임계값 계산 및 적용
        for i in range(1, len(coeffs)):
            # 임계값 계산 (Universal threshold)
            sigma = (np.median(np.abs(coeffs[i])) / 0.6745)
            threshold = sigma * np.sqrt(2 * np.log(len(data)))
            
            # 임계값 적용
            coeffs[i] = pywt.threshold(coeffs[i], threshold, mode=threshold_mode)
        
        # 역변환 수행
        denoised_data = pywt.waverec(coeffs, wavelet)
        
        # 원래 신호 길이에 맞게 조정 (경우에 따라 길이가 달라질 수 있음)
        return denoised_data[:len(data)]
    
    def adaptive_filter(self, primary: np.ndarray, reference: np.ndarray, 
                       mu: float = 0.01, filter_length: int = 32) -> np.ndarray:
        """적응형 노이즈 제거 필터 (LMS 알고리즘)
        
        Args:
            primary (np.ndarray): 주 신호 (노이즈가 포함된 대상 신호)
            reference (np.ndarray): 참조 신호 (노이즈 추정치)
            mu (float, optional): 스텝 크기 파라미터. 기본값은 0.01.
            filter_length (int, optional): 필터 길이. 기본값은 32.
            
        Returns:
            np.ndarray: 필터링된 신호
        """
        # 필터 계수 초기화
        w = np.zeros(filter_length)
        output = np.zeros(len(primary))
        
        # LMS 알고리즘 적용
        for n in range(filter_length, len(primary)):
            x = reference[n-filter_length:n][::-1]  # 참조 신호 버퍼
            y = np.dot(w, x)  # 필터 출력
            output[n] = primary[n] - y  # 오차 계산
            w = w + mu * output[n] * x  # 필터 계수 업데이트
        
        return output
    
    def process_emg(self, emg_data: np.ndarray) -> np.ndarray:
        """EMG 신호 처리 파이프라인
        
        Args:
            emg_data (np.ndarray): 원시 EMG 데이터
            
        Returns:
            np.ndarray: 처리된 EMG 신호
        """
        # 전원 노이즈 제거 (50Hz 또는 60Hz)
        emg_filtered = self.notch_filter(emg_data, 50.0)
        
        # 대역 통과 필터링 (EMG 일반적인 대역: 20-450Hz)
        emg_filtered = self.bandpass_filter(emg_filtered, 20.0, 450.0)
        
        # 웨이블릿 잡음 제거
        emg_denoised = self.wavelet_denoising(emg_filtered)
        
        return emg_denoised
    
    def process_eng(self, eng_data: np.ndarray) -> np.ndarray:
        """ENG(신경 전도) 신호 처리 파이프라인
        
        Args:
            eng_data (np.ndarray): 원시 ENG 데이터
            
        Returns:
            np.ndarray: 처리된 ENG 신호
        """
        # 전원 노이즈 제거
        eng_filtered = self.notch_filter(eng_data, 50.0)
        
        # 대역 통과 필터링 (ENG 일반적인 대역: 100-2000Hz)
        eng_filtered = self.bandpass_filter(eng_filtered, 100.0, 2000.0)
        
        # 웨이블릿 잡음 제거
        eng_denoised = self.wavelet_denoising(eng_filtered, wavelet='sym8')
        
        return eng_denoised
    
    def process_blood_flow(self, bf_data: np.ndarray) -> np.ndarray:
        """혈류 신호 처리 파이프라인
        
        Args:
            bf_data (np.ndarray): 원시 혈류 데이터
            
        Returns:
            np.ndarray: 처리된 혈류 신호
        """
        # 저주파 노이즈 제거 (고대역 통과 필터)
        bf_filtered = self.bandpass_filter(bf_data, 0.05, 15.0)
        
        # 이동 평균 필터 적용 (추가적인 평활화)
        window_size = int(self.fs / 5)  # 0.2초 윈도우
        bf_smoothed = np.convolve(bf_filtered, np.ones(window_size)/window_size, mode='same')
        
        return bf_smoothed
    
    def process_cytokine_signal(self, cytokine_data: np.ndarray) -> np.ndarray:
        """염증성/항염증성 사이토카인 신호 처리 파이프라인
        
        Args:
            cytokine_data (np.ndarray): 원시 사이토카인 데이터
            
        Returns:
            np.ndarray: 처리된 사이토카인 신호
        """
        # 이상치 제거 (중간값 필터)
        window_size = 5  # 중간값 필터 윈도우 크기
        cytokine_filtered = signal.medfilt(cytokine_data, window_size)
        
        # 저주파 필터링 (빠른 변동 제거)
        cytokine_filtered = self.bandpass_filter(cytokine_filtered, 0.01, 0.5)
        
        # 이동 평균 필터 적용
        window_size = int(self.fs / 2)  # 0.5초 윈도우
        cytokine_smoothed = np.convolve(cytokine_filtered, np.ones(window_size)/window_size, mode='same')
        
        return cytokine_smoothed
    
    def process_neurotrophic_factors(self, factor_data: np.ndarray) -> np.ndarray:
        """신경영양인자(BDNF, GDNF, NGF 등) 신호 처리 파이프라인
        
        Args:
            factor_data (np.ndarray): 원시 신경영양인자 데이터
            
        Returns:
            np.ndarray: 처리된 신경영양인자 신호
        """
        # 이상치 제거 및 평활화
        window_size = 5
        factor_filtered = signal.medfilt(factor_data, window_size)
        
        # 저주파 통과 필터 (장기 트렌드 분석용)
        factor_filtered = self.bandpass_filter(factor_filtered, 0.005, 0.2)
        
        return factor_filtered
    
    def segment_signal(self, data: np.ndarray, window_size: float, overlap: float = 0.5) -> List[np.ndarray]:
        """신호를 분석 가능한 세그먼트로 나눔
        
        Args:
            data (np.ndarray): 입력 신호
            window_size (float): 윈도우 크기 (초)
            overlap (float, optional): 윈도우 간 중첩 비율 (0~1). 기본값은 0.5.
            
        Returns:
            List[np.ndarray]: 세그먼트 목록
        """
        samples_per_window = int(window_size * self.fs)
        step = int(samples_per_window * (1 - overlap))
        
        segments = []
        for i in range(0, len(data) - samples_per_window + 1, step):
            segments.append(data[i:i + samples_per_window])
        
        return segments
    
    def compute_spectral_features(self, signal_data: np.ndarray, 
                                 bands: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, float]:
        """신호의 주파수 대역별 특징 계산
        
        Args:
            signal_data (np.ndarray): 입력 신호
            bands (Dict[str, Tuple[float, float]], optional): 분석할 주파수 대역. 
                                                             기본값은 신경 신호 표준 대역.
                                                             
        Returns:
            Dict[str, float]: 대역별 파워 및 상대적 파워 특징
        """
        if bands is None:
            # 신경 신호 표준 대역
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 100)
            }
        
        # 주파수 및 파워 스펙트럴 밀도 계산
        freqs, psd = welch(signal_data, fs=self.fs, nperseg=min(1024, len(signal_data)))
        
        # 전체 파워 계산
        total_power = np.sum(psd)
        
        # 각 대역별 특징 계산
        features = {}
        
        for band_name, (low_freq, high_freq) in bands.items():
            # 해당 주파수 대역의 인덱스 찾기
            idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
            
            # 대역 파워 계산
            band_power = np.sum(psd[idx_band])
            
            # 절대 파워
            features[f'{band_name}_power'] = band_power
            
            # 상대 파워
            features[f'{band_name}_rel_power'] = band_power / total_power if total_power > 0 else 0
        
        # 주파수 대역 비율 (특정 비율이 특정 신경 상태와 관련 있음)
        if 'theta_power' in features and 'beta_power' in features:
            features['theta_beta_ratio'] = features['theta_power'] / features['beta_power'] if features['beta_power'] > 0 else 0
            
        if 'alpha_power' in features and 'beta_power' in features:
            features['alpha_beta_ratio'] = features['alpha_power'] / features['beta_power'] if features['beta_power'] > 0 else 0
        
        # 중심 주파수 및 평균 주파수
        if total_power > 0:
            features['mean_frequency'] = np.sum(freqs * psd) / total_power
            
            # 중심 주파수 (누적 파워의 50%에 해당하는 주파수)
            cumulative_power = np.cumsum(psd) / total_power
            features['median_frequency'] = freqs[np.where(cumulative_power >= 0.5)[0][0]]
        else:
            features['mean_frequency'] = 0
            features['median_frequency'] = 0
        
        return features
    
    def compute_envelope(self, signal_data: np.ndarray, method: str = 'hilbert') -> np.ndarray:
        """신호의 포락선(envelope) 계산
        
        Args:
            signal_data (np.ndarray): 입력 신호
            method (str, optional): 포락선 계산 방법. 'hilbert' 또는 'rms'. 기본값은 'hilbert'.
            
        Returns:
            np.ndarray: 신호 포락선
        """
        if method == 'hilbert':
            # 힐버트 변환을 이용한 포락선 계산
            analytic_signal = signal.hilbert(signal_data)
            envelope = np.abs(analytic_signal)
        elif method == 'rms':
            # RMS 기반 포락선
            window_size = int(self.fs / 10)  # 0.1초 윈도우
            envelope = np.zeros_like(signal_data)
            
            for i in range(len(signal_data)):
                start = max(0, i - window_size // 2)
                end = min(len(signal_data), i + window_size // 2)
                envelope[i] = np.sqrt(np.mean(signal_data[start:end] ** 2))
        else:
            raise ValueError(f"지원하지 않는 포락선 계산 방법: {method}")
        
        return envelope
    
    def compute_neural_connectivity(self, signal1: np.ndarray, signal2: np.ndarray, 
                                   method: str = 'coherence', 
                                   freq_range: Tuple[float, float] = None) -> Dict[str, Any]:
        """두 신경 신호 간의 연결성(connectivity) 측정
        
        Args:
            signal1 (np.ndarray): 첫 번째 신호
            signal2 (np.ndarray): 두 번째 신호
            method (str, optional): 연결성 측정 방법. 'coherence', 'correlation', 'plv'. 기본값은 'coherence'.
            freq_range (Tuple[float, float], optional): 분석할 주파수 범위 (Hz). 기본값은 None (전체 범위).
            
        Returns:
            Dict[str, Any]: 연결성 측정 결과
        """
        results = {}
        
        if method == 'coherence':
            # 코히어런스 계산 (주파수 도메인 상관관계)
            freqs, coh = coherence(signal1, signal2, fs=self.fs, nperseg=min(1024, len(signal1)))
            
            if freq_range is not None:
                # 지정된 주파수 범위만 선택
                mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
                freqs = freqs[mask]
                coh = coh[mask]
            
            results['freqs'] = freqs
            results['coherence'] = coh
            results['mean_coherence'] = np.mean(coh)
            
        elif method == 'correlation':
            # 시간 도메인 상관관계
            correlation = np.corrcoef(signal1, signal2)[0, 1]
            results['correlation'] = correlation
            
        elif method == 'plv':  # Phase Locking Value
            # 위상 동기화 지수 (Phase Locking Value)
            # 힐버트 변환으로 신호 위상 추출
            phase1 = np.angle(signal.hilbert(signal1))
            phase2 = np.angle(signal.hilbert(signal2))
            
            # 위상 차이 계산
            phase_diff = phase1 - phase2
            
            # PLV 계산
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            results['plv'] = plv
            
        else:
            raise ValueError(f"지원하지 않는 연결성 측정 방법: {method}")
        
        return results
    
    def extract_biomarker_features(self, biomarker_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """신경재생 관련 바이오마커 데이터에서 특징 추출
        
        Args:
            biomarker_data (Dict[str, np.ndarray]): 바이오마커 데이터 딕셔너리
                                                  (키: 바이오마커 이름, 값: 시계열 데이터)
            
        Returns:
            Dict[str, float]: 추출된 바이오마커 특징
        """
        features = {}
        
        # 재생 촉진 인자 분석
        regenerative_markers = ['BDNF', 'GDNF', 'NGF', 'GAP43', 'CREB', 'cAMP']
        
        # 염증 인자 분석
        inflammatory_markers = ['IL1b', 'TNFa', 'IL6']
        
        # 항염증 인자 분석
        anti_inflammatory_markers = ['IL10', 'TGFb', 'IL4']
        
        # 신경재생 인덱스 계산 (BDNF/TNFa 비율 등)
        # 최신 연구에 따르면 이러한 비율이 재생 잠재력과 관련이 있음
        
        for marker, data in biomarker_data.items():
            if len(data) == 0:
                continue
                
            # 기본 통계 특징
            features[f'{marker}_mean'] = np.mean(data)
            features[f'{marker}_std'] = np.std(data)
            features[f'{marker}_max'] = np.max(data)
            features[f'{marker}_min'] = np.min(data)
            
            # 변화율 특징 (시간에 따른 변화)
            if len(data) > 1:
                # 마지막 50% 데이터의 선형 추세 계산
                half_len = len(data) // 2
                if half_len > 0:
                    x = np.arange(half_len)
                    y = data[-half_len:]
                    slope, _ = np.polyfit(x, y, 1)
                    features[f'{marker}_trend'] = slope
                
                # 초기값 대비 최근값 변화 비율
                if data[0] != 0:
                    features[f'{marker}_change_ratio'] = data[-1] / data[0]
                else:
                    features[f'{marker}_change_ratio'] = 0
        
        # 바이오마커 비율 특징 (특정 비율이 재생 결과와 상관관계가 높음)
        if 'BDNF_mean' in features and 'TNFa_mean' in features and features['TNFa_mean'] != 0:
            features['BDNF_TNFa_ratio'] = features['BDNF_mean'] / features['TNFa_mean']
            
        if 'IL10_mean' in features and 'IL1b_mean' in features and features['IL1b_mean'] != 0:
            features['IL10_IL1b_ratio'] = features['IL10_mean'] / features['IL1b_mean']
        
        # 재생 지수 계산 (재생 촉진 인자와 억제 인자의 종합적 점수)
        regenerative_score = 0
        regenerative_count = 0
        
        inflammatory_score = 0
        inflammatory_count = 0
        
        for marker in regenerative_markers:
            key = f'{marker}_mean'
            if key in features:
                regenerative_score += features[key]
                regenerative_count += 1
                
        for marker in inflammatory_markers:
            key = f'{marker}_mean'
            if key in features:
                inflammatory_score += features[key]
                inflammatory_count += 1
        
        if regenerative_count > 0:
            features['regenerative_score'] = regenerative_score / regenerative_count
            
        if inflammatory_count > 0:
            features['inflammatory_score'] = inflammatory_score / inflammatory_count
            
        if 'regenerative_score' in features and 'inflammatory_score' in features and features['inflammatory_score'] != 0:
            features['regeneration_index'] = features['regenerative_score'] / features['inflammatory_score']
        
        return features
    
    def compute_independent_components(self, multi_channel_data: np.ndarray, n_components: int = 5) -> Dict[str, np.ndarray]:
        """다중 채널 신호에서 독립 성분 분석 (ICA)
        
        Args:
            multi_channel_data (np.ndarray): 다중 채널 신호 데이터 (channels × samples)
            n_components (int, optional): 추출할 독립 성분 수. 기본값은 5.
            
        Returns:
            Dict[str, np.ndarray]: ICA 결과 (독립 성분, 믹싱 행렬, 역믹싱 행렬)
        """
        # FastICA 적용
        ica = FastICA(n_components=min(n_components, multi_channel_data.shape[0]))
        components = ica.fit_transform(multi_channel_data.T).T
        
        return {
            'components': components,
            'mixing_matrix': ica.mixing_,
            'unmixing_matrix': ica.components_
        }
    
    def extract_action_potential_features(self, eng_data: np.ndarray, 
                                         threshold_factor: float = 4.0) -> Dict[str, Any]:
        """신경 신호에서 활동 전위(action potential) 특징 추출
        
        Args:
            eng_data (np.ndarray): 처리된 ENG 데이터
            threshold_factor (float, optional): 역치 설정을 위한 표준편차 배수. 기본값은 4.0.
            
        Returns:
            Dict[str, Any]: 활동 전위 특징 (스파이크 횟수, 주파수, 진폭, 지속시간 등)
        """
        # 활동 전위 탐지 역치 설정
        threshold = threshold_factor * np.std(eng_data)
        
        # 피크 탐지
        peaks, properties = signal.find_peaks(eng_data, height=threshold, distance=int(0.001 * self.fs))
        
        if len(peaks) == 0:
            return {
                'spike_count': 0,
                'spike_frequency': 0,
                'mean_amplitude': 0,
                'max_amplitude': 0,
                'spikes': [],
                'spike_times': []
            }
        
        # 스파이크 시간 계산
        spike_times = peaks / self.fs  # 초 단위
        
        # 스파이크 진폭
        spike_amplitudes = properties['peak_heights']
        
        # 스파이크 주파수 (Hz)
        record_duration = len(eng_data) / self.fs  # 초 단위
        spike_frequency = len(peaks) / record_duration if record_duration > 0 else 0
        
        # 스파이크 추출 (각 스파이크 주변 5ms 포함)
        window_half_size = int(0.0025 * self.fs)  # 2.5ms (반 윈도우)
        spikes = []
        
        for peak in peaks:
            start = max(0, peak - window_half_size)
            end = min(len(eng_data), peak + window_half_size)
            spike_waveform = eng_data[start:end]
            
            # 모든 스파이크 길이 통일 (패딩)
            if len(spike_waveform) < 2 * window_half_size:
                padded_spike = np.zeros(2 * window_half_size)
                padded_spike[:len(spike_waveform)] = spike_waveform
                spike_waveform = padded_spike
                
            spikes.append(spike_waveform)
        
        return {
            'spike_count': len(peaks),
            'spike_frequency': spike_frequency,
            'mean_amplitude': np.mean(spike_amplitudes),
            'max_amplitude': np.max(spike_amplitudes),
            'spikes': spikes,
            'spike_times': spike_times
        }
        
    def calculate_neurophysiological_metrics(self, signals: Dict[str, np.ndarray]) -> Dict[str, float]:
        """신경재생 평가를 위한 종합적 신경생리학적 지표 계산
        
        Args:
            signals (Dict[str, np.ndarray]): 다양한 신경생리학적 신호 딕셔너리
                                            (예: 'emg', 'eng', 'blood_flow' 등)
            
        Returns:
            Dict[str, float]: 계산된 신경생리학적 지표들
        """
        metrics = {}
        
        # 1. 복합근육활동전위(CMAP) 분석 (운동신경 기능)
        if 'emg' in signals:
            emg_data = signals['emg']
            emg_envelope = self.compute_envelope(emg_data)
            
            metrics['cmap_amplitude'] = np.max(emg_envelope)
            
            # 반치전폭(FWHM) 계산 - 지속시간 지표
            max_idx = np.argmax(emg_envelope)
            half_max = emg_envelope[max_idx] / 2
            
            # 왼쪽 반치점 찾기
            left_idx = max_idx
            while left_idx > 0 and emg_envelope[left_idx] > half_max:
                left_idx -= 1
                
            # 오른쪽 반치점 찾기
            right_idx = max_idx
            while right_idx < len(emg_envelope) - 1 and emg_envelope[right_idx] > half_max:
                right_idx += 1
                
            metrics['cmap_duration'] = (right_idx - left_idx) / self.fs  # 초 단위
            
            # EMG 주파수 특성 (주파수 중앙값은 운동단위 구성과 관련)
            emg_spectral = self.compute_spectral_features(emg_data)
            metrics['emg_median_frequency'] = emg_spectral['median_frequency']
        
        # 2. 신경전도검사(NCS) 지표 (전도 속도, 잠복기)
        if 'eng' in signals and 'emg' in signals:
            eng_data = signals['eng']
            emg_data = signals['emg']
            
            # 상호상관(cross-correlation)으로 잠복기(latency) 추정
            correlation = signal.correlate(eng_data, emg_data, mode='full')
            max_corr_idx = np.argmax(correlation)
            latency_samples = max_corr_idx - len(eng_data) + 1
            
            # 음수 잠복기는 의미가 없으므로 보정
            if latency_samples < 0:
                latency_samples = np.argmax(correlation[len(eng_data)-1:]) + 1
                
            metrics['latency'] = latency_samples / self.fs * 1000  # 밀리초 단위
            
            # 신경 활동 특성
            ap_features = self.extract_action_potential_features(eng_data)
            metrics['neural_firing_rate'] = ap_features['spike_frequency']
            metrics['neural_signal_amplitude'] = ap_features['mean_amplitude']
        
        # 3. 감각신경활동전위(SNAP) 분석 (감각신경 기능)
        if 'snap' in signals:
            snap_data = signals['snap']
            snap_envelope = self.compute_envelope(snap_data)
            
            metrics['snap_amplitude'] = np.max(snap_envelope)
            
            # 감각신경 반응성 지표
            metrics['sensory_responsiveness'] = metrics['snap_amplitude'] / np.std(snap_data) if np.std(snap_data) > 0 else 0
        
        # 4. 혈류 및 관류 지표 (신경 재생에서 혈관신생 중요)
        if 'blood_flow' in signals:
            bf_data = signals['blood_flow']
            
            metrics['mean_blood_flow'] = np.mean(bf_data)
            metrics['blood_flow_variability'] = np.std(bf_data) / np.mean(bf_data) if np.mean(bf_data) > 0 else 0
            
            # 혈류 변동성 (리듬 분석)
            if len(bf_data) > 10 * self.fs:  # 최소 10초 데이터 필요
                bf_spectral = self.compute_spectral_features(
                    bf_data, 
                    bands={'vlf': (0.003, 0.04), 'lf': (0.04, 0.15), 'hf': (0.15, 0.4)}
                )
                metrics['blood_flow_lf_power'] = bf_spectral['lf_power']
                metrics['blood_flow_hf_power'] = bf_spectral['hf_power']
                
                if bf_spectral['lf_power'] > 0:
                    metrics['blood_flow_hf_lf_ratio'] = bf_spectral['hf_power'] / bf_spectral['lf_power']
        
        # 5. 신경-근육 전달 지표 (신경근접합부 기능)
        if 'emg' in signals and 'eng' in signals:
            # 신경-근육 연결 효율성 (ENG-EMG 코히어런스)
            coherence_result = self.compute_neural_connectivity(
                signals['eng'], signals['emg'], method='coherence', 
                freq_range=(10, 100)  # 운동신경-근육 관련 주파수 대역
            )
            metrics['neuromuscular_coherence'] = coherence_result['mean_coherence']
        
        # 6. 신경-혈관 상호작용 지표
        if 'eng' in signals and 'blood_flow' in signals:
            # 신경활동-혈류 관계 (지연된 상관관계)
            bf_data = signals['blood_flow']
            eng_data = signals['eng']
            
            # 신경 활동 포락선
            eng_envelope = self.compute_envelope(eng_data)
            
            # 다양한 지연에 대한 상관관계 계산 (최대 5초까지)
            max_lag = min(int(5 * self.fs), len(eng_envelope) // 2)
            lag_range = np.arange(-max_lag, max_lag + 1)
            
            correlations = []
            for lag in lag_range:
                if lag > 0:
                    corr = np.corrcoef(eng_envelope[:-lag], bf_data[lag:])[0, 1]
                elif lag < 0:
                    corr = np.corrcoef(eng_envelope[-lag:], bf_data[:lag])[0, 1]
                else:
                    corr = np.corrcoef(eng_envelope, bf_data)[0, 1]
                    
                correlations.append(corr)
            
            # 최대 상관관계 및 해당 지연시간
            max_corr_idx = np.nanargmax(correlations)
            max_correlation = correlations[max_corr_idx]
            optimal_lag = lag_range[max_corr_idx] / self.fs  # 초 단위
            
            metrics['neurovascular_coupling'] = max_correlation
            metrics['neurovascular_delay'] = optimal_lag
        
        # 7. 종합 신경재생 지수 (여러 지표의 가중 평균)
        # 최신 연구 기반 가중치 설정
        weights = {
            'cmap_amplitude': 0.20,
            'neural_firing_rate': 0.15,
            'neuromuscular_coherence': 0.15,
            'snap_amplitude': 0.15,
            'mean_blood_flow': 0.10,
            'neurovascular_coupling': 0.15,
            'latency': -0.10  # 잠복기는 낮을수록 좋음 (음수 가중치)
        }
        
        # 가용한 지표만 사용하여 종합 지수 계산
        total_weight = 0
        regeneration_index = 0
        
        for metric, weight in weights.items():
            if metric in metrics:
                # 잠복기는 역수로 변환 (값이 낮을수록 좋음)
                if metric == 'latency' and metrics[metric] > 0:
                    value = 1.0 / metrics[metric]
                else:
                    value = metrics[metric]
                
                # 가중합 계산
                regeneration_index += abs(weight) * value
                total_weight += abs(weight)
        
        if total_weight > 0:
            metrics['neural_regeneration_index'] = regeneration_index / total_weight
        
        return metrics
    
    def plot_signal_comparison(self, original: np.ndarray, processed: np.ndarray, 
                              title: str = "Signal Comparison", time_unit: str = "seconds"):
        """원본 신호와 처리된 신호 비교 시각화
        
        Args:
            original (np.ndarray): 원본 신호
            processed (np.ndarray): 처리된 신호
            title (str, optional): 그래프 제목. 기본값은 "Signal Comparison".
            time_unit (str, optional): 시간 단위. 기본값은 "seconds".
        """
        # 시간 배열 생성
        time = np.arange(len(original)) / self.fs
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(time, original)
        plt.title(f"Original Signal")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(time[:len(processed)], processed)
        plt.title(f"Processed Signal")
        plt.xlabel(f"Time ({time_unit})")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def analyze_frequency_spectrum(self, signal: np.ndarray, plot: bool = False) -> Dict[str, np.ndarray]:
        """신호의 주파수 스펙트럼 분석
        
        Args:
            signal (np.ndarray): 입력 신호
            plot (bool, optional): 스펙트럼 시각화 여부. 기본값은 False.
            
        Returns:
            Dict[str, np.ndarray]: 주파수 및 파워 스펙트럼 정보
        """
        # FFT 수행
        n = len(signal)
        fft_result = np.fft.rfft(signal)
        fft_magnitude = np.abs(fft_result)
        fft_power = fft_magnitude ** 2 / n
        freqs = np.fft.rfftfreq(n, 1.0/self.fs)
        
        # 필요시 시각화
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(freqs, fft_power)
            plt.title("Power Spectrum")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        return {
            "frequencies": freqs,
            "power": fft_power,
            "magnitude": fft_magnitude
        }
    
    def detect_artifacts(self, signal: np.ndarray, threshold_std: float = 3.0) -> List[Tuple[int, int]]:
        """신호에서 아티팩트 탐지
        
        Args:
            signal (np.ndarray): 입력 신호
            threshold_std (float, optional): 표준편차 기반 역치 계수. 기본값은 3.0.
            
        Returns:
            List[Tuple[int, int]]: 아티팩트 시작-끝 인덱스 목록
        """
        # 기본 통계량 계산
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        
        # 역치 설정
        upper_threshold = signal_mean + threshold_std * signal_std
        lower_threshold = signal_mean - threshold_std * signal_std
        
        # 역치를 벗어나는 지점 탐지
        artifacts = []
        in_artifact = False
        start_idx = 0
        
        for i, val in enumerate(signal):
            if not in_artifact and (val > upper_threshold or val < lower_threshold):
                # 아티팩트 시작
                in_artifact = True
                start_idx = i
            elif in_artifact and (val <= upper_threshold and val >= lower_threshold):
                # 아티팩트 종료
                in_artifact = False
                # 너무 짧은 아티팩트는 무시 (5포인트 미만)
                if i - start_idx >= 5:
                    artifacts.append((start_idx, i))
        
        # 마지막 아티팩트가 끝나지 않은 경우
        if in_artifact and len(signal) - start_idx >= 5:
            artifacts.append((start_idx, len(signal)-1))
        
        return artifacts
    
    def interpolate_artifacts(self, signal: np.ndarray, artifacts: List[Tuple[int, int]]) -> np.ndarray:
        """탐지된 아티팩트를 선형 보간법으로 처리
        
        Args:
            signal (np.ndarray): 입력 신호
            artifacts (List[Tuple[int, int]]): 아티팩트 시작-끝 인덱스 목록
            
        Returns:
            np.ndarray: 아티팩트가 보정된 신호
        """
        corrected_signal = signal.copy()
        
        for start, end in artifacts:
            # 아티팩트 영역의 앞뒤 값
            if start > 0 and end < len(signal) - 1:
                # 선형 보간
                for i in range(start, end + 1):
                    alpha = (i - start) / (end - start + 1)
                    corrected_signal[i] = (1 - alpha) * signal[start-1] + alpha * signal[end+1]
            elif start == 0:
                # 시작 부분 아티팩트는 다음 값으로 채움
                corrected_signal[start:end+1] = signal[end+1]
            else:
                # 끝 부분 아티팩트는 이전 값으로 채움
                corrected_signal[start:end+1] = signal[start-1]
        
        return corrected_signal
    
    def visualize_biomarker_trends(self, biomarker_data: Dict[str, np.ndarray], 
                                  time_points: Optional[np.ndarray] = None,
                                  markers: Optional[List[str]] = None,
                                  figsize: Tuple[int, int] = (12, 8)):
        """신경재생 바이오마커 트렌드 시각화
        
        Args:
            biomarker_data (Dict[str, np.ndarray]): 바이오마커 데이터
            time_points (np.ndarray, optional): 시간 포인트. 기본값은 None (인덱스 사용).
            markers (List[str], optional): 표시할 마커 목록. 기본값은 None (모든 마커).
            figsize (Tuple[int, int], optional): 그림 크기. 기본값은 (12, 8).
        """
        if not biomarker_data:
            print("바이오마커 데이터가 없습니다.")
            return
        
        # 표시할 마커 선택
        if markers is None:
            markers = list(biomarker_data.keys())
        else:
            # 존재하는 마커만 필터링
            markers = [m for m in markers if m in biomarker_data]
        
        if not markers:
            print("표시할 마커가 없습니다.")
            return
        
        # 재생 촉진/억제 마커 분류
        pro_regenerative = ['BDNF', 'GDNF', 'NGF', 'cAMP', 'GAP43', 'CREB', 'IL10', 'TGFb']
        anti_regenerative = ['TNFa', 'IL1b', 'IL6']
        
        plt.figure(figsize=figsize)
        
        # 첫 번째 그래프: 재생 촉진 마커
        plt.subplot(2, 1, 1)
        for marker in markers:
            if marker in pro_regenerative:
                data = biomarker_data[marker]
                if time_points is None:
                    time_points = np.arange(len(data))
                plt.plot(time_points, data, label=marker, linewidth=2)
        
        plt.title("Regeneration Promoting Biomarkers")
        plt.xlabel("Time" if time_points is None else "Time (days)")
        plt.ylabel("Relative Expression")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 두 번째 그래프: 재생 억제 마커
        plt.subplot(2, 1, 2)
        for marker in markers:
            if marker in anti_regenerative:
                data = biomarker_data[marker]
                if time_points is None:
                    time_points = np.arange(len(data))
                plt.plot(time_points, data, label=marker, linewidth=2, linestyle='--')
        
        plt.title("Regeneration Inhibiting Biomarkers")
        plt.xlabel("Time" if time_points is None else "Time (days)")
        plt.ylabel("Relative Expression")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # 재생 지수 그래프 (BDNF/TNFa 비율 등)
        if 'BDNF' in biomarker_data and 'TNFa' in biomarker_data:
            bdnf_data = biomarker_data['BDNF']
            tnfa_data = biomarker_data['TNFa']
            
            # 0으로 나누기 방지
            tnfa_safe = np.maximum(tnfa_data, 0.001)
            bdnf_tnfa_ratio = bdnf_data / tnfa_safe
            
            plt.figure(figsize=(10, 4))
            plt.plot(time_points, bdnf_tnfa_ratio, 'g-', linewidth=2)
            plt.title("BDNF/TNFα Ratio (Regeneration Index)")
            plt.xlabel("Time" if time_points is None else "Time (days)")
            plt.ylabel("Ratio Value")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

# 사용 예시
if __name__ == "__main__":
    # 예시 데이터 생성 (EMG 신호를 모방)
    fs = 1000  # 샘플링 레이트 1000Hz
    t = np.arange(0, 10, 1/fs)  # 0~10초 신호
    
    # 근전도 모방 (20-150Hz 기본 활동)
    emg_base = np.random.randn(len(t)) * 0.1
    emg_activity = np.zeros_like(t)
    
    # 근육 활동 시뮬레이션 (2~3초, 5~7초에 활동)
    activity_mask1 = (t >= 2) & (t <= 3)
    activity_mask2 = (t >= 5) & (t <= 7)
    emg_activity[activity_mask1] = 0.5
    emg_activity[activity_mask2] = 0.8
    
    # 노이즈 생성
    power_noise = 0.2 * np.sin(2 * np.pi * 50 * t)  # 50Hz 전원 노이즈
    high_freq_noise = 0.05 * np.sin(2 * np.pi * 500 * t)  # 고주파 노이즈
    
    # 최종 신호 생성
    emg_signal = emg_base + emg_activity + power_noise + high_freq_noise
    
    # 아티팩트 추가 (갑작스러운 움직임 등)
    artifact_mask = (t >= 4) & (t <= 4.2)
    emg_signal[artifact_mask] += 2 * np.random.randn(np.sum(artifact_mask))
    
    # 전처리 객체 생성
    preprocessor = BiosignalPreprocessor(fs)
    
    # EMG 신호 처리
    processed_emg = preprocessor.process_emg(emg_signal)
    
    # 아티팩트 감지 및 보정
    artifacts = preprocessor.detect_artifacts(emg_signal, threshold_std=2.5)
    corrected_emg = preprocessor.interpolate_artifacts(emg_signal, artifacts)
    
    # 신호 비교 시각화
    preprocessor.plot_signal_comparison(emg_signal, processed_emg, "EMG Signal Processing")
    
    # 주파수 스펙트럼 분석
    orig_spectrum = preprocessor.analyze_frequency_spectrum(emg_signal, plot=True)
    processed_spectrum = preprocessor.analyze_frequency_spectrum(processed_emg, plot=True)
    
    print(f"원본 신호 아티팩트 수: {len(artifacts)}")
    print("아티팩트 위치:", artifacts)
    
    # 신경재생 관련 바이오마커 분석 예시
    # 시뮬레이션된 바이오마커 데이터 (10일간의 측정)
    time_days = np.arange(10)
    biomarker_data = {
        'BDNF': 1.0 + 0.5 * np.exp(0.2 * time_days) + 0.1 * np.random.randn(10),
        'TNFa': 2.0 - 0.15 * time_days + 0.1 * np.random.randn(10),
        'IL10': 0.5 + 0.3 * time_days + 0.1 * np.random.randn(10),
        'IL1b': 3.0 - 0.25 * time_days + 0.2 * np.random.randn(10),
        'GDNF': 0.8 + 0.3 * np.log(1 + time_days) + 0.1 * np.random.randn(10),
        'GAP43': 0.5 + 0.4 * np.sqrt(time_days) + 0.1 * np.random.randn(10)
    }
    
    # 바이오마커 트렌드 시각화
    preprocessor.visualize_biomarker_trends(biomarker_data, time_points=time_days)
    
    # 바이오마커 특징 추출
    biomarker_features = preprocessor.extract_biomarker_features(biomarker_data)
    print("\n신경재생 바이오마커 특징:")
    for feature, value in biomarker_features.items():
        if feature in ['BDNF_TNFa_ratio', 'regeneration_index', 'regenerative_score', 'inflammatory_score']:
            print(f"{feature}: {value:.4f}")
            
    # 종합 신경생리학적 지표 계산 예시
    # 여러 생체신호 시뮬레이션
    signals = {
        'emg': emg_signal,
        'eng': np.random.randn(len(t)) * 0.1 + 0.2 * np.sin(2 * np.pi * 5 * t),
        'blood_flow': 0.5 + 0.3 * np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.random.randn(len(t))
    }
    
    # 신경생리학적 지표 계산
    metrics = preprocessor.calculate_neurophysiological_metrics(signals)
    print("\n신경생리학적 지표:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
