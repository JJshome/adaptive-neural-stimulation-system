"""
생체신호 전처리 모듈

이 모듈은 다양한 생체신호(EMG, EEG, ENG, 혈류 등)의 전처리를 위한 기능을 제공합니다.
노이즈 제거, 필터링, 신호 정규화 등의 기능이 포함됩니다.
"""

import numpy as np
import scipy.signal as signal
from scipy.signal import butter, filtfilt, iirnotch
import pywt
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Dict, Optional


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
