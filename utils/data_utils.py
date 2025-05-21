"""
데이터 처리 유틸리티 모듈

이 모듈은 신경 신호 데이터의 로드, 전처리, 특성 추출을 위한 함수들을 제공합니다.
"""

import numpy as np
import pandas as pd
import os
from scipy import signal
from sklearn.preprocessing import StandardScaler


def load_neural_data(data_path):
    """
    신경 신호 데이터 로드 함수
    
    Parameters:
    -----------
    data_path : str
        신경 신호 데이터가 저장된 경로
        
    Returns:
    --------
    data : dict
        'signals': 신경 신호 데이터 배열 (samples x channels)
        'labels': 신경 상태 레이블 배열
        'time': 시간 배열
        'channel_names': 채널 이름 리스트
    """
    try:
        # 실제 데이터 로드 로직 (예시)
        # 실제 구현에서는 파일 형식에 따라 적절한 로드 코드 작성 필요
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            raise FileNotFoundError(f"데이터 경로 '{data_path}'에 파일이 없습니다.")
        
        files = [f for f in os.listdir(data_path) if f.endswith('.csv') or f.endswith('.npy')]
        if not files:
            raise FileNotFoundError(f"데이터 경로 '{data_path}'에 CSV 또는 NPY 파일이 없습니다.")
        
        # 예제: CSV 파일 로드
        data_file = os.path.join(data_path, files[0])
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
            # 데이터 구조에 따라 적절히 처리
            signals = df.iloc[:, :-1].values  # 마지막 열을 제외한 모든 열이 신호라고 가정
            labels = df.iloc[:, -1].values    # 마지막 열이 레이블이라고 가정
            time = np.arange(len(signals)) / 100.0  # 샘플링 레이트 100Hz 가정
            channel_names = [f'채널{i+1}' for i in range(signals.shape[1])]
        
        # 예제: NPY 파일 로드
        elif data_file.endswith('.npy'):
            data_array = np.load(data_file)
            signals = data_array[:, :-1]  # 마지막 열을 제외한 모든 열이 신호라고 가정
            labels = data_array[:, -1]    # 마지막 열이 레이블이라고 가정
            time = np.arange(len(signals)) / 100.0  # 샘플링 레이트 100Hz 가정
            channel_names = [f'채널{i+1}' for i in range(signals.shape[1])]
        
        return {
            'signals': signals,
            'labels': labels,
            'time': time,
            'channel_names': channel_names
        }
    
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        # 샘플 데이터 생성
        return generate_sample_data()


def generate_sample_data(n_samples=1000, n_channels=6):
    """
    샘플 신경 신호 데이터 생성 함수
    
    Parameters:
    -----------
    n_samples : int
        생성할 샘플 수
    n_channels : int
        생성할 채널 수
        
    Returns:
    --------
    data : dict
        'signals': 신경 신호 데이터 배열 (samples x channels)
        'labels': 신경 상태 레이블 배열
        'time': 시간 배열
        'channel_names': 채널 이름 리스트
    """
    # 랜덤 시드 설정
    np.random.seed(42)
    
    # 시간 배열 생성
    time = np.linspace(0, 10, n_samples)
    
    # 신경 상태에 따른 신호 생성
    # 정상 상태: 주파수가 높고 규칙적인 신호
    normal_signal = np.sin(2 * np.pi * 5 * time) + 0.5 * np.sin(2 * np.pi * 10 * time)
    normal_signal += np.random.normal(0, 0.2, n_samples)
    
    # 손상 상태: 불규칙하고 진폭이 낮은 신호
    damaged_signal = 0.5 * np.sin(2 * np.pi * 2 * time) + 0.2 * np.sin(2 * np.pi * 7.5 * time)
    damaged_signal += np.random.normal(0, 0.5, n_samples)
    
    # 재생 상태: 규칙성이 회복되고 있는 신호
    recovery_signal = 0.8 * np.sin(2 * np.pi * 4 * time) + 0.3 * np.sin(2 * np.pi * 8.5 * time)
    recovery_signal += np.random.normal(0, 0.3, n_samples)
    
    # 여러 채널의 신호 결합
    base_signals = [normal_signal, damaged_signal, recovery_signal]
    
    # 총 n_channels 개의 채널 생성
    signals = np.zeros((n_samples, n_channels))
    for i in range(n_channels):
        # 기본 신호에 무작위 노이즈 추가
        base_idx = i % len(base_signals)
        signals[:, i] = base_signals[base_idx] + np.random.normal(0, 0.1, n_samples)
    
    # 레이블 생성 (0: 정상, 1: 손상, 2: 재생)
    labels = np.zeros(n_samples, dtype=int)
    labels[n_samples//3:(2*n_samples)//3] = 1  # 손상 상태
    labels[(2*n_samples)//3:] = 2  # 재생 상태
    
    return {
        'signals': signals,
        'labels': labels,
        'time': time,
        'channel_names': [f'채널{i+1}' for i in range(n_channels)]
    }


def preprocess_neural_signals(signals, labels, sampling_rate=100):
    """
    신경 신호 전처리 함수
    
    Parameters:
    -----------
    signals : ndarray
        신경 신호 데이터 배열 (samples x channels)
    labels : ndarray
        신경 상태 레이블 배열
    sampling_rate : int
        샘플링 레이트 (Hz)
        
    Returns:
    --------
    X : ndarray
        전처리된 특성 배열
    y : ndarray
        레이블 배열
    feature_names : list
        특성 이름 리스트
    """
    n_samples, n_channels = signals.shape
    
    # 1. 신호 전처리
    processed_signals = np.zeros_like(signals)
    for ch in range(n_channels):
        # 기준선 제거 (고역 통과 필터)
        b, a = signal.butter(4, 0.5/(sampling_rate/2), 'highpass')
        baseline_removed = signal.filtfilt(b, a, signals[:, ch])
        
        # 노이즈 제거 (60Hz 노치 필터 - 전원 노이즈)
        b, a = signal.iirnotch(60, 30, sampling_rate)
        notch_filtered = signal.filtfilt(b, a, baseline_removed)
        
        # 대역 통과 필터 (관심 주파수 대역 추출)
        b, a = signal.butter(4, [0.5/(sampling_rate/2), 100/(sampling_rate/2)], 'bandpass')
        processed_signals[:, ch] = signal.filtfilt(b, a, notch_filtered)
    
    # 2. 특성 추출
    window_size = 50  # 윈도우 크기
    step = 25  # 윈도우 스텝
    
    # 윈도우 수 계산
    n_windows = (n_samples - window_size) // step + 1
    
    # 시간 도메인 특성
    n_time_features = 5  # 채널당 시간 도메인 특성 수
    X_time = np.zeros((n_windows, n_channels * n_time_features))
    
    # 주파수 도메인 특성
    n_freq_bands = 5  # 주파수 대역 수
    X_freq = np.zeros((n_windows, n_channels * n_freq_bands))
    
    # 각 윈도우별 특성 추출
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        window = processed_signals[start:end, :]
        
        for ch in range(n_channels):
            ch_data = window[:, ch]
            
            # 시간 도메인 특성
            X_time[i, ch*n_time_features + 0] = np.mean(ch_data)  # 평균
            X_time[i, ch*n_time_features + 1] = np.std(ch_data)   # 표준편차
            X_time[i, ch*n_time_features + 2] = np.max(ch_data) - np.min(ch_data)  # 범위
            # 첨도 (분포의 뾰족함)
            X_time[i, ch*n_time_features + 3] = np.mean((ch_data - np.mean(ch_data))**4) / (np.std(ch_data)**4) if np.std(ch_data) > 0 else 0
            # 제로 교차율
            X_time[i, ch*n_time_features + 4] = np.sum(np.diff(np.signbit(ch_data))) / (len(ch_data) - 1)
            
            # 주파수 도메인 특성
            fft_result = np.abs(np.fft.rfft(ch_data))
            freqs = np.fft.rfftfreq(len(ch_data), 1/sampling_rate)
            
            # 주파수 대역 정의
            bands = [
                (0.5, 4),    # 델타
                (4, 8),      # 세타
                (8, 13),     # 알파
                (13, 30),    # 베타
                (30, 100)    # 감마
            ]
            
            # 각 대역별 파워 계산
            for j, (low, high) in enumerate(bands):
                band_mask = (freqs >= low) & (freqs <= high)
                if np.any(band_mask):
                    X_freq[i, ch*n_freq_bands + j] = np.sum(fft_result[band_mask]**2)
                else:
                    X_freq[i, ch*n_freq_bands + j] = 0
    
    # 특성 결합
    X = np.hstack((X_time, X_freq))
    
    # 각 윈도우의 레이블 (윈도우 중앙 지점의 레이블 사용)
    window_centers = [start + window_size//2 for start in range(0, n_samples - window_size + 1, step)]
    y = labels[window_centers]
    
    # 특성 이름 생성
    time_feature_names = []
    for ch in range(n_channels):
        for feature in ['mean', 'std', 'range', 'kurtosis', 'zero_crossing']:
            time_feature_names.append(f'ch{ch+1}_{feature}')
    
    freq_feature_names = []
    for ch in range(n_channels):
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            freq_feature_names.append(f'ch{ch+1}_{band}')
    
    feature_names = time_feature_names + freq_feature_names
    
    return X, y, feature_names


def save_processed_data(data, labels, feature_names, output_path):
    """
    전처리된 데이터 저장 함수
    
    Parameters:
    -----------
    data : ndarray
        전처리된 특성 배열
    labels : ndarray
        레이블 배열
    feature_names : list
        특성 이름 리스트
    output_path : str
        출력 파일 경로
    """
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 데이터 프레임 생성
    df = pd.DataFrame(data, columns=feature_names)
    df['label'] = labels
    
    # CSV로 저장
    df.to_csv(output_path, index=False)
    print(f"전처리된 데이터가 {output_path}에 저장되었습니다.")
