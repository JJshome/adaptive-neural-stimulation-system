#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
신경 신호 샘플 데이터 생성 스크립트

이 스크립트는 신경 전기자극 시스템 테스트 및 데모를 위한 합성 신경 신호 샘플을 생성합니다.
다양한 신경 상태를 모사하는 1,000개 이상의 샘플을 생성합니다.
"""

import os
import numpy as np
import pandas as pd
import h5py
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# 상수 정의
SAMPLING_RATE = 20000  # Hz
DURATION = 1.0  # 초
N_SAMPLES = int(SAMPLING_RATE * DURATION)
TIME = np.linspace(0, DURATION, N_SAMPLES)

# 저장 경로
BASE_DIR = Path(__file__).parent
SAMPLES_DIR = BASE_DIR / 'samples'
NEURAL_RECORDINGS_DIR = BASE_DIR / 'neural_recordings'
PROCESSED_DIR = BASE_DIR / 'processed'

# 디렉토리 생성
for directory in [SAMPLES_DIR, NEURAL_RECORDINGS_DIR, PROCESSED_DIR]:
    directory.mkdir(exist_ok=True)


def generate_normal_signal(n_samples=N_SAMPLES, noise_level=0.1):
    """
    정상 신경 신호 생성
    
    Parameters:
    - n_samples: 샘플 수
    - noise_level: 노이즈 수준 (0-1)
    
    Returns:
    - 정상 신경 신호
    """
    # 기본 리듬 활동 (5-10Hz)
    t = np.arange(n_samples) / SAMPLING_RATE
    base_rhythm = 0.5 * np.sin(2 * np.pi * 8 * t)
    
    # 랜덤 스파이크 추가
    n_spikes = int(n_samples * 0.01)  # 샘플의 약 1%에 스파이크
    spike_indices = np.random.choice(n_samples, n_spikes, replace=False)
    spikes = np.zeros(n_samples)
    for idx in spike_indices:
        # 스파이크 모양: 빠른 상승, 천천히 감소
        if idx < n_samples - 20:
            spikes[idx:idx+20] += np.exp(-np.arange(20)/5)
    
    # 배경 노이즈 추가
    noise = noise_level * np.random.randn(n_samples)
    
    # 신호 조합
    normal_signal = base_rhythm + spikes + noise
    
    # 정규화
    normal_signal = (normal_signal - normal_signal.min()) / (normal_signal.max() - normal_signal.min())
    return normal_signal


def generate_damaged_signal(n_samples=N_SAMPLES, damage_level=0.6):
    """
    손상된 신경 신호 생성
    
    Parameters:
    - n_samples: 샘플 수
    - damage_level: 손상 수준 (0-1)
    
    Returns:
    - 손상된 신경 신호
    """
    # 감소된 기본 리듬 활동
    t = np.arange(n_samples) / SAMPLING_RATE
    base_rhythm = 0.2 * np.sin(2 * np.pi * 5 * t)
    
    # 활동 중단 구간 추가
    silent_start = int(n_samples * 0.3)
    silent_end = int(n_samples * (0.3 + damage_level * 0.4))
    activity_mask = np.ones(n_samples)
    activity_mask[silent_start:silent_end] = 0.2
    
    # 더 적은 랜덤 스파이크 추가
    n_spikes = int(n_samples * 0.005)  # 정상의 절반
    spike_indices = np.random.choice(n_samples, n_spikes, replace=False)
    spikes = np.zeros(n_samples)
    for idx in spike_indices:
        if idx < n_samples - 20:
            spikes[idx:idx+20] += 0.6 * np.exp(-np.arange(20)/5)  # 감소된 진폭
    
    # 더 높은 노이즈 수준
    noise = 0.2 * np.random.randn(n_samples)
    
    # 신호 조합
    damaged_signal = (base_rhythm + spikes) * activity_mask + noise
    
    # 정규화
    damaged_signal = (damaged_signal - damaged_signal.min()) / (damaged_signal.max() - damaged_signal.min())
    return damaged_signal


def generate_stimulation_response(n_samples=N_SAMPLES, stim_frequency=50):
    """
    전기자극 반응 신호 생성
    
    Parameters:
    - n_samples: 샘플 수
    - stim_frequency: 자극 주파수 (Hz)
    
    Returns:
    - 자극 반응 신호
    """
    t = np.arange(n_samples) / SAMPLING_RATE
    
    # 자극 아티팩트 생성 (일정한 간격으로 발생하는 짧은 펄스)
    stim_interval = SAMPLING_RATE / stim_frequency
    stim_indices = np.arange(0, n_samples, int(stim_interval))
    stim_artifact = np.zeros(n_samples)
    
    for idx in stim_indices:
        if idx < n_samples - 10:
            # 자극 아티팩트 모양: 빠른 상승, 빠른 하강, 약간의 반동
            stim_artifact[idx:idx+10] += np.array([0, 1, 0.8, -0.5, -0.3, -0.1, 0, 0, 0, 0])
    
    # 자극 후 신경 활동 증가 (각 자극 후 발생)
    post_stim_response = np.zeros(n_samples)
    for idx in stim_indices:
        if idx < n_samples - 100:
            # 자극 후 20ms 지연된 반응, 약 80ms 지속
            response_start = idx + 20
            response = 0.5 * np.exp(-np.arange(80)/20) * np.sin(2 * np.pi * 20 * np.arange(80) / SAMPLING_RATE)
            post_stim_response[response_start:response_start+80] += response
    
    # 기본 신경 활동 및 노이즈
    base_activity = 0.3 * np.sin(2 * np.pi * 7 * t)
    noise = 0.1 * np.random.randn(n_samples)
    
    # 신호 조합
    stim_response = stim_artifact + post_stim_response + base_activity + noise
    
    # 정규화
    stim_response = (stim_response - stim_response.min()) / (stim_response.max() - stim_response.min())
    return stim_response


def generate_noisy_signal(n_samples=N_SAMPLES):
    """
    잡음이 많은 신호 생성
    
    Parameters:
    - n_samples: 샘플 수
    
    Returns:
    - 잡음이 많은 신호
    """
    t = np.arange(n_samples) / SAMPLING_RATE
    
    # 기본 신호
    base_signal = 0.4 * np.sin(2 * np.pi * 6 * t)
    
    # 전원선 노이즈 (60Hz)
    powerline_noise = 0.2 * np.sin(2 * np.pi * 60 * t)
    
    # 고주파 노이즈
    high_freq_noise = 0.15 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.sin(2 * np.pi * 300 * t)
    
    # 저주파 드리프트
    drift = 0.3 * np.sin(2 * np.pi * 0.2 * t)
    
    # 임의 노이즈
    random_noise = 0.2 * np.random.randn(n_samples)
    
    # 노이즈 특성이 다른 구간
    noise_mask = np.ones((3, n_samples))
    segment_length = n_samples // 3
    noise_mask[0, segment_length:] = 0  # 첫 구간: 전원선 노이즈 집중
    noise_mask[1, :segment_length] = 0
    noise_mask[1, 2*segment_length:] = 0  # 두번째 구간: 고주파 노이즈 집중
    noise_mask[2, :2*segment_length] = 0  # 세번째 구간: 저주파 드리프트 집중
    
    # 신호 조합
    noisy_signal = base_signal + \
                  powerline_noise * noise_mask[0] + \
                  high_freq_noise * noise_mask[1] + \
                  drift * noise_mask[2] + \
                  random_noise
    
    # 정규화
    noisy_signal = (noisy_signal - noisy_signal.min()) / (noisy_signal.max() - noisy_signal.min())
    return noisy_signal


def generate_multichannel_signal(n_samples=N_SAMPLES, n_channels=16):
    """
    다채널 신경 신호 생성
    
    Parameters:
    - n_samples: 샘플 수
    - n_channels: 채널 수
    
    Returns:
    - 다채널 신경 신호 (채널 수 x 샘플 수)
    """
    t = np.arange(n_samples) / SAMPLING_RATE
    
    # 공통 리듬 활동 (모든 채널에 영향)
    common_rhythm = 0.3 * np.sin(2 * np.pi * 8 * t)
    
    # 채널별 데이터 생성
    multichannel_data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # 채널별 고유 리듬 (약간 다른 주파수)
        ch_freq = 5 + ch * 0.5  # 5-12.5 Hz
        ch_rhythm = 0.2 * np.sin(2 * np.pi * ch_freq * t + np.random.rand() * 2 * np.pi)
        
        # 채널별 스파이크 패턴
        n_spikes = int(n_samples * 0.005) + int(n_samples * 0.01 * np.random.rand())  # 채널별 다른 스파이크 수
        spike_indices = np.random.choice(n_samples, n_spikes, replace=False)
        spikes = np.zeros(n_samples)
        for idx in spike_indices:
            if idx < n_samples - 20:
                spikes[idx:idx+20] += (0.5 + 0.5 * np.random.rand()) * np.exp(-np.arange(20)/5)
        
        # 채널별 노이즈 수준
        noise_level = 0.05 + 0.15 * np.random.rand()  # 0.05-0.2 범위의 노이즈
        noise = noise_level * np.random.randn(n_samples)
        
        # 채널 신호 조합
        ch_signal = common_rhythm + ch_rhythm + spikes + noise
        
        # 정규화
        ch_signal = (ch_signal - ch_signal.min()) / (ch_signal.max() - ch_signal.min())
        multichannel_data[ch] = ch_signal
    
    return multichannel_data


def save_signal_as_csv(signal, filename, multichannel=False):
    """
    신호를 CSV 파일로 저장
    
    Parameters:
    - signal: 저장할 신호 데이터
    - filename: 파일 이름
    - multichannel: 다채널 신호 여부
    """
    if multichannel:
        # 다채널 신호
        n_channels, n_samples = signal.shape
        df = pd.DataFrame(signal.T, columns=[f'channel_{i+1}' for i in range(n_channels)])
        df.insert(0, 'time', np.arange(n_samples) / SAMPLING_RATE)
    else:
        # 단일 채널 신호
        df = pd.DataFrame({
            'time': np.arange(len(signal)) / SAMPLING_RATE,
            'amplitude': signal
        })
    
    df.to_csv(filename, index=False)
    print(f"저장 완료: {filename}")


def save_signal_as_npy(signal, filename):
    """
    신호를 NumPy 배열로 저장
    
    Parameters:
    - signal: 저장할 신호 데이터
    - filename: 파일 이름
    """
    np.save(filename, signal)
    print(f"저장 완료: {filename}")


def save_signal_as_h5(signal, filename, attrs=None, multichannel=False):
    """
    신호를 HDF5 파일로 저장
    
    Parameters:
    - signal: 저장할 신호 데이터
    - filename: 파일 이름
    - attrs: 메타데이터 사전 (기본값: None)
    - multichannel: 다채널 신호 여부
    """
    if attrs is None:
        attrs = {}
    
    with h5py.File(filename, 'w') as f:
        # 데이터 저장
        f.create_dataset('data', data=signal)
        
        # 기본 메타데이터 추가
        f.attrs['sampling_rate'] = SAMPLING_RATE
        f.attrs['duration'] = DURATION
        
        if multichannel:
            f.attrs['n_channels'] = signal.shape[0]
        
        # 추가 메타데이터 저장
        for key, value in attrs.items():
            f.attrs[key] = value
    
    print(f"저장 완료: {filename}")


def generate_and_save_sample_set(set_name, n_samples=1000, output_dir=None):
    """
    샘플 세트 생성 및 저장
    
    Parameters:
    - set_name: 샘플 세트 이름 (normal, damaged, 등)
    - n_samples: 생성할 샘플 수
    - output_dir: 출력 디렉토리 (기본값: neural_recordings)
    """
    if output_dir is None:
        output_dir = NEURAL_RECORDINGS_DIR
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 생성 함수 선택
    if set_name == 'normal':
        generator = generate_normal_signal
        params = [{'noise_level': 0.05 + 0.15 * np.random.rand()} for _ in range(n_samples)]
    elif set_name == 'damaged':
        generator = generate_damaged_signal
        params = [{'damage_level': 0.4 + 0.5 * np.random.rand()} for _ in range(n_samples)]
    elif set_name == 'stimulation':
        generator = generate_stimulation_response
        params = [{'stim_frequency': 20 + 80 * np.random.rand()} for _ in range(n_samples)]
    elif set_name == 'noisy':
        generator = generate_noisy_signal
        params = [{}] * n_samples
    else:
        print(f"알 수 없는 샘플 세트 이름: {set_name}")
        return
    
    # 샘플 생성 및 저장
    for i in range(n_samples):
        # 신호 생성
        signal = generator(**params[i])
        
        # 파일 이름 생성
        base_filename = f"{set_name}_signal_{i+1:04d}"
        csv_filename = os.path.join(output_dir, f"{base_filename}.csv")
        npy_filename = os.path.join(output_dir, f"{base_filename}.npy")
        h5_filename = os.path.join(output_dir, f"{base_filename}.h5")
        
        # 다양한 형식으로 저장
        save_signal_as_csv(signal, csv_filename)
        save_signal_as_npy(signal, npy_filename)
        save_signal_as_h5(signal, h5_filename, attrs=params[i])
        
        # 진행 상황 표시
        if (i+1) % 100 == 0 or i+1 == n_samples:
            print(f"{set_name} 샘플 생성 진행률: {i+1}/{n_samples}")


def generate_multichannel_sample_set(n_samples=50, output_dir=None):
    """
    다채널 샘플 세트 생성 및 저장
    
    Parameters:
    - n_samples: 생성할 샘플 수
    - output_dir: 출력 디렉토리 (기본값: neural_recordings)
    """
    if output_dir is None:
        output_dir = NEURAL_RECORDINGS_DIR
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 샘플 생성 및 저장
    for i in range(n_samples):
        # 채널 수 변경 (8-32 채널)
        n_channels = np.random.choice([8, 16, 24, 32])
        
        # 다채널 신호 생성
        signal = generate_multichannel_signal(n_channels=n_channels)
        
        # 파일 이름 생성
        base_filename = f"multichannel_signal_{i+1:04d}"
        csv_filename = os.path.join(output_dir, f"{base_filename}.csv")
        npy_filename = os.path.join(output_dir, f"{base_filename}.npy")
        h5_filename = os.path.join(output_dir, f"{base_filename}.h5")
        
        # 다양한 형식으로 저장
        save_signal_as_csv(signal, csv_filename, multichannel=True)
        save_signal_as_npy(signal, npy_filename)
        save_signal_as_h5(signal, h5_filename, attrs={'n_channels': n_channels}, multichannel=True)
        
        # 진행 상황 표시
        if (i+1) % 10 == 0 or i+1 == n_samples:
            print(f"다채널 샘플 생성 진행률: {i+1}/{n_samples}")


def generate_basic_sample_set():
    """
    기본 샘플 세트 생성 및 저장 (samples 디렉토리)
    """
    # 신호 생성
    normal_signal = generate_normal_signal()
    damaged_signal = generate_damaged_signal()
    stim_response = generate_stimulation_response()
    noisy_signal = generate_noisy_signal()
    multichannel_signal = generate_multichannel_signal(n_channels=5)  # 5채널로 제한하여 CSV 크기 감소
    
    # 기본 샘플 저장 (samples 디렉토리)
    save_signal_as_csv(normal_signal, SAMPLES_DIR / 'normal_signal.csv')
    save_signal_as_csv(damaged_signal, SAMPLES_DIR / 'damaged_signal.csv')
    save_signal_as_csv(stim_response, SAMPLES_DIR / 'stimulation_response.csv')
    save_signal_as_csv(noisy_signal, SAMPLES_DIR / 'noisy_signal.csv')
    save_signal_as_csv(multichannel_signal, SAMPLES_DIR / 'multichannel_signal.csv', multichannel=True)


def generate_processed_samples(n_samples=200):
    """
    처리된 샘플 데이터 생성 (전처리된 특성 추출 결과)
    
    Parameters:
    - n_samples: 생성할 샘플 수
    """
    # 출력 디렉토리 생성
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # 특성 이름 정의
    features = [
        'mean', 'std', 'rms', 'zero_crossing_rate', 'peak_count',
        'power_low', 'power_medium', 'power_high',
        'wavelet_coef_1', 'wavelet_coef_2', 'wavelet_coef_3', 'wavelet_coef_4', 'wavelet_coef_5',
        'entropy', 'sample_entropy', 'lyapunov_exp'
    ]
    
    # 레이블 정의
    labels = ['normal', 'damaged', 'recovering', 'stimulated']
    
    # 데이터 프레임 생성을 위한 데이터 준비
    data = []
    
    for i in range(n_samples):
        # 랜덤 레이블 할당
        label = np.random.choice(labels)
        
        # 레이블에 따라 특성값 생성 (실제 시스템에서는 신호 처리로 계산됨)
        if label == 'normal':
            feature_values = np.random.normal(0.7, 0.1, len(features))
        elif label == 'damaged':
            feature_values = np.random.normal(0.3, 0.15, len(features))
        elif label == 'recovering':
            feature_values = np.random.normal(0.5, 0.2, len(features))
        else:  # 'stimulated'
            feature_values = np.random.normal(0.8, 0.1, len(features))
        
        # 0-1 사이로 제한
        feature_values = np.clip(feature_values, 0, 1)
        
        # 특성값 딕셔너리 생성
        sample = {feature: value for feature, value in zip(features, feature_values)}
        sample['label'] = label
        sample['sample_id'] = f'sample_{i+1:04d}'
        
        data.append(sample)
    
    # 데이터프레임 생성
    df = pd.DataFrame(data)
    
    # 파일로 저장
    csv_filename = os.path.join(PROCESSED_DIR, f"processed_features_{n_samples}.csv")
    h5_filename = os.path.join(PROCESSED_DIR, f"processed_features_{n_samples}.h5")
    
    df.to_csv(csv_filename, index=False)
    
    with h5py.File(h5_filename, 'w') as f:
        # 특성 데이터 저장
        feature_data = df[features].values
        f.create_dataset('features', data=feature_data)
        
        # 레이블 저장 (숫자로 변환)
        label_map = {label: i for i, label in enumerate(labels)}
        label_data = np.array([label_map[l] for l in df['label']])
        f.create_dataset('labels', data=label_data)
        
        # 샘플 ID 저장
        sample_ids = df['sample_id'].values
        dt = h5py.special_dtype(vlen=str)
        sample_id_dataset = f.create_dataset('sample_ids', (len(sample_ids),), dtype=dt)
        for i, sample_id in enumerate(sample_ids):
            sample_id_dataset[i] = sample_id
        
        # 메타데이터 추가
        f.attrs['feature_names'] = np.array(features, dtype=dt)
        f.attrs['label_names'] = np.array(labels, dtype=dt)
    
    print(f"처리된 샘플 저장 완료: {csv_filename}, {h5_filename}")


def main():
    """
    모든 샘플 데이터 생성
    """
    print("신경 신호 샘플 데이터 생성 시작...")
    
    # 기본 샘플 생성 (samples 디렉토리)
    print("\n기본 샘플 세트 생성...")
    generate_basic_sample_set()
    
    # 대량 샘플 생성 (neural_recordings 디렉토리)
    print("\n정상 신경 신호 샘플 생성...")
    generate_and_save_sample_set('normal', n_samples=300)
    
    print("\n손상된 신경 신호 샘플 생성...")
    generate_and_save_sample_set('damaged', n_samples=300)
    
    print("\n자극 반응 신호 샘플 생성...")
    generate_and_save_sample_set('stimulation', n_samples=300)
    
    print("\n잡음이 많은 신호 샘플 생성...")
    generate_and_save_sample_set('noisy', n_samples=100)
    
    print("\n다채널 신호 샘플 생성...")
    generate_multichannel_sample_set(n_samples=50)
    
    # 처리된 샘플 생성 (processed 디렉토리)
    print("\n처리된 샘플 데이터 생성...")
    generate_processed_samples(n_samples=200)
    
    print("\n모든 샘플 데이터 생성 완료.")


if __name__ == "__main__":
    main()
