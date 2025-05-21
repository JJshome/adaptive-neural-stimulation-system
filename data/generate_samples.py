"""
신경 신호 샘플 데이터 생성 스크립트

이 스크립트는 다양한 유형의 신경 신호 샘플을 생성하여 저장합니다.
테스트 및 데모 목적으로 사용할 수 있는 가상 신경 데이터를 제공합니다.

생성되는 샘플 데이터 유형:
1. 정상 신경 신호: 일반적인 신경 활동을 모사한 신호
2. 손상된 신경 신호: 손상된 신경의 감소된 활동을 모사한 신호
3. 자극 반응 신호: 전기자극에 대한 신경 반응을 모사한 신호
4. 잡음이 많은 신호: 다양한 노이즈 수준을 포함한 신호
5. 다채널 신경 신호: 여러 전극에서 동시에 기록된 신호를 모사
"""

import os
import numpy as np
import pandas as pd
import h5py
from scipy import signal
import matplotlib.pyplot as plt

# 데이터 저장 디렉토리
SAVE_DIR = 'data/samples'
os.makedirs(SAVE_DIR, exist_ok=True)

# 샘플링 레이트 설정
SAMPLING_RATE = 1000.0  # Hz
DURATION = 10.0  # 초
TIME = np.arange(0, DURATION, 1/SAMPLING_RATE)
NUM_SAMPLES = len(TIME)

def generate_normal_signal():
    """정상 신경 신호 생성"""
    # 기본 리듬 활동 (5-10Hz)
    base_rhythm = 0.5 * np.sin(2 * np.pi * 8 * TIME)
    
    # 무작위 스파이크 생성
    spikes = np.zeros(NUM_SAMPLES)
    num_spikes = 20
    spike_indices = np.random.choice(NUM_SAMPLES, num_spikes, replace=False)
    for idx in spike_indices:
        # 스파이크 모양 (간단한 가우시안 펄스)
        width = int(0.002 * SAMPLING_RATE)  # 2ms 너비
        t = np.arange(-width, width)
        spike = 2.0 * np.exp(-(t**2) / (2 * (width/5)**2))
        
        # 스파이크 추가 (경계 확인)
        start = max(0, idx - width)
        end = min(NUM_SAMPLES, idx + width)
        s_start = max(0, width - idx)
        s_end = min(2 * width, width + (NUM_SAMPLES - idx))
        spikes[start:end] += spike[s_start:s_end]
    
    # 리듬과 스파이크 결합
    signal = base_rhythm + spikes
    
    # 배경 노이즈 추가
    noise = 0.1 * np.random.randn(NUM_SAMPLES)
    signal += noise
    
    return signal

def generate_damaged_signal():
    """손상된 신경 신호 생성"""
    # 감소된 리듬 활동 (불규칙)
    base_rhythm = 0.3 * np.sin(2 * np.pi * 6 * TIME) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * TIME))
    
    # 더 적은 스파이크
    spikes = np.zeros(NUM_SAMPLES)
    num_spikes = 8  # 정상보다 적은 수
    spike_indices = np.random.choice(NUM_SAMPLES, num_spikes, replace=False)
    for idx in spike_indices:
        width = int(0.002 * SAMPLING_RATE)
        t = np.arange(-width, width)
        spike = 1.5 * np.exp(-(t**2) / (2 * (width/5)**2))  # 정상보다 진폭 감소
        
        start = max(0, idx - width)
        end = min(NUM_SAMPLES, idx + width)
        s_start = max(0, width - idx)
        s_end = min(2 * width, width + (NUM_SAMPLES - idx))
        spikes[start:end] += spike[s_start:s_end]
    
    # 특정 구간에서 활동 중단 모사
    dead_zone_start = int(0.3 * NUM_SAMPLES)
    dead_zone_end = int(0.5 * NUM_SAMPLES)
    signal = base_rhythm + spikes
    signal[dead_zone_start:dead_zone_end] *= 0.2
    
    # 더 많은 노이즈 추가
    noise = 0.2 * np.random.randn(NUM_SAMPLES)
    signal += noise
    
    return signal

def generate_stimulation_response():
    """자극 반응 신호 생성"""
    # 기본 신호
    base_signal = 0.3 * np.sin(2 * np.pi * 7 * TIME)
    
    # 자극 시점
    stim_times = [1.0, 3.0, 5.0, 7.0, 9.0]  # 초
    stim_indices = [int(t * SAMPLING_RATE) for t in stim_times]
    
    # 자극 아티팩트 및 반응 생성
    signal = base_signal.copy()
    for idx in stim_indices:
        # 자극 아티팩트 (큰 스파이크)
        artifact_width = int(0.01 * SAMPLING_RATE)  # 10ms
        signal[idx:idx+artifact_width] += 3.0 * np.exp(-np.arange(artifact_width) / (artifact_width/5))
        
        # 자극 후 반응 (증가된 활동)
        response_start = idx + artifact_width
        response_duration = int(0.3 * SAMPLING_RATE)  # 300ms
        response_end = min(NUM_SAMPLES, response_start + response_duration)
        
        # 증가된 스파이크 활동
        num_response_spikes = 10
        response_indices = response_start + np.random.randint(0, response_duration, num_response_spikes)
        for r_idx in response_indices:
            if r_idx < NUM_SAMPLES:
                width = int(0.002 * SAMPLING_RATE)
                t = np.arange(-width, width)
                spike = 1.8 * np.exp(-(t**2) / (2 * (width/5)**2))
                
                r_start = max(0, r_idx - width)
                r_end = min(NUM_SAMPLES, r_idx + width)
                s_start = max(0, width - (r_idx - r_start))
                s_end = min(2 * width, width + (r_end - r_idx))
                signal[r_start:r_end] += spike[s_start:s_end]
    
    # 노이즈 추가
    noise = 0.15 * np.random.randn(NUM_SAMPLES)
    signal += noise
    
    return signal

def generate_noisy_signal():
    """다양한 노이즈 수준을 가진 신호 생성"""
    # 기본 신경 신호
    base_signal = generate_normal_signal()
    
    # 전원선 노이즈 (60Hz)
    power_noise = 0.5 * np.sin(2 * np.pi * 60 * TIME)
    
    # 고주파 노이즈
    high_freq_noise = 0.3 * np.sin(2 * np.pi * 200 * TIME)
    
    # 저주파 드리프트
    drift = 0.5 * np.sin(2 * np.pi * 0.1 * TIME)
    
    # 구간별 다른 노이즈 수준
    signal = base_signal.copy()
    
    # 구간 1: 기본 노이즈
    signal[:int(0.2 * NUM_SAMPLES)] += 0.2 * np.random.randn(int(0.2 * NUM_SAMPLES))
    
    # 구간 2: 전원선 노이즈 추가
    signal[int(0.2 * NUM_SAMPLES):int(0.4 * NUM_SAMPLES)] += power_noise[int(0.2 * NUM_SAMPLES):int(0.4 * NUM_SAMPLES)]
    
    # 구간 3: 전원선 + 고주파 노이즈
    signal[int(0.4 * NUM_SAMPLES):int(0.6 * NUM_SAMPLES)] += power_noise[int(0.4 * NUM_SAMPLES):int(0.6 * NUM_SAMPLES)]
    signal[int(0.4 * NUM_SAMPLES):int(0.6 * NUM_SAMPLES)] += high_freq_noise[int(0.4 * NUM_SAMPLES):int(0.6 * NUM_SAMPLES)]
    
    # 구간 4: 모든 노이즈 유형
    signal[int(0.6 * NUM_SAMPLES):int(0.8 * NUM_SAMPLES)] += power_noise[int(0.6 * NUM_SAMPLES):int(0.8 * NUM_SAMPLES)]
    signal[int(0.6 * NUM_SAMPLES):int(0.8 * NUM_SAMPLES)] += high_freq_noise[int(0.6 * NUM_SAMPLES):int(0.8 * NUM_SAMPLES)]
    signal[int(0.6 * NUM_SAMPLES):int(0.8 * NUM_SAMPLES)] += drift[int(0.6 * NUM_SAMPLES):int(0.8 * NUM_SAMPLES)]
    
    # 구간 5: 매우 높은 노이즈 수준
    signal[int(0.8 * NUM_SAMPLES):] += power_noise[int(0.8 * NUM_SAMPLES):]
    signal[int(0.8 * NUM_SAMPLES):] += high_freq_noise[int(0.8 * NUM_SAMPLES):]
    signal[int(0.8 * NUM_SAMPLES):] += drift[int(0.8 * NUM_SAMPLES):]
    signal[int(0.8 * NUM_SAMPLES):] += 0.5 * np.random.randn(NUM_SAMPLES - int(0.8 * NUM_SAMPLES))
    
    return signal

def generate_multichannel_signal(num_channels=4):
    """다채널 신경 신호 생성"""
    # 기본 리듬 (모든 채널에 공통)
    base_rhythm = 0.3 * np.sin(2 * np.pi * 8 * TIME)
    
    # 채널별 신호 생성
    channels = []
    for ch in range(num_channels):
        # 채널별 고유 리듬 추가
        channel_rhythm = base_rhythm + 0.2 * np.sin(2 * np.pi * (5 + ch * 2) * TIME)
        
        # 채널별 스파이크 생성
        spikes = np.zeros(NUM_SAMPLES)
        num_spikes = 15 + ch * 2  # 채널마다 다른 수의 스파이크
        spike_indices = np.random.choice(NUM_SAMPLES, num_spikes, replace=False)
        for idx in spike_indices:
            width = int(0.002 * SAMPLING_RATE)
            t = np.arange(-width, width)
            spike = (1.0 + 0.2 * ch) * np.exp(-(t**2) / (2 * (width/5)**2))  # 채널마다 다른 진폭
            
            start = max(0, idx - width)
            end = min(NUM_SAMPLES, idx + width)
            s_start = max(0, width - idx)
            s_end = min(2 * width, width + (NUM_SAMPLES - idx))
            spikes[start:end] += spike[s_start:s_end]
        
        # 채널별 노이즈 추가
        noise = (0.1 + 0.02 * ch) * np.random.randn(NUM_SAMPLES)  # 채널마다 다른 노이즈 수준
        
        # 최종 채널 신호
        channel_signal = channel_rhythm + spikes + noise
        channels.append(channel_signal)
    
    return np.array(channels)

def save_as_csv(signal, filename, multichannel=False):
    """신호를 CSV 형식으로 저장"""
    if multichannel:
        # 다채널 신호
        df = pd.DataFrame({
            'time': TIME
        })
        for i in range(signal.shape[0]):
            df[f'channel_{i+1}'] = signal[i, :]
    else:
        # 단일 채널 신호
        df = pd.DataFrame({
            'time': TIME,
            'signal': signal
        })
    
    df.to_csv(os.path.join(SAVE_DIR, filename), index=False)
    print(f"CSV 파일 저장됨: {filename}")

def save_as_npy(signal, filename):
    """신호를 NumPy 배열로 저장"""
    np.save(os.path.join(SAVE_DIR, filename), signal)
    print(f"NPY 파일 저장됨: {filename}")

def save_as_hdf5(signal, filename, sampling_rate=SAMPLING_RATE, multichannel=False):
    """신호를 HDF5 형식으로 저장"""
    with h5py.File(os.path.join(SAVE_DIR, filename), 'w') as f:
        if multichannel:
            # 다채널 데이터
            f.create_dataset('data', data=signal)
        else:
            # 단일 채널 데이터
            f.create_dataset('data', data=signal.reshape(1, -1))
            
        # 메타데이터 추가
        f.attrs['sampling_rate'] = sampling_rate
        f.attrs['duration'] = DURATION
        f.attrs['num_samples'] = NUM_SAMPLES
        
        if multichannel:
            f.attrs['num_channels'] = signal.shape[0]
        else:
            f.attrs['num_channels'] = 1
    
    print(f"HDF5 파일 저장됨: {filename}")

def plot_and_save_figure(signals, titles, filename):
    """신호 시각화 및 그림 저장"""
    n = len(signals)
    fig, axs = plt.subplots(n, 1, figsize=(10, 2 * n), sharex=True)
    
    if n == 1:
        axs = [axs]  # 단일 신호인 경우 리스트로 변환
    
    for i, (signal, title) in enumerate(zip(signals, titles)):
        if len(signal.shape) > 1:
            # 다채널 신호
            for ch in range(signal.shape[0]):
                axs[i].plot(TIME, signal[ch], label=f'Channel {ch+1}')
            axs[i].legend()
        else:
            # 단일 채널 신호
            axs[i].plot(TIME, signal)
            
        axs[i].set_title(title)
        axs[i].set_ylabel('Amplitude')
        axs[i].grid(True)
    
    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    
    # 그림 저장
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=300, bbox_inches='tight')
    print(f"그림 파일 저장됨: {filename}")
    plt.close()

def create_data_description():
    """샘플 데이터 설명 파일 생성"""
    description = """# 신경 신호 샘플 데이터 설명

이 디렉토리에는 신경 전기자극 시스템 테스트 및 데모를 위한 가상 신경 신호 샘플이 포함되어 있습니다.

## 데이터 형식

샘플 데이터는 다음 형식으로 제공됩니다:
- CSV 파일 (.csv): 시간 열과 신호 열을 포함하는 표 형식
- NumPy 배열 (.npy): 원시 신호 데이터를 포함하는 NumPy 배열
- HDF5 파일 (.h5): 신호 데이터와 메타데이터(샘플링 레이트, 채널 수 등)를 포함하는 계층적 데이터 형식

## 데이터 유형

1. **정상 신경 신호 (normal_signal)**
   - 정상적인 신경 활동을 모사한 신호
   - 기본 리듬 활동(5-10Hz)과 무작위 스파이크를 포함
   - 약간의 배경 노이즈 포함

2. **손상된 신경 신호 (damaged_signal)**
   - 손상된 신경의 감소된 활동을 모사한 신호
   - 감소된 리듬 활동과 더 적은 스파이크
   - 특정 구간에서의 활동 중단 포함
   - 더 높은 노이즈 수준

3. **자극 반응 신호 (stimulation_response)**
   - 전기자극에 대한 신경 반응을 모사한 신호
   - 자극 아티팩트와 후속 활동 증가를 포함
   - 주기적인 자극 패턴

4. **잡음이 많은 신호 (noisy_signal)**
   - 다양한 노이즈 유형과 수준을 포함한 신호
   - 전원선 노이즈(60Hz), 고주파 노이즈, 저주파 드리프트 등
   - 구간별로 다른 노이즈 특성

5. **다채널 신경 신호 (multichannel_signal)**
   - 여러 전극에서 동시에 기록된 신호를 모사
   - 채널 간 공통 리듬과 채널별 고유 특성 포함
   - 채널별로 다른 스파이크 패턴과 노이즈 수준

## 사용 방법

이 샘플 데이터는 다음과 같은 용도로 사용할 수 있습니다:
- 신호 처리 알고리즘 테스트
- 특성 추출 및 분석 기능 검증
- 자극 최적화 알고리즘 평가
- 시각화 및 데모 목적

## 참고 사항

이 데이터는 실제 신경 신호를 모사한 가상 데이터로, 교육 및 개발 목적으로만 사용해야 합니다.
실제 임상 응용에는 적합하지 않습니다.

샘플링 레이트: {0} Hz
신호 길이: {1} 초
샘플 수: {2}
""".format(SAMPLING_RATE, DURATION, NUM_SAMPLES)

    with open(os.path.join(SAVE_DIR, 'README.md'), 'w') as f:
        f.write(description)
    
    print("데이터 설명 파일 생성됨: README.md")

def main():
    """샘플 데이터 생성 및 저장 메인 함수"""
    print("신경 신호 샘플 데이터 생성 중...")
    
    # 정상 신경 신호
    normal_signal = generate_normal_signal()
    save_as_csv(normal_signal, 'normal_signal.csv')
    save_as_npy(normal_signal, 'normal_signal.npy')
    save_as_hdf5(normal_signal, 'normal_signal.h5')
    
    # 손상된 신경 신호
    damaged_signal = generate_damaged_signal()
    save_as_csv(damaged_signal, 'damaged_signal.csv')
    save_as_npy(damaged_signal, 'damaged_signal.npy')
    save_as_hdf5(damaged_signal, 'damaged_signal.h5')
    
    # 자극 반응 신호
    stim_response = generate_stimulation_response()
    save_as_csv(stim_response, 'stimulation_response.csv')
    save_as_npy(stim_response, 'stimulation_response.npy')
    save_as_hdf5(stim_response, 'stimulation_response.h5')
    
    # 잡음이 많은 신호
    noisy_signal = generate_noisy_signal()
    save_as_csv(noisy_signal, 'noisy_signal.csv')
    save_as_npy(noisy_signal, 'noisy_signal.npy')
    save_as_hdf5(noisy_signal, 'noisy_signal.h5')
    
    # 다채널 신경 신호
    multichannel_signal = generate_multichannel_signal(num_channels=4)
    save_as_csv(multichannel_signal, 'multichannel_signal.csv', multichannel=True)
    save_as_npy(multichannel_signal, 'multichannel_signal.npy')
    save_as_hdf5(multichannel_signal, 'multichannel_signal.h5', multichannel=True)
    
    # 그림 생성
    plot_and_save_figure(
        [normal_signal, damaged_signal, stim_response, noisy_signal],
        ['Normal Signal', 'Damaged Signal', 'Stimulation Response', 'Noisy Signal'],
        'signal_comparison.png'
    )
    
    # 다채널 신호 그림
    plot_and_save_figure(
        [multichannel_signal],
        ['Multichannel Signal'],
        'multichannel_signal.png'
    )
    
    # 데이터 설명 파일 생성
    create_data_description()
    
    print("샘플 데이터 생성 완료!")

if __name__ == "__main__":
    main()
