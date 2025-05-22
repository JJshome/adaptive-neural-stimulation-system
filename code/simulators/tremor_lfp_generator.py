"""
떨림 관련 LFP (Local Field Potential) 데이터 시뮬레이터

이 모듈은 파킨슨병 및 본태성 떨림 환자의 신경 활동을 모방하는 
합성 LFP 데이터를 생성합니다. 실제 떨림 환자의 신경생리학적 특성을
반영하여 더 현실적인 시뮬레이션을 제공합니다.

주요 특징:
- 병리학적 베타 밴드 (13-30Hz) 진동 증가
- 떨림 주파수 대역 (4-12Hz) 활성화
- STN, GPi, 시상 등 다양한 뇌 영역 시뮬레이션
- 자극 전/후 상태 변화 모델링
"""

import numpy as np
import scipy.signal as signal
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class BrainRegion(Enum):
    """떨림 관련 주요 뇌 영역"""
    STN = "Subthalamic Nucleus"  # 시상하핵
    GPI = "Globus Pallidus Internus"  # 내측창백핵
    THALAMUS = "Thalamus"  # 시상
    MOTOR_CORTEX = "Motor Cortex"  # 운동피질
    CEREBELLUM = "Cerebellum"  # 소뇌


class TremorType(Enum):
    """떨림 유형"""
    PARKINSONIAN = "Parkinsonian Tremor"  # 파킨슨병 떨림
    ESSENTIAL = "Essential Tremor"  # 본태성 떨림
    DYSTONIC = "Dystonic Tremor"  # 근긴장이상 떨림


@dataclass
class TremorCharacteristics:
    """떨림 특성 파라미터"""
    tremor_frequency: float  # 주 떨림 주파수 (Hz)
    tremor_amplitude: float  # 떨림 진폭
    beta_power: float  # 베타 밴드 파워
    beta_frequency: float  # 주 베타 주파수
    gamma_coupling: float  # 베타-감마 커플링 강도
    
    
class PathologicalLFPGenerator:
    """병리학적 LFP 신호 생성기"""
    
    def __init__(self, sampling_rate: int = 1000):
        """
        초기화
        
        Args:
            sampling_rate: 샘플링 주파수 (Hz)
        """
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
        
        # 떨림 유형별 기본 특성
        self.tremor_profiles = {
            TremorType.PARKINSONIAN: TremorCharacteristics(
                tremor_frequency=5.0,  # 4-6 Hz
                tremor_amplitude=2.0,
                beta_power=3.0,  # 병리학적으로 증가
                beta_frequency=20.0,  # 15-25 Hz
                gamma_coupling=0.7
            ),
            TremorType.ESSENTIAL: TremorCharacteristics(
                tremor_frequency=7.0,  # 6-12 Hz
                tremor_amplitude=2.5,
                beta_power=1.5,  # 상대적으로 낮음
                beta_frequency=18.0,
                gamma_coupling=0.3
            ),
            TremorType.DYSTONIC: TremorCharacteristics(
                tremor_frequency=4.5,  # 4-7 Hz
                tremor_amplitude=1.8,
                beta_power=2.2,
                beta_frequency=22.0,
                gamma_coupling=0.5
            )
        }
        
    def generate_base_neural_noise(self, duration: float) -> np.ndarray:
        """
        기본 신경 노이즈 생성 (1/f 특성)
        
        Args:
            duration: 신호 지속 시간 (초)
            
        Returns:
            1/f 노이즈 신호
        """
        n_samples = int(duration * self.sampling_rate)
        
        # 백색 노이즈 생성
        white_noise = np.random.randn(n_samples)
        
        # FFT를 통한 주파수 도메인 변환
        fft_noise = np.fft.rfft(white_noise)
        frequencies = np.fft.rfftfreq(n_samples, 1/self.sampling_rate)
        
        # 1/f 스펙트럼 적용 (DC 성분 제외)
        frequencies[0] = 1  # DC 성분 처리
        pink_filter = 1 / np.sqrt(frequencies)
        
        # 필터 적용 및 역변환
        fft_noise *= pink_filter
        pink_noise = np.fft.irfft(fft_noise, n_samples)
        
        return pink_noise
    
    def generate_tremor_component(self, 
                                 duration: float,
                                 frequency: float,
                                 amplitude: float,
                                 phase_noise: float = 0.1) -> np.ndarray:
        """
        떨림 주파수 성분 생성
        
        Args:
            duration: 신호 지속 시간 (초)
            frequency: 떨림 주파수 (Hz)
            amplitude: 떨림 진폭
            phase_noise: 위상 노이즈 수준
            
        Returns:
            떨림 성분 신호
        """
        n_samples = int(duration * self.sampling_rate)
        t = np.arange(n_samples) / self.sampling_rate
        
        # 위상 노이즈 추가 (떨림의 불규칙성 모사)
        phase_modulation = np.cumsum(np.random.randn(n_samples) * phase_noise)
        
        # 떨림 신호 생성
        tremor = amplitude * np.sin(2 * np.pi * frequency * t + phase_modulation)
        
        # 진폭 변조 (떨림 강도의 변동성)
        amplitude_modulation = 1 + 0.2 * np.sin(2 * np.pi * 0.5 * t)
        tremor *= amplitude_modulation
        
        return tremor
    
    def generate_beta_oscillation(self,
                                 duration: float,
                                 center_freq: float,
                                 bandwidth: float = 4.0,
                                 power: float = 1.0) -> np.ndarray:
        """
        병리학적 베타 진동 생성
        
        Args:
            duration: 신호 지속 시간 (초)
            center_freq: 중심 주파수 (Hz)
            bandwidth: 대역폭 (Hz)
            power: 진동 강도
            
        Returns:
            베타 진동 신호
        """
        n_samples = int(duration * self.sampling_rate)
        t = np.arange(n_samples) / self.sampling_rate
        
        # 베타 버스트 생성 (간헐적 발생)
        burst_duration = 0.2  # 200ms 버스트
        burst_interval = 0.5  # 평균 500ms 간격
        
        beta = np.zeros(n_samples)
        current_time = 0
        
        while current_time < duration:
            # 버스트 시작 위치
            start_idx = int(current_time * self.sampling_rate)
            end_idx = int((current_time + burst_duration) * self.sampling_rate)
            
            if end_idx > n_samples:
                end_idx = n_samples
                
            # 버스트 생성
            burst_t = t[start_idx:end_idx] - current_time
            envelope = np.exp(-10 * (burst_t - burst_duration/2)**2)
            
            # 주파수 변조 추가
            freq_mod = center_freq + np.random.randn() * bandwidth/4
            burst = power * envelope * np.sin(2 * np.pi * freq_mod * burst_t)
            
            beta[start_idx:end_idx] += burst
            
            # 다음 버스트까지 간격 (포아송 분포)
            current_time += burst_duration + np.random.exponential(burst_interval)
            
        return beta
    
    def apply_cross_frequency_coupling(self,
                                     low_freq: np.ndarray,
                                     high_freq: np.ndarray,
                                     coupling_strength: float) -> np.ndarray:
        """
        교차 주파수 커플링 적용 (Phase-Amplitude Coupling)
        
        Args:
            low_freq: 저주파 신호 (예: 베타)
            high_freq: 고주파 신호 (예: 감마)
            coupling_strength: 커플링 강도 (0-1)
            
        Returns:
            커플링된 신호
        """
        # 저주파 신호의 위상 추출
        analytic_signal = signal.hilbert(low_freq)
        phase = np.angle(analytic_signal)
        
        # 위상에 따른 진폭 변조
        modulation = 1 + coupling_strength * np.sin(phase)
        coupled_signal = high_freq * modulation
        
        return coupled_signal
    
    def generate_pathological_lfp(self,
                                 brain_region: BrainRegion,
                                 tremor_type: TremorType,
                                 duration: float,
                                 severity: float = 1.0,
                                 dbs_on: bool = False) -> Dict[str, np.ndarray]:
        """
        병리학적 LFP 신호 생성
        
        Args:
            brain_region: 뇌 영역
            tremor_type: 떨림 유형
            duration: 신호 지속 시간 (초)
            severity: 증상 심각도 (0-2, 1이 기본)
            dbs_on: DBS 자극 상태
            
        Returns:
            생성된 신호 및 구성 요소들
        """
        characteristics = self.tremor_profiles[tremor_type]
        
        # 기본 신경 노이즈
        base_noise = self.generate_base_neural_noise(duration) * 0.5
        
        # 떨림 성분
        tremor_component = self.generate_tremor_component(
            duration,
            characteristics.tremor_frequency,
            characteristics.tremor_amplitude * severity
        )
        
        # 병리학적 베타 진동
        beta_component = self.generate_beta_oscillation(
            duration,
            characteristics.beta_frequency,
            bandwidth=5.0,
            power=characteristics.beta_power * severity
        )
        
        # 감마 대역 (30-100 Hz) 활동
        gamma_noise = self._generate_filtered_noise(duration, 30, 100) * 0.3
        
        # 베타-감마 커플링
        coupled_gamma = self.apply_cross_frequency_coupling(
            beta_component,
            gamma_noise,
            characteristics.gamma_coupling * severity
        )
        
        # DBS 효과 모델링
        if dbs_on:
            # DBS는 병리학적 진동을 억제
            tremor_component *= 0.3  # 70% 감소
            beta_component *= 0.2   # 80% 감소
            
            # DBS 자극 아티팩트 추가 (선택적)
            stim_artifact = self._generate_dbs_artifact(duration)
            base_noise += stim_artifact * 0.1
        
        # 영역별 가중치 적용
        region_weights = self._get_region_weights(brain_region, tremor_type)
        
        # 최종 LFP 신호 조합
        lfp_signal = (
            base_noise +
            tremor_component * region_weights['tremor'] +
            beta_component * region_weights['beta'] +
            coupled_gamma * region_weights['gamma']
        )
        
        # 신호 정규화
        lfp_signal = self._normalize_signal(lfp_signal)
        
        return {
            'lfp': lfp_signal,
            'tremor': tremor_component,
            'beta': beta_component,
            'gamma': coupled_gamma,
            'noise': base_noise,
            'sampling_rate': self.sampling_rate,
            'metadata': {
                'brain_region': brain_region.value,
                'tremor_type': tremor_type.value,
                'severity': severity,
                'dbs_on': dbs_on,
                'duration': duration
            }
        }
    
    def _generate_filtered_noise(self, 
                               duration: float,
                               low_freq: float,
                               high_freq: float) -> np.ndarray:
        """대역 필터링된 노이즈 생성"""
        n_samples = int(duration * self.sampling_rate)
        noise = np.random.randn(n_samples)
        
        # 버터워스 필터 설계
        sos = signal.butter(4, [low_freq, high_freq], 
                          btype='band', 
                          fs=self.sampling_rate, 
                          output='sos')
        filtered = signal.sosfilt(sos, noise)
        
        return filtered
    
    def _generate_dbs_artifact(self, duration: float, 
                             stim_freq: float = 130.0) -> np.ndarray:
        """DBS 자극 아티팩트 생성"""
        n_samples = int(duration * self.sampling_rate)
        t = np.arange(n_samples) / self.sampling_rate
        
        # 펄스 트레인 생성
        pulse_width = 0.06e-3  # 60 μs
        pulse_indices = (t % (1/stim_freq)) < pulse_width
        
        artifact = np.zeros(n_samples)
        artifact[pulse_indices] = 1.0
        
        return artifact
    
    def _get_region_weights(self, 
                          brain_region: BrainRegion,
                          tremor_type: TremorType) -> Dict[str, float]:
        """뇌 영역별 신호 성분 가중치"""
        weights = {
            BrainRegion.STN: {
                TremorType.PARKINSONIAN: {'tremor': 0.8, 'beta': 1.0, 'gamma': 0.6},
                TremorType.ESSENTIAL: {'tremor': 0.6, 'beta': 0.4, 'gamma': 0.3},
                TremorType.DYSTONIC: {'tremor': 0.7, 'beta': 0.8, 'gamma': 0.5}
            },
            BrainRegion.GPI: {
                TremorType.PARKINSONIAN: {'tremor': 0.7, 'beta': 0.9, 'gamma': 0.5},
                TremorType.ESSENTIAL: {'tremor': 0.5, 'beta': 0.3, 'gamma': 0.2},
                TremorType.DYSTONIC: {'tremor': 0.6, 'beta': 0.7, 'gamma': 0.4}
            },
            BrainRegion.THALAMUS: {
                TremorType.PARKINSONIAN: {'tremor': 1.0, 'beta': 0.6, 'gamma': 0.4},
                TremorType.ESSENTIAL: {'tremor': 1.0, 'beta': 0.3, 'gamma': 0.3},
                TremorType.DYSTONIC: {'tremor': 0.9, 'beta': 0.5, 'gamma': 0.4}
            },
            BrainRegion.MOTOR_CORTEX: {
                TremorType.PARKINSONIAN: {'tremor': 0.6, 'beta': 0.7, 'gamma': 0.8},
                TremorType.ESSENTIAL: {'tremor': 0.7, 'beta': 0.4, 'gamma': 0.6},
                TremorType.DYSTONIC: {'tremor': 0.5, 'beta': 0.6, 'gamma': 0.7}
            },
            BrainRegion.CEREBELLUM: {
                TremorType.PARKINSONIAN: {'tremor': 0.4, 'beta': 0.3, 'gamma': 0.5},
                TremorType.ESSENTIAL: {'tremor': 0.9, 'beta': 0.2, 'gamma': 0.4},
                TremorType.DYSTONIC: {'tremor': 0.6, 'beta': 0.4, 'gamma': 0.5}
            }
        }
        
        return weights[brain_region][tremor_type]
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """신호 정규화 (실제 LFP 범위로)"""
        # 실제 LFP는 보통 수십-수백 μV 범위
        # 표준화 후 스케일링
        normalized = (signal - np.mean(signal)) / np.std(signal)
        scaled = normalized * 50  # ~50 μV RMS
        
        return scaled
    

class TremorLFPDataset:
    """떨림 LFP 데이터셋 생성 및 관리"""
    
    def __init__(self, generator: PathologicalLFPGenerator):
        """
        초기화
        
        Args:
            generator: LFP 생성기 인스턴스
        """
        self.generator = generator
        self.data_cache = []
        
    def generate_dataset(self,
                        n_samples: int,
                        duration: float = 10.0,
                        include_dbs: bool = True) -> pd.DataFrame:
        """
        떨림 LFP 데이터셋 생성
        
        Args:
            n_samples: 생성할 샘플 수
            duration: 각 샘플의 지속 시간 (초)
            include_dbs: DBS on/off 상태 포함 여부
            
        Returns:
            생성된 데이터셋 DataFrame
        """
        dataset = []
        
        for i in range(n_samples):
            # 무작위로 조건 선택
            brain_region = np.random.choice(list(BrainRegion))
            tremor_type = np.random.choice(list(TremorType))
            severity = np.random.uniform(0.5, 2.0)
            dbs_on = include_dbs and np.random.choice([True, False])
            
            # LFP 생성
            result = self.generator.generate_pathological_lfp(
                brain_region,
                tremor_type,
                duration,
                severity,
                dbs_on
            )
            
            # 특징 추출
            features = self._extract_features(result['lfp'])
            
            # 데이터셋에 추가
            sample = {
                'sample_id': i,
                'brain_region': brain_region.value,
                'tremor_type': tremor_type.value,
                'severity': severity,
                'dbs_on': dbs_on,
                **features,
                'lfp_signal': result['lfp'],
                'tremor_component': result['tremor'],
                'beta_component': result['beta'],
                'gamma_component': result['gamma']
            }
            
            dataset.append(sample)
            
        return pd.DataFrame(dataset)
    
    def _extract_features(self, lfp_signal: np.ndarray) -> Dict[str, float]:
        """LFP 신호에서 특징 추출"""
        # 파워 스펙트럼 밀도 계산
        freqs, psd = signal.periodogram(lfp_signal, 
                                       fs=self.generator.sampling_rate)
        
        # 주파수 대역별 파워
        features = {
            'delta_power': self._band_power(freqs, psd, 0.5, 4),
            'theta_power': self._band_power(freqs, psd, 4, 8),
            'alpha_power': self._band_power(freqs, psd, 8, 13),
            'beta_power': self._band_power(freqs, psd, 13, 30),
            'low_gamma_power': self._band_power(freqs, psd, 30, 60),
            'high_gamma_power': self._band_power(freqs, psd, 60, 100),
            'tremor_band_power': self._band_power(freqs, psd, 4, 12),
        }
        
        # 스펙트럼 엔트로피
        features['spectral_entropy'] = self._spectral_entropy(freqs, psd)
        
        # 피크 주파수
        features['peak_frequency'] = freqs[np.argmax(psd)]
        
        # 시간 도메인 특징
        features['rms'] = np.sqrt(np.mean(lfp_signal**2))
        features['variance'] = np.var(lfp_signal)
        
        return features
    
    def _band_power(self, freqs: np.ndarray, psd: np.ndarray,
                   low: float, high: float) -> float:
        """특정 주파수 대역의 파워 계산"""
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        return np.trapz(psd[idx], freqs[idx])
    
    def _spectral_entropy(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """스펙트럼 엔트로피 계산"""
        # 정규화된 PSD
        psd_norm = psd / np.sum(psd)
        
        # 엔트로피 계산
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-15))
        
        return entropy


# 사용 예시
if __name__ == "__main__":
    # 생성기 초기화
    generator = PathologicalLFPGenerator(sampling_rate=1000)
    
    # 파킨슨병 환자의 STN LFP 생성
    pd_stn_lfp = generator.generate_pathological_lfp(
        brain_region=BrainRegion.STN,
        tremor_type=TremorType.PARKINSONIAN,
        duration=10.0,
        severity=1.5,
        dbs_on=False
    )
    
    print(f"Generated {pd_stn_lfp['metadata']}")
    print(f"LFP shape: {pd_stn_lfp['lfp'].shape}")
    print(f"Mean amplitude: {np.mean(np.abs(pd_stn_lfp['lfp'])):.2f} μV")
    
    # 데이터셋 생성
    dataset_generator = TremorLFPDataset(generator)
    dataset = dataset_generator.generate_dataset(n_samples=100, duration=5.0)
    
    print(f"\nGenerated dataset with {len(dataset)} samples")
    print(f"Features: {list(dataset.columns)}")
