"""
신경 전기자극 제어 모듈

이 모듈은 신경 전기자극을 제어하는 알고리즘을 구현합니다.
자극 파라미터 조정, 자극 패턴 생성, 피드백 기반 자극 제어 기능을 제공합니다.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

class StimulationController:
    """신경 전기자극 제어를 위한 클래스"""
    
    def __init__(self, sampling_rate: float = 1000.0):
        """
        StimulationController 초기화
        
        매개변수:
            sampling_rate (float): 샘플링 레이트 (Hz)
        """
        self.sampling_rate = sampling_rate
        self.current_params = {
            'amplitude': 1.0,  # mA
            'frequency': 130.0,  # Hz
            'pulse_width': 60.0,  # μs
            'waveform': 'biphasic',
            'active': False
        }
    
    def generate_stimulation_waveform(self, duration: float, **params) -> np.ndarray:
        """
        지정된 매개변수로 자극 파형 생성
        
        매개변수:
            duration (float): 자극 지속 시간 (초)
            **params: 자극 매개변수 (amplitude, frequency, pulse_width, waveform)
            
        반환값:
            np.ndarray: 생성된 자극 파형
        """
        # 기본 매개변수 설정
        amplitude = params.get('amplitude', self.current_params['amplitude'])
        frequency = params.get('frequency', self.current_params['frequency'])
        pulse_width = params.get('pulse_width', self.current_params['pulse_width'])
        waveform_type = params.get('waveform', self.current_params['waveform'])
        
        # 샘플 수 계산
        num_samples = int(duration * self.sampling_rate)
        stimulation = np.zeros(num_samples)
        
        # 펄스 간격 (초) 계산
        pulse_interval = 1.0 / frequency
        
        # 펄스 폭 (샘플 수)
        pulse_width_samples = int((pulse_width / 1e6) * self.sampling_rate)
        
        # 펄스 간격 (샘플 수)
        pulse_interval_samples = int(pulse_interval * self.sampling_rate)
        
        # 자극 파형 생성
        if waveform_type == 'monophasic':
            # 단상 펄스
            for i in range(0, num_samples, pulse_interval_samples):
                if i + pulse_width_samples < num_samples:
                    stimulation[i:i + pulse_width_samples] = amplitude
                    
        elif waveform_type == 'biphasic':
            # 이상 펄스 (양/음 펄스)
            for i in range(0, num_samples, pulse_interval_samples):
                if i + pulse_width_samples * 2 < num_samples:
                    stimulation[i:i + pulse_width_samples] = amplitude
                    stimulation[i + pulse_width_samples:i + 2*pulse_width_samples] = -amplitude
                    
        elif waveform_type == 'triphasic':
            # 삼상 펄스
            for i in range(0, num_samples, pulse_interval_samples):
                if i + pulse_width_samples * 3 < num_samples:
                    stimulation[i:i + pulse_width_samples] = amplitude
                    stimulation[i + pulse_width_samples:i + 2*pulse_width_samples] = -amplitude
                    stimulation[i + 2*pulse_width_samples:i + 3*pulse_width_samples] = amplitude / 2
                    
        elif waveform_type == 'burst':
            # 버스트 자극 (고주파 펄스 그룹)
            burst_frequency = params.get('burst_frequency', 10.0)  # Hz
            pulses_per_burst = params.get('pulses_per_burst', 5)
            
            burst_interval = 1.0 / burst_frequency
            burst_interval_samples = int(burst_interval * self.sampling_rate)
            
            for burst_start in range(0, num_samples, burst_interval_samples):
                for pulse_idx in range(pulses_per_burst):
                    pulse_start = burst_start + pulse_idx * pulse_interval_samples
                    if pulse_start + pulse_width_samples * 2 < num_samples:
                        stimulation[pulse_start:pulse_start + pulse_width_samples] = amplitude
                        stimulation[pulse_start + pulse_width_samples:pulse_start + 2*pulse_width_samples] = -amplitude
        
        return stimulation
    
    def update_parameters(self, **params) -> Dict[str, Any]:
        """
        자극 매개변수 업데이트
        
        매개변수:
            **params: 업데이트할 자극 매개변수
            
        반환값:
            Dict[str, Any]: 업데이트된 매개변수
        """
        for key, value in params.items():
            if key in self.current_params:
                self.current_params[key] = value
                
        return self.current_params
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        현재 자극 매개변수 반환
        
        반환값:
            Dict[str, Any]: 현재 자극 매개변수
        """
        return self.current_params.copy()

    def activate_stimulation(self) -> bool:
        """
        자극 활성화
        
        반환값:
            bool: 활성화 성공 여부
        """
        self.current_params['active'] = True
        return True
    
    def deactivate_stimulation(self) -> bool:
        """
        자극 비활성화
        
        반환값:
            bool: 비활성화 성공 여부
        """
        self.current_params['active'] = False
        return True
    
    def is_active(self) -> bool:
        """
        자극 활성화 상태 확인
        
        반환값:
            bool: 활성화 상태
        """
        return self.current_params['active']
