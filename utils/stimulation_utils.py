"""
전기자극 시뮬레이션 유틸리티 모듈

이 모듈은 신경재생을 위한 전기자극을 시뮬레이션하고 분석하는 함수들을 제공합니다.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import signal


def simulate_stimulation_response(params, nerve_state='damaged', duration_days=30, random_seed=None):
    """
    전기자극에 대한 신경 반응 시뮬레이션 함수
    
    Parameters:
    -----------
    params : dict
        전기자극 파라미터 (주파수, 진폭, 펄스폭, 듀티 사이클, 자극 기간)
    nerve_state : str
        신경 상태 ('normal', 'damaged', 'recovery')
    duration_days : int
        시뮬레이션 기간 (일)
    random_seed : int
        랜덤 시드
        
    Returns:
    --------
    response : dict
        시뮬레이션 결과 (일별 성장률, 신경인자 발현, 기능적 회복 등)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 각 파라미터의 유효성 검사 및 기본값 설정
    default_params = {
        'frequency': 50,   # Hz
        'amplitude': 2.0,  # mA
        'pulse_width': 300,  # µs
        'duty_cycle': 50,  # %
        'duration': 30     # 분
    }
    
    for key, default_value in default_params.items():
        if key not in params:
            params[key] = default_value
    
    # 신경 상태별 응답 모델 파라미터
    nerve_params = {
        'normal': {
            'base_growth_rate': 5.0,  # µm/일
            'growth_factor_base': 3.0,
            'recovery_rate': 0.95,
            'injury_threshold': 7.0,  # 고강도 자극으로 인한 손상 임계값
            'optimal_frequency': 50,
            'optimal_amplitude': 1.0,
            'optimal_pulse_width': 200,
            'response_delay': 1  # 일
        },
        'damaged': {
            'base_growth_rate': 1.0,
            'growth_factor_base': 1.0,
            'recovery_rate': 0.3,
            'injury_threshold': 10.0,
            'optimal_frequency': 100,
            'optimal_amplitude': 3.0,
            'optimal_pulse_width': 500,
            'response_delay': 3
        },
        'recovery': {
            'base_growth_rate': 3.0,
            'growth_factor_base': 2.0,
            'recovery_rate': 0.6,
            'injury_threshold': 8.0,
            'optimal_frequency': 75,
            'optimal_amplitude': 2.0,
            'optimal_pulse_width': 300,
            'response_delay': 2
        }
    }
    
    # 선택한 상태의 파라미터 가져오기
    state_params = nerve_params[nerve_state]
    
    # 전기자극 효과 계산
    # 주파수 효과: 최적 주파수에서 최대
    freq_effect = 1.0 - 0.01 * abs(params['frequency'] - state_params['optimal_frequency']) / state_params['optimal_frequency']
    freq_effect = max(0.1, min(1.0, freq_effect))
    
    # 진폭 효과: 너무 낮으면 효과 적음, 너무 높으면 손상 위험
    amp_ratio = params['amplitude'] / state_params['optimal_amplitude']
    if amp_ratio < 1.0:
        amp_effect = amp_ratio  # 최적보다 낮으면 비례해서 효과 감소
    else:
        # 최적보다 높으면 효과 감소 시작, 손상 임계값 이상이면 부정적 효과
        injury_ratio = params['amplitude'] / state_params['injury_threshold']
        amp_effect = 1.0 - 0.5 * (amp_ratio - 1.0) * (1.0 + 2.0 * min(1.0, injury_ratio))
    amp_effect = max(0.0, min(1.0, amp_effect))
    
    # 펄스폭 효과: 최적 펄스폭에서 최대
    pw_effect = 1.0 - 0.005 * abs(params['pulse_width'] - state_params['optimal_pulse_width']) / state_params['optimal_pulse_width']
    pw_effect = max(0.2, min(1.0, pw_effect))
    
    # 듀티 사이클 효과: 신경 상태별로 다름
    if nerve_state == 'normal':
        duty_effect = 1.0 - params['duty_cycle'] / 100.0 * 0.3  # 정상에서는 낮은 듀티 사이클 선호
    elif nerve_state == 'damaged':
        duty_effect = params['duty_cycle'] / 100.0  # 손상에서는 높은 듀티 사이클 선호
    else:  # recovery
        duty_effect = 1.0 - abs(params['duty_cycle'] - 50) / 50.0 * 0.5  # 회복에서는 중간 듀티 사이클 선호
    duty_effect = max(0.3, min(1.0, duty_effect))
    
    # 자극 기간 효과: 너무 짧으면 효과 적음, 너무 길면 부정적 영향
    duration_effect = min(1.0, params['duration'] / 30.0)  # 30분을 기준으로 정규화
    if params['duration'] > 60:
        # 60분 이상은 효과 감소 시작
        duration_effect = 1.0 - 0.1 * (params['duration'] - 60) / 30.0
    duration_effect = max(0.2, min(1.0, duration_effect))
    
    # 총 자극 효과 계산
    total_effect = (freq_effect * 0.3 + 
                    amp_effect * 0.3 + 
                    pw_effect * 0.2 + 
                    duty_effect * 0.1 + 
                    duration_effect * 0.1)
    
    # 일별 응답 시뮬레이션
    days = np.arange(1, duration_days + 1)
    
    # 응답 지연 적용
    delayed_effect = np.zeros(duration_days)
    response_start = state_params['response_delay']
    for i in range(response_start, duration_days):
        # 처음에는 효과가 점차 증가하다가 안정화
        ramp_factor = min(1.0, (i - response_start + 1) / 7)  # 7일에 걸쳐 최대 효과에 도달
        delayed_effect[i] = total_effect * ramp_factor
    
    # 기본 성장률에 자극 효과 추가
    growth_rates = np.zeros(duration_days)
    for i in range(duration_days):
        base = state_params['base_growth_rate']
        stim_effect = delayed_effect[i] * 10.0  # 자극으로 인한 추가 성장
        daily_variation = np.random.normal(0, 0.5)  # 일별 변동
        
        growth_rates[i] = base + stim_effect + daily_variation
        growth_rates[i] = max(0, growth_rates[i])  # 음수 방지
    
    # 신경성장인자 발현 (BDNF, GDNF 등)
    growth_factors = np.zeros(duration_days)
    for i in range(duration_days):
        base = state_params['growth_factor_base']
        stim_effect = delayed_effect[i] * 5.0
        daily_variation = np.random.normal(0, 0.3)
        
        # BDNF는 자극 후 일시적으로 급증했다가 서서히 감소
        if i > 0 and i < duration_days - 1:
            growth_factors[i] = base + stim_effect + daily_variation
            if delayed_effect[i] > delayed_effect[i-1]:
                # 자극 효과가 증가하면 성장인자 급증
                growth_factors[i] *= 1.5
        else:
            growth_factors[i] = base + stim_effect + daily_variation
        
        growth_factors[i] = max(0, min(10, growth_factors[i]))  # 0-10 범위 유지
    
    # 기능적 회복 점수 (누적 효과 있음)
    recovery_scores = np.zeros(duration_days)
    current_recovery = 20 if nerve_state == 'normal' else (10 if nerve_state == 'recovery' else 0)
    
    for i in range(duration_days):
        # 회복 속도는 현재 상태와 자극 효과에 비례
        recovery_increment = delayed_effect[i] * 2.0 * (1.0 - current_recovery / 100.0)
        current_recovery += recovery_increment
        daily_variation = np.random.normal(0, 0.2)
        
        recovery_scores[i] = current_recovery + daily_variation
        recovery_scores[i] = max(0, min(100, recovery_scores[i]))  # 0-100 범위 유지
    
    # 신경 전도 속도 (회복의 지표)
    conduction_velocities = np.zeros(duration_days)
    base_velocity = 30 if nerve_state == 'normal' else (15 if nerve_state == 'recovery' else 5)
    max_velocity = 60
    
    for i in range(duration_days):
        improvement = delayed_effect[i] * 0.5 * (max_velocity - base_velocity) / 100.0 * (1.0 - (base_velocity / max_velocity))
        base_velocity += improvement
        daily_variation = np.random.normal(0, 0.3)
        
        conduction_velocities[i] = base_velocity + daily_variation
        conduction_velocities[i] = max(0, min(max_velocity, conduction_velocities[i]))
    
    # 부작용 심각도
    side_effects = np.zeros(duration_days)
    for i in range(duration_days):
        # 진폭과 주파수가 높을수록 부작용 위험 증가
        intensity_factor = (params['amplitude'] / state_params['injury_threshold']) * (params['frequency'] / 200.0)
        
        # 듀티 사이클이 높을수록 부작용 위험 증가
        duty_factor = params['duty_cycle'] / 100.0
        
        # 시간이 지남에 따라 약간의 적응으로 부작용 감소
        adaptation = min(0.5, i / (duration_days / 2)) if i > 0 else 0
        
        base_effect = intensity_factor * duty_factor * (1.0 - adaptation)
        daily_variation = np.random.normal(0, 0.2)
        
        side_effects[i] = base_effect * 10 + daily_variation  # 0-10 스케일
        side_effects[i] = max(0, min(10, side_effects[i]))
    
    # 결과 반환
    response = {
        'days': days,
        'growth_rates': growth_rates,
        'growth_factors': growth_factors,
        'recovery_scores': recovery_scores,
        'conduction_velocities': conduction_velocities,
        'side_effects': side_effects,
        'params': params,
        'nerve_state': nerve_state,
        'total_effect': total_effect
    }
    
    return response


def visualize_stimulation_response(response, title=None, save_path=None):
    """
    전기자극 반응 시뮬레이션 결과 시각화 함수
    
    Parameters:
    -----------
    response : dict
        시뮬레이션 결과
    title : str
        그래프 제목
    save_path : str
        이미지 저장 경로
    """
    days = response['days']
    
    plt.figure(figsize=(15, 12))
    
    # 축삭 성장률 플롯
    plt.subplot(3, 2, 1)
    plt.plot(days, response['growth_rates'], 'g-', linewidth=2)
    plt.fill_between(days, response['growth_rates'] - 0.5, response['growth_rates'] + 0.5, color='g', alpha=0.2)
    plt.title('축삭 성장률')
    plt.xlabel('시간 (일)')
    plt.ylabel('성장률 (µm/일)')
    plt.grid(True, alpha=0.3)
    
    # 신경성장인자 발현 플롯
    plt.subplot(3, 2, 2)
    plt.plot(days, response['growth_factors'], 'b-', linewidth=2)
    plt.fill_between(days, response['growth_factors'] - 0.3, response['growth_factors'] + 0.3, color='b', alpha=0.2)
    plt.title('신경성장인자 발현')
    plt.xlabel('시간 (일)')
    plt.ylabel('발현 수준 (상대적 단위)')
    plt.grid(True, alpha=0.3)
    
    # 기능적 회복 점수 플롯
    plt.subplot(3, 2, 3)
    plt.plot(days, response['recovery_scores'], 'r-', linewidth=2)
    plt.fill_between(days, response['recovery_scores'] - 1, response['recovery_scores'] + 1, color='r', alpha=0.2)
    plt.title('기능적 회복 점수')
    plt.xlabel('시간 (일)')
    plt.ylabel('회복 점수 (0-100)')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    
    # 신경 전도 속도 플롯
    plt.subplot(3, 2, 4)
    plt.plot(days, response['conduction_velocities'], 'c-', linewidth=2)
    plt.fill_between(days, response['conduction_velocities'] - 0.5, response['conduction_velocities'] + 0.5, color='c', alpha=0.2)
    plt.title('신경 전도 속도')
    plt.xlabel('시간 (일)')
    plt.ylabel('전도 속도 (m/s)')
    plt.grid(True, alpha=0.3)
    
    # 부작용 심각도 플롯
    plt.subplot(3, 2, 5)
    plt.plot(days, response['side_effects'], 'k-', linewidth=2)
    plt.fill_between(days, response['side_effects'] - 0.2, response['side_effects'] + 0.2, color='k', alpha=0.2)
    plt.title('부작용 심각도')
    plt.xlabel('시간 (일)')
    plt.ylabel('심각도 (0-10)')
    plt.ylim(0, 10)
    plt.grid(True, alpha=0.3)
    
    # 자극 파라미터 요약
    plt.subplot(3, 2, 6)
    plt.axis('off')
    params = response['params']
    params_text = f"자극 파라미터:\n\n" \
                 f"주파수: {params['frequency']} Hz\n" \
                 f"진폭: {params['amplitude']} mA\n" \
                 f"펄스폭: {params['pulse_width']} µs\n" \
                 f"듀티 사이클: {params['duty_cycle']}%\n" \
                 f"자극 기간: {params['duration']} 분\n\n" \
                 f"신경 상태: {response['nerve_state']}\n" \
                 f"총 효과 점수: {response['total_effect']:.2f}"
    plt.text(0.5, 0.5, params_text, horizontalalignment='center', verticalalignment='center', fontsize=12)
    
    # 전체 제목 설정
    if title:
        plt.suptitle(title, fontsize=16, y=1.02)
    else:
        plt.suptitle(f"신경 상태 '{response['nerve_state']}'에 대한 전기자극 반응 시뮬레이션", fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    # 이미지 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_stimulation_protocols(responses, protocol_names=None, save_path=None):
    """
    여러 전기자극 프로토콜의 결과 비교 시각화 함수
    
    Parameters:
    -----------
    responses : list of dict
        여러 시뮬레이션 결과 목록
    protocol_names : list of str
        프로토콜 이름 목록
    save_path : str
        이미지 저장 경로
    """
    if protocol_names is None:
        protocol_names = [f"프로토콜 {i+1}" for i in range(len(responses))]
    
    # 모든 응답이 같은 길이인지 확인
    days = responses[0]['days']
    for resp in responses[1:]:
        if len(resp['days']) != len(days):
            raise ValueError("모든 응답은 같은 기간을 가져야 합니다.")
    
    # 플롯 색상
    colors = plt.cm.tab10(np.linspace(0, 1, len(responses)))
    
    plt.figure(figsize=(15, 15))
    
    # 성장률 비교
    plt.subplot(3, 2, 1)
    for i, resp in enumerate(responses):
        plt.plot(days, resp['growth_rates'], color=colors[i], linewidth=2, label=protocol_names[i])
    plt.title('축삭 성장률 비교')
    plt.xlabel('시간 (일)')
    plt.ylabel('성장률 (µm/일)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # 신경성장인자 발현 비교
    plt.subplot(3, 2, 2)
    for i, resp in enumerate(responses):
        plt.plot(days, resp['growth_factors'], color=colors[i], linewidth=2, label=protocol_names[i])
    plt.title('신경성장인자 발현 비교')
    plt.xlabel('시간 (일)')
    plt.ylabel('발현 수준 (상대적 단위)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # 기능적 회복 점수 비교
    plt.subplot(3, 2, 3)
    for i, resp in enumerate(responses):
        plt.plot(days, resp['recovery_scores'], color=colors[i], linewidth=2, label=protocol_names[i])
    plt.title('기능적 회복 점수 비교')
    plt.xlabel('시간 (일)')
    plt.ylabel('회복 점수 (0-100)')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # 신경 전도 속도 비교
    plt.subplot(3, 2, 4)
    for i, resp in enumerate(responses):
        plt.plot(days, resp['conduction_velocities'], color=colors[i], linewidth=2, label=protocol_names[i])
    plt.title('신경 전도 속도 비교')
    plt.xlabel('시간 (일)')
    plt.ylabel('전도 속도 (m/s)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # 부작용 심각도 비교
    plt.subplot(3, 2, 5)
    for i, resp in enumerate(responses):
        plt.plot(days, resp['side_effects'], color=colors[i], linewidth=2, label=protocol_names[i])
    plt.title('부작용 심각도 비교')
    plt.xlabel('시간 (일)')
    plt.ylabel('심각도 (0-10)')
    plt.ylim(0, 10)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # 효과 점수 요약
    plt.subplot(3, 2, 6)
    plt.axis('off')
    
    # 프로토콜별 총 효과 점수 및 30일 후 지표 요약
    summary_text = "프로토콜 비교 요약:\n\n"
    
    for i, resp in enumerate(responses):
        final_growth = resp['growth_rates'][-1]
        final_recovery = resp['recovery_scores'][-1]
        final_velocity = resp['conduction_velocities'][-1]
        final_side_effect = resp['side_effects'][-1]
        
        summary_text += f"{protocol_names[i]}:\n" \
                       f"  총 효과 점수: {resp['total_effect']:.2f}\n" \
                       f"  최종 성장률: {final_growth:.2f} µm/일\n" \
                       f"  최종 회복 점수: {final_recovery:.2f}/100\n" \
                       f"  최종 전도 속도: {final_velocity:.2f} m/s\n" \
                       f"  최종 부작용 심각도: {final_side_effect:.2f}/10\n\n"
    
    plt.text(0.5, 0.5, summary_text, horizontalalignment='center', verticalalignment='center', fontsize=11)
    
    plt.suptitle("전기자극 프로토콜 비교", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # 이미지 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def load_stimulation_data(file_path):
    """
    전기자극 데이터 로드 함수
    
    Parameters:
    -----------
    file_path : str
        데이터 파일 경로
        
    Returns:
    --------
    data : DataFrame
        전기자극 데이터
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.csv':
            data = pd.read_csv(file_path)
        elif ext == '.xlsx' or ext == '.xls':
            data = pd.read_excel(file_path)
        elif ext == '.json':
            data = pd.read_json(file_path)
        else:
            raise ValueError(f"지원되지 않는 파일 형식: {ext}")
        
        return data
    
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return None


def save_stimulation_protocol(protocol, output_path):
    """
    전기자극 프로토콜 저장 함수
    
    Parameters:
    -----------
    protocol : dict
        전기자극 프로토콜 정보
    output_path : str
        출력 파일 경로
    """
    try:
        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 파일 확장자에 따라 저장 방식 결정
        ext = os.path.splitext(output_path)[1].lower()
        
        if ext == '.json':
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(protocol, f, ensure_ascii=False, indent=4)
        elif ext == '.csv':
            # 프로토콜 구조를 평탄화하여 CSV로 저장
            flat_protocol = {}
            for state, data in protocol.items():
                for param, value in data['parameters'].items():
                    flat_protocol[f"{state}_{param}"] = value
                for metric, value in data['expected_metrics'].items():
                    flat_protocol[f"{state}_{metric}"] = value
            
            pd.DataFrame([flat_protocol]).to_csv(output_path, index=False)
        else:
            raise ValueError(f"지원되지 않는 파일 형식: {ext}")
        
        print(f"프로토콜이 {output_path}에 저장되었습니다.")
    
    except Exception as e:
        print(f"프로토콜 저장 실패: {e}")
