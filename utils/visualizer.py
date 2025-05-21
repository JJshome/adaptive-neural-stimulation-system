"""
데이터 시각화 유틸리티

이 모듈은 신경 신호와 자극 매개변수의 시각화 도구를 제공합니다.
신호 파형, 스펙트럼, 발화율, 최적화 과정 등의 시각화 기능을 구현합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Union

class Visualizer:
    """데이터 시각화를 위한 클래스"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Visualizer 초기화
        
        매개변수:
            figsize (Tuple[int, int]): 기본 그래프 크기
        """
        self.figsize = figsize
        
    def plot_signal(self, data: np.ndarray, sampling_rate: float = 1000.0, 
                    title: str = "신경 신호", save_path: Optional[str] = None) -> plt.Figure:
        """
        시계열 신호 플롯
        
        매개변수:
            data (np.ndarray): 시각화할 신호 데이터
            sampling_rate (float): 샘플링 레이트 (Hz)
            title (str): 그래프 제목
            save_path (Optional[str]): 저장 경로
            
        반환값:
            plt.Figure: 생성된 그림 객체
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 시간 축 생성
        time = np.arange(len(data)) / sampling_rate
        
        # 신호 플롯
        ax.plot(time, data)
        
        # 축 및 제목 설정
        ax.set_xlabel('시간 (초)')
        ax.set_ylabel('진폭')
        ax.set_title(title)
        ax.grid(True)
        
        # 선택적으로 파일 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_spectrogram(self, data: np.ndarray, sampling_rate: float = 1000.0,
                         title: str = "스펙트로그램", save_path: Optional[str] = None) -> plt.Figure:
        """
        신호의 스펙트로그램 플롯
        
        매개변수:
            data (np.ndarray): 시각화할 신호 데이터
            sampling_rate (float): 샘플링 레이트 (Hz)
            title (str): 그래프 제목
            save_path (Optional[str]): 저장 경로
            
        반환값:
            plt.Figure: 생성된 그림 객체
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 스펙트로그램 계산 및 플롯
        spec, freqs, t, im = ax.specgram(data, NFFT=256, Fs=sampling_rate, 
                                        noverlap=128, cmap='viridis')
        
        # 축 및 제목 설정
        ax.set_xlabel('시간 (초)')
        ax.set_ylabel('주파수 (Hz)')
        ax.set_title(title)
        
        # 컬러바 추가
        fig.colorbar(im, ax=ax, label='전력 스펙트럼 밀도 (dB)')
        
        # 선택적으로 파일 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_psd(self, data: np.ndarray, sampling_rate: float = 1000.0,
                 title: str = "전력 스펙트럼 밀도", save_path: Optional[str] = None) -> plt.Figure:
        """
        전력 스펙트럼 밀도 플롯
        
        매개변수:
            data (np.ndarray): 시각화할 신호 데이터
            sampling_rate (float): 샘플링 레이트 (Hz)
            title (str): 그래프 제목
            save_path (Optional[str]): 저장 경로
            
        반환값:
            plt.Figure: 생성된 그림 객체
        """
        from scipy import signal
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 전력 스펙트럼 밀도 계산
        f, psd = signal.welch(data, fs=sampling_rate, nperseg=1024)
        
        # dB 단위로 변환
        psd_db = 10 * np.log10(psd)
        
        # PSD 플롯
        ax.plot(f, psd_db)
        
        # 축 및 제목 설정
        ax.set_xlabel('주파수 (Hz)')
        ax.set_ylabel('전력 스펙트럼 밀도 (dB/Hz)')
        ax.set_title(title)
        ax.grid(True)
        
        # y축 범위 설정
        ax.set_ylim([np.min(psd_db) - 5, np.max(psd_db) + 5])
        
        # 선택적으로 파일 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_firing_rate(self, firing_rate: np.ndarray, bin_size: float = 0.1,
                         title: str = "발화율", save_path: Optional[str] = None) -> plt.Figure:
        """
        발화율 플롯
        
        매개변수:
            firing_rate (np.ndarray): 발화율 데이터
            bin_size (float): 빈 크기 (초)
            title (str): 그래프 제목
            save_path (Optional[str]): 저장 경로
            
        반환값:
            plt.Figure: 생성된 그림 객체
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 시간 축 생성
        time = np.arange(len(firing_rate)) * bin_size
        
        # 발화율 플롯
        ax.plot(time, firing_rate)
        
        # 축 및 제목 설정
        ax.set_xlabel('시간 (초)')
        ax.set_ylabel('발화율 (스파이크/초)')
        ax.set_title(title)
        ax.grid(True)
        
        # 선택적으로 파일 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_spikes(self, spike_times: List[float], duration: float,
                    title: str = "스파이크 발생 시간", save_path: Optional[str] = None) -> plt.Figure:
        """
        스파이크 발생 시간 플롯
        
        매개변수:
            spike_times (List[float]): 스파이크 발생 시간 (초)
            duration (float): 총 기록 시간 (초)
            title (str): 그래프 제목
            save_path (Optional[str]): 저장 경로
            
        반환값:
            plt.Figure: 생성된 그림 객체
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 스파이크 발생 위치에 수직선 그리기
        for spike_time in spike_times:
            ax.axvline(x=spike_time, color='r', linestyle='-', alpha=0.5)
        
        # 축 및 제목 설정
        ax.set_xlabel('시간 (초)')
        ax.set_yticks([])  # y축 눈금 제거
        ax.set_xlim([0, duration])
        ax.set_title(title)
        
        # 선택적으로 파일 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_stimulation_waveform(self, stimulation: np.ndarray, sampling_rate: float = 1000.0,
                                  title: str = "자극 파형", save_path: Optional[str] = None) -> plt.Figure:
        """
        자극 파형 플롯
        
        매개변수:
            stimulation (np.ndarray): 자극 파형 데이터
            sampling_rate (float): 샘플링 레이트 (Hz)
            title (str): 그래프 제목
            save_path (Optional[str]): 저장 경로
            
        반환값:
            plt.Figure: 생성된 그림 객체
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 시간 축 생성
        time = np.arange(len(stimulation)) / sampling_rate
        
        # 자극 파형 플롯
        ax.plot(time, stimulation)
        
        # 축 및 제목 설정
        ax.set_xlabel('시간 (초)')
        ax.set_ylabel('진폭 (mA)')
        ax.set_title(title)
        ax.grid(True)
        
        # 선택적으로 파일 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_optimization_progress(self, history: List[Dict[str, Any]], 
                                  parameter: str = 'score',
                                  title: str = "최적화 진행 상황", 
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        최적화 진행 상황 플롯
        
        매개변수:
            history (List[Dict[str, Any]]): 최적화 히스토리
            parameter (str): 시각화할 매개변수
            title (str): 그래프 제목
            save_path (Optional[str]): 저장 경로
            
        반환값:
            plt.Figure: 생성된 그림 객체
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 히스토리에서 데이터 추출
        if parameter == 'score':
            # 점수는 모든 히스토리 항목에 존재
            values = [entry['score'] for entry in history]
            indices = range(len(values))
            y_label = '목적 함수 값'
        else:
            # 매개변수 값 추출
            values = [entry['parameters'].get(parameter, np.nan) for entry in history]
            indices = range(len(values))
            y_label = f'{parameter} 값'
        
        # 데이터 플롯
        ax.plot(indices, values, '-o')
        
        # 축 및 제목 설정
        ax.set_xlabel('평가 회수')
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True)
        
        # 선택적으로 파일 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_parameter_comparison(self, history: List[Dict[str, Any]], 
                                 parameters: List[str],
                                 title: str = "매개변수 비교", 
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        여러 매개변수 비교 플롯
        
        매개변수:
            history (List[Dict[str, Any]]): 최적화 히스토리
            parameters (List[str]): 비교할 매개변수 목록
            title (str): 그래프 제목
            save_path (Optional[str]): 저장 경로
            
        반환값:
            plt.Figure: 생성된 그림 객체
        """
        fig, axs = plt.subplots(len(parameters), 1, figsize=(self.figsize[0], self.figsize[1] * len(parameters) // 2), 
                               sharex=True)
        
        # 단일 매개변수인 경우 리스트로 변환
        if len(parameters) == 1:
            axs = [axs]
        
        # 각 매개변수에 대한 그래프 그리기
        for i, param in enumerate(parameters):
            values = [entry['parameters'].get(param, np.nan) for entry in history]
            scores = [entry['score'] for entry in history]
            
            # 색상은 점수에 따라 매핑
            scatter = axs[i].scatter(range(len(values)), values, c=scores, cmap='viridis')
            
            # 축 및 라벨 설정
            axs[i].set_ylabel(f'{param}')
            axs[i].grid(True)
        
        # x축 라벨 (마지막 서브플롯에만)
        axs[-1].set_xlabel('평가 회수')
        
        # 전체 제목
        fig.suptitle(title)
        
        # 컬러바 추가
        plt.colorbar(scatter, ax=axs, label='목적 함수 값')
        
        # 여백 조정
        plt.tight_layout()
        
        # 선택적으로 파일 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_interactive_plot(self, data: Dict[str, np.ndarray], sampling_rate: float = 1000.0,
                               title: str = "대화형 플롯") -> None:
        """
        대화형 신호 시각화 플롯 (Jupyter Notebook용)
        
        매개변수:
            data (Dict[str, np.ndarray]): 시각화할 데이터 딕셔너리
            sampling_rate (float): 샘플링 레이트 (Hz)
            title (str): 그래프 제목
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            print("대화형 플롯에는 ipywidgets 패키지가 필요합니다.")
            return
        
        # 시간 범위 계산
        max_duration = max([len(signal) / sampling_rate for signal in data.values()])
        
        # 위젯 생성
        signal_dropdown = widgets.Dropdown(
            options=list(data.keys()),
            description='신호:'
        )
        
        time_slider = widgets.FloatRangeSlider(
            value=[0, min(10, max_duration)],
            min=0,
            max=max_duration,
            step=0.1,
            description='시간 범위:',
            continuous_update=False
        )
        
        # 플롯 업데이트 함수
        def update_plot(signal_name, time_range):
            plt.figure(figsize=self.figsize)
            
            # 선택된 신호 데이터
            selected_data = data[signal_name]
            
            # 시간 범위에 해당하는 데이터 인덱스
            start_idx = int(time_range[0] * sampling_rate)
            end_idx = int(time_range[1] * sampling_rate)
            
            # 시간 축 생성
            time = np.arange(start_idx, end_idx) / sampling_rate
            
            # 신호 데이터 플롯
            plt.plot(time, selected_data[start_idx:end_idx])
            
            # 축 및 제목 설정
            plt.xlabel('시간 (초)')
            plt.ylabel('진폭')
            plt.title(f"{title} - {signal_name}")
            plt.grid(True)
            
            plt.show()
            
        # 대화형 출력 위젯
        out = widgets.interactive(
            update_plot,
            signal_name=signal_dropdown,
            time_range=time_slider
        )
        
        display(out)
