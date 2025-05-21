"""
데이터 관리 유틸리티

이 모듈은 신경 신호 데이터 세트를 로드, 변환, 저장하는 기능을 제공합니다.
다양한 형식의 데이터 파일을 처리하고 표준화된 형식으로 변환합니다.
"""

import os
import numpy as np
import pandas as pd
import h5py
from typing import List, Tuple, Dict, Any, Optional, Union

class DataLoader:
    """다양한 형식의 신경 신호 데이터를 로드하는 클래스"""
    
    def __init__(self):
        """DataLoader 초기화"""
        pass
        
    def load_csv(self, file_path: str) -> Tuple[np.ndarray, float]:
        """
        CSV 파일에서 신경 신호 데이터 로드
        
        매개변수:
            file_path (str): CSV 파일 경로
            
        반환값:
            Tuple[np.ndarray, float]: (데이터 배열, 샘플링 레이트)
        """
        try:
            # CSV 파일 로드
            df = pd.read_csv(file_path)
            
            # 첫 번째 열이 시간 또는 인덱스인지 확인
            if 'time' in df.columns or 'timestamp' in df.columns:
                time_col = 'time' if 'time' in df.columns else 'timestamp'
                # 샘플링 레이트 추정
                time_diff = np.diff(df[time_col].values)
                sampling_rate = 1.0 / np.mean(time_diff)
                
                # 신호 데이터 열 추출
                signal_cols = [col for col in df.columns if col != time_col]
                data = df[signal_cols].values
            else:
                # 샘플링 레이트를 결정할 수 없는 경우 기본값 사용
                sampling_rate = 1000.0  # 기본 1kHz
                data = df.values
                
            return data, sampling_rate
            
        except Exception as e:
            print(f"CSV 파일 로드 중 오류: {e}")
            return np.array([]), 0.0
    
    def load_mat(self, file_path: str, var_name: Optional[str] = None) -> Tuple[np.ndarray, float]:
        """
        MATLAB .mat 파일에서 신경 신호 데이터 로드
        
        매개변수:
            file_path (str): .mat 파일 경로
            var_name (Optional[str]): 로드할 변수 이름 (None인 경우 첫 번째 변수 사용)
            
        반환값:
            Tuple[np.ndarray, float]: (데이터 배열, 샘플링 레이트)
        """
        try:
            from scipy import io
            
            # .mat 파일 로드
            mat_data = io.loadmat(file_path)
            
            # 변수 이름이 지정되지 않은 경우 첫 번째 변수 사용
            if var_name is None:
                # 'built-in' 변수 제외
                var_names = [key for key in mat_data.keys() if not key.startswith('__')]
                if not var_names:
                    raise ValueError("유효한 변수를 찾을 수 없습니다")
                var_name = var_names[0]
            
            # 데이터 추출
            data = mat_data[var_name]
            
            # 샘플링 레이트 추출 시도
            sampling_rate = 1000.0  # 기본값
            if 'fs' in mat_data:
                sampling_rate = float(mat_data['fs'])
            elif 'Fs' in mat_data:
                sampling_rate = float(mat_data['Fs'])
            elif 'sampling_rate' in mat_data:
                sampling_rate = float(mat_data['sampling_rate'])
                
            return data, sampling_rate
            
        except Exception as e:
            print(f".mat 파일 로드 중 오류: {e}")
            return np.array([]), 0.0
    
    def load_npy(self, file_path: str, sampling_rate: float = 1000.0) -> Tuple[np.ndarray, float]:
        """
        NumPy .npy 파일에서 신경 신호 데이터 로드
        
        매개변수:
            file_path (str): .npy 파일 경로
            sampling_rate (float): 샘플링 레이트 (외부에서 제공)
            
        반환값:
            Tuple[np.ndarray, float]: (데이터 배열, 샘플링 레이트)
        """
        try:
            data = np.load(file_path)
            return data, sampling_rate
        except Exception as e:
            print(f".npy 파일 로드 중 오류: {e}")
            return np.array([]), 0.0
            
    def load_hdf5(self, file_path: str, dataset_name: str = 'data') -> Tuple[np.ndarray, float]:
        """
        HDF5 파일에서 신경 신호 데이터 로드
        
        매개변수:
            file_path (str): HDF5 파일 경로
            dataset_name (str): 로드할 데이터셋 이름
            
        반환값:
            Tuple[np.ndarray, float]: (데이터 배열, 샘플링 레이트)
        """
        try:
            with h5py.File(file_path, 'r') as f:
                data = f[dataset_name][:]
                
                # 샘플링 레이트 추출 시도
                sampling_rate = 1000.0  # 기본값
                if 'sampling_rate' in f.attrs:
                    sampling_rate = float(f.attrs['sampling_rate'])
                elif 'fs' in f.attrs:
                    sampling_rate = float(f.attrs['fs'])
                    
                return data, sampling_rate
                
        except Exception as e:
            print(f"HDF5 파일 로드 중 오류: {e}")
            return np.array([]), 0.0

class DataTransformer:
    """신경 신호 데이터 변환을 위한 클래스"""
    
    def __init__(self):
        """DataTransformer 초기화"""
        pass
        
    def normalize(self, data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        데이터 정규화
        
        매개변수:
            data (np.ndarray): 정규화할 데이터
            method (str): 정규화 방법 ('minmax', 'zscore', 'robust')
            
        반환값:
            np.ndarray: 정규화된 데이터
        """
        if method == 'minmax':
            # Min-Max 정규화 (0-1 범위)
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val == min_val:
                return np.zeros_like(data)
            return (data - min_val) / (max_val - min_val)
            
        elif method == 'zscore':
            # Z-score 정규화 (평균 0, 표준편차 1)
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return np.zeros_like(data)
            return (data - mean) / std
            
        elif method == 'robust':
            # 로버스트 정규화 (중앙값과 IQR 사용)
            median = np.median(data)
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                return np.zeros_like(data)
            return (data - median) / iqr
            
        else:
            raise ValueError(f"지원되지 않는 정규화 방법: {method}")
    
    def segment_data(self, data: np.ndarray, window_size: int, overlap: int = 0) -> List[np.ndarray]:
        """
        데이터를 고정 크기 세그먼트로 분할
        
        매개변수:
            data (np.ndarray): 분할할 데이터
            window_size (int): 윈도우 크기 (샘플 수)
            overlap (int): 인접 윈도우 간 중첩 샘플 수
            
        반환값:
            List[np.ndarray]: 분할된 세그먼트 리스트
        """
        if len(data) < window_size:
            return [data]
            
        stride = window_size - overlap
        segments = []
        
        for i in range(0, len(data) - window_size + 1, stride):
            segment = data[i:i + window_size]
            segments.append(segment)
            
        return segments
    
    def apply_windowing(self, data: np.ndarray, window_type: str = 'hann') -> np.ndarray:
        """
        데이터에 윈도우 함수 적용
        
        매개변수:
            data (np.ndarray): 처리할 데이터
            window_type (str): 윈도우 타입 ('hann', 'hamming', 'blackman', 등)
            
        반환값:
            np.ndarray: 윈도우가 적용된 데이터
        """
        if window_type == 'hann':
            window = np.hanning(len(data))
        elif window_type == 'hamming':
            window = np.hamming(len(data))
        elif window_type == 'blackman':
            window = np.blackman(len(data))
        else:
            raise ValueError(f"지원되지 않는 윈도우 타입: {window_type}")
            
        return data * window
    
    def prepare_sequence_data(self, data: np.ndarray, sequence_length: int, step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        시계열 예측을 위한 시퀀스 데이터 준비
        
        매개변수:
            data (np.ndarray): 원본 시계열 데이터
            sequence_length (int): 입력 시퀀스 길이
            step (int): 연속된 시퀀스 간 스텝 크기
            
        반환값:
            Tuple[np.ndarray, np.ndarray]: (입력 시퀀스, 타겟값)
        """
        X, y = [], []
        
        for i in range(0, len(data) - sequence_length, step):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
            
        return np.array(X), np.array(y)
