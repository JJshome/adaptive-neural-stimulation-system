"""
공개 DBS/LFP 데이터셋 로더 및 통합 모듈

이 모듈은 공개된 파킨슨병 환자의 DBS/LFP 데이터셋을 
로드하고 통합하는 기능을 제공합니다.

지원하는 데이터셋:
- OpenNeuro datasets
- NDAR (National Database for Autism Research)
- Physionet databases
- Custom formats
"""

import os
import json
import h5py
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import scipy.io as sio
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """데이터셋 정보"""
    name: str
    source: str
    url: str
    description: str
    n_subjects: int
    n_sessions: int
    brain_regions: List[str]
    sampling_rate: float
    license: str


class BaseDatasetLoader(ABC):
    """데이터셋 로더 기본 클래스"""
    
    def __init__(self, data_dir: str = "./data/external"):
        """
        초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def load_dataset(self, dataset_id: str) -> Dict:
        """데이터셋 로드"""
        pass
    
    @abstractmethod
    def preprocess_data(self, raw_data: Dict) -> pd.DataFrame:
        """데이터 전처리"""
        pass
    
    def download_if_needed(self, url: str, filename: str) -> Path:
        """필요시 파일 다운로드"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.info(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded to {filepath}")
            
        return filepath


class OpenNeuroLoader(BaseDatasetLoader):
    """OpenNeuro 데이터셋 로더"""
    
    KNOWN_DATASETS = {
        'ds003420': DatasetInfo(
            name='STN-LFP Recordings in Parkinson\'s Disease',
            source='OpenNeuro',
            url='https://openneuro.org/datasets/ds003420',
            description='STN LFP recordings from PD patients during DBS surgery',
            n_subjects=12,
            n_sessions=24,
            brain_regions=['STN', 'GPi'],
            sampling_rate=24000.0,
            license='CC0'
        ),
        'ds002778': DatasetInfo(
            name='Intraoperative STN recordings',
            source='OpenNeuro',
            url='https://openneuro.org/datasets/ds002778',
            description='Microelectrode recordings from STN during DBS implantation',
            n_subjects=20,
            n_sessions=20,
            brain_regions=['STN'],
            sampling_rate=44100.0,
            license='PDDL'
        )
    }
    
    def load_dataset(self, dataset_id: str) -> Dict:
        """
        OpenNeuro 데이터셋 로드
        
        Args:
            dataset_id: 데이터셋 ID (예: 'ds003420')
            
        Returns:
            로드된 데이터
        """
        if dataset_id not in self.KNOWN_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_id}")
            
        dataset_info = self.KNOWN_DATASETS[dataset_id]
        
        # 실제 구현시에는 OpenNeuro API를 사용하여 다운로드
        # 여기서는 예시 구조만 제공
        logger.info(f"Loading {dataset_info.name}...")
        
        # 데이터 구조 예시
        data = {
            'info': dataset_info,
            'subjects': [],
            'lfp_data': {},
            'metadata': {}
        }
        
        # 실제 데이터 로딩 로직
        # (OpenNeuro API 또는 직접 다운로드 구현 필요)
        
        return data
    
    def preprocess_data(self, raw_data: Dict) -> pd.DataFrame:
        """
        데이터 전처리
        
        Args:
            raw_data: 원시 데이터
            
        Returns:
            전처리된 DataFrame
        """
        processed_data = []
        
        # 데이터 전처리 로직
        # - 리샘플링
        # - 필터링
        # - 세그멘테이션
        # - 특징 추출
        
        return pd.DataFrame(processed_data)


class PhysionetLoader(BaseDatasetLoader):
    """Physionet 데이터베이스 로더"""
    
    DATABASES = {
        'gaitpdb': {
            'name': 'Gait in Parkinson\'s Disease',
            'url': 'https://physionet.org/content/gaitpdb/1.0.0/',
            'description': 'Gait recordings from PD patients and controls'
        },
        'parkinsonsdisease': {
            'name': 'Parkinson\'s Disease Classification',
            'url': 'https://physionet.org/content/parkinsonsdisease/1.0.0/',
            'description': 'Voice recordings for PD detection'
        }
    }
    
    def load_dataset(self, dataset_id: str) -> Dict:
        """Physionet 데이터 로드"""
        # 구현 필요
        pass
    
    def preprocess_data(self, raw_data: Dict) -> pd.DataFrame:
        """데이터 전처리"""
        # 구현 필요
        pass


class CustomLFPLoader(BaseDatasetLoader):
    """사용자 정의 LFP 데이터 로더"""
    
    SUPPORTED_FORMATS = ['.mat', '.h5', '.hdf5', '.csv', '.txt', '.edf']
    
    def load_dataset(self, filepath: str) -> Dict:
        """
        다양한 형식의 LFP 데이터 로드
        
        Args:
            filepath: 데이터 파일 경로
            
        Returns:
            로드된 데이터
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        ext = filepath.suffix.lower()
        
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}")
            
        if ext == '.mat':
            return self._load_matlab(filepath)
        elif ext in ['.h5', '.hdf5']:
            return self._load_hdf5(filepath)
        elif ext == '.csv':
            return self._load_csv(filepath)
        elif ext == '.txt':
            return self._load_text(filepath)
        elif ext == '.edf':
            return self._load_edf(filepath)
            
    def _load_matlab(self, filepath: Path) -> Dict:
        """MATLAB 파일 로드"""
        try:
            mat_data = sio.loadmat(str(filepath))
            
            # MATLAB 구조체 파싱
            data = {
                'lfp_data': None,
                'sampling_rate': None,
                'metadata': {}
            }
            
            # 일반적인 키 이름들 확인
            for key in ['lfp', 'LFP', 'data', 'signal']:
                if key in mat_data:
                    data['lfp_data'] = mat_data[key]
                    break
                    
            for key in ['fs', 'Fs', 'sampling_rate', 'srate']:
                if key in mat_data:
                    data['sampling_rate'] = float(mat_data[key])
                    break
                    
            return data
            
        except Exception as e:
            logger.error(f"Error loading MATLAB file: {e}")
            raise
            
    def _load_hdf5(self, filepath: Path) -> Dict:
        """HDF5 파일 로드"""
        try:
            with h5py.File(filepath, 'r') as f:
                data = {
                    'lfp_data': f['lfp_data'][:] if 'lfp_data' in f else None,
                    'sampling_rate': f.attrs.get('sampling_rate', None),
                    'metadata': dict(f.attrs)
                }
                
            return data
            
        except Exception as e:
            logger.error(f"Error loading HDF5 file: {e}")
            raise
            
    def _load_csv(self, filepath: Path) -> Dict:
        """CSV 파일 로드"""
        try:
            df = pd.read_csv(filepath)
            
            # 첫 번째 열이 시간이라고 가정
            if 'time' in df.columns:
                time = df['time'].values
                sampling_rate = 1 / np.mean(np.diff(time))
                lfp_columns = [col for col in df.columns if col != 'time']
            else:
                # 샘플링 레이트를 메타데이터에서 찾아야 함
                sampling_rate = None
                lfp_columns = df.columns
                
            data = {
                'lfp_data': df[lfp_columns].values.T,
                'sampling_rate': sampling_rate,
                'metadata': {'columns': list(lfp_columns)}
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
            
    def _load_text(self, filepath: Path) -> Dict:
        """텍스트 파일 로드"""
        try:
            data_array = np.loadtxt(filepath)
            
            data = {
                'lfp_data': data_array.T if data_array.ndim > 1 else data_array,
                'sampling_rate': None,  # 메타데이터에서 확인 필요
                'metadata': {'shape': data_array.shape}
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading text file: {e}")
            raise
            
    def _load_edf(self, filepath: Path) -> Dict:
        """EDF (European Data Format) 파일 로드"""
        # pyedflib 라이브러리 필요
        try:
            import pyedflib
            
            with pyedflib.EdfReader(str(filepath)) as f:
                n_channels = f.signals_in_file
                signal_labels = f.getSignalLabels()
                sampling_rates = [f.getSampleFrequency(i) for i in range(n_channels)]
                
                # 모든 채널 데이터 로드
                lfp_data = np.array([f.readSignal(i) for i in range(n_channels)])
                
                data = {
                    'lfp_data': lfp_data,
                    'sampling_rate': sampling_rates[0] if len(set(sampling_rates)) == 1 else sampling_rates,
                    'metadata': {
                        'labels': signal_labels,
                        'patient_info': f.getPatientInfo(),
                        'recording_info': f.getRecordingInfo()
                    }
                }
                
            return data
            
        except ImportError:
            logger.error("pyedflib not installed. Install with: pip install pyedflib")
            raise
        except Exception as e:
            logger.error(f"Error loading EDF file: {e}")
            raise
            
    def preprocess_data(self, raw_data: Dict) -> pd.DataFrame:
        """
        LFP 데이터 전처리
        
        Args:
            raw_data: 원시 데이터
            
        Returns:
            전처리된 DataFrame
        """
        lfp_data = raw_data['lfp_data']
        sampling_rate = raw_data['sampling_rate']
        
        if lfp_data is None or sampling_rate is None:
            raise ValueError("Missing LFP data or sampling rate")
            
        # 데이터 형태 확인
        if lfp_data.ndim == 1:
            lfp_data = lfp_data.reshape(1, -1)
            
        n_channels, n_samples = lfp_data.shape
        
        # 시간 벡터 생성
        time = np.arange(n_samples) / sampling_rate
        
        # DataFrame 생성
        df_data = {'time': time}
        for i in range(n_channels):
            df_data[f'channel_{i+1}'] = lfp_data[i, :]
            
        df = pd.DataFrame(df_data)
        
        # 메타데이터 추가
        df.attrs['sampling_rate'] = sampling_rate
        df.attrs['n_channels'] = n_channels
        df.attrs['duration'] = n_samples / sampling_rate
        
        return df


class UnifiedDatasetManager:
    """통합 데이터셋 관리자"""
    
    def __init__(self, data_dir: str = "./data"):
        """
        초기화
        
        Args:
            data_dir: 데이터 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.loaders = {
            'openneuro': OpenNeuroLoader(data_dir),
            'physionet': PhysionetLoader(data_dir),
            'custom': CustomLFPLoader(data_dir)
        }
        self.cached_datasets = {}
        
    def load_dataset(self, 
                    source: str,
                    dataset_id: str,
                    cache: bool = True) -> pd.DataFrame:
        """
        데이터셋 로드
        
        Args:
            source: 데이터 소스 ('openneuro', 'physionet', 'custom')
            dataset_id: 데이터셋 ID 또는 파일 경로
            cache: 캐시 사용 여부
            
        Returns:
            로드된 데이터 DataFrame
        """
        cache_key = f"{source}:{dataset_id}"
        
        if cache and cache_key in self.cached_datasets:
            logger.info(f"Loading from cache: {cache_key}")
            return self.cached_datasets[cache_key]
            
        if source not in self.loaders:
            raise ValueError(f"Unknown source: {source}")
            
        loader = self.loaders[source]
        raw_data = loader.load_dataset(dataset_id)
        processed_data = loader.preprocess_data(raw_data)
        
        if cache:
            self.cached_datasets[cache_key] = processed_data
            
        return processed_data
    
    def combine_datasets(self, 
                        datasets: List[Tuple[str, str]],
                        labels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        여러 데이터셋 결합
        
        Args:
            datasets: (source, dataset_id) 튜플 리스트
            labels: 각 데이터셋의 레이블
            
        Returns:
            결합된 DataFrame
        """
        combined_data = []
        
        if labels is None:
            labels = [f"dataset_{i}" for i in range(len(datasets))]
            
        for (source, dataset_id), label in zip(datasets, labels):
            data = self.load_dataset(source, dataset_id)
            data['dataset_label'] = label
            combined_data.append(data)
            
        return pd.concat(combined_data, ignore_index=True)
    
    def export_for_training(self,
                          data: pd.DataFrame,
                          output_dir: str,
                          format: str = 'npz') -> Path:
        """
        학습용 데이터 내보내기
        
        Args:
            data: 데이터 DataFrame
            output_dir: 출력 디렉토리
            format: 출력 형식 ('npz', 'h5', 'tfrecord')
            
        Returns:
            저장된 파일 경로
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'npz':
            filepath = output_dir / f"lfp_data_{timestamp}.npz"
            
            # 수치 데이터만 추출
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            np.savez(filepath,
                    data=data[numeric_cols].values,
                    columns=numeric_cols.tolist(),
                    metadata=data.attrs)
                    
        elif format == 'h5':
            filepath = output_dir / f"lfp_data_{timestamp}.h5"
            data.to_hdf(filepath, key='lfp_data', mode='w')
            
        elif format == 'tfrecord':
            # TensorFlow 형식 (구현 필요)
            raise NotImplementedError("TFRecord export not yet implemented")
            
        else:
            raise ValueError(f"Unknown format: {format}")
            
        logger.info(f"Exported data to {filepath}")
        return filepath


# 사용 예시
if __name__ == "__main__":
    # 데이터셋 매니저 초기화
    manager = UnifiedDatasetManager()
    
    # OpenNeuro 데이터셋 로드 (예시)
    # data = manager.load_dataset('openneuro', 'ds003420')
    
    # 사용자 정의 파일 로드
    # custom_data = manager.load_dataset('custom', './my_lfp_data.mat')
    
    # 여러 데이터셋 결합
    # combined = manager.combine_datasets([
    #     ('openneuro', 'ds003420'),
    #     ('custom', './my_data.csv')
    # ])
    
    # 학습용으로 내보내기
    # manager.export_for_training(combined, './processed_data')
    
    print("Dataset loader module initialized successfully")
