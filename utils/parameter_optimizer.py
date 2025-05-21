"""
자극 매개변수 최적화 모듈

이 모듈은 신경 자극의 효과를 최적화하기 위한 알고리즘을 구현합니다.
다양한 최적화 전략과 신경생리학적 응답을 기반으로 자극 매개변수를 조정합니다.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

class ParameterOptimizer:
    """자극 매개변수 최적화를 위한 클래스"""
    
    def __init__(self):
        """
        ParameterOptimizer 초기화
        """
        # 매개변수 제약 조건 설정
        self.constraints = {
            'amplitude': {'min': 0.1, 'max': 10.0},  # mA
            'frequency': {'min': 1.0, 'max': 300.0},  # Hz
            'pulse_width': {'min': 10.0, 'max': 500.0},  # μs
        }
        
        # 최적화 히스토리
        self.optimization_history = []
    
    def grid_search(self, objective_function: Callable, parameter_ranges: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        그리드 탐색으로 최적 매개변수 찾기
        
        매개변수:
            objective_function (Callable): 최적화할 목적 함수
            parameter_ranges (Dict[str, List[float]]): 탐색할 매개변수 범위
            
        반환값:
            Dict[str, Any]: 최적 매개변수 및 목적 함수 값
        """
        # 입력 유효성 검사
        for param, values in parameter_ranges.items():
            if param not in self.constraints:
                raise ValueError(f"지원되지 않는 매개변수: {param}")
                
            # 매개변수 범위 제약 조건 적용
            min_val = self.constraints[param]['min']
            max_val = self.constraints[param]['max']
            parameter_ranges[param] = [v for v in values if min_val <= v <= max_val]
        
        # 그리드 탐색 초기화
        best_score = float('-inf')
        best_params = {}
        
        # 모든 매개변수 조합에 대해 목적 함수 평가
        param_names = list(parameter_ranges.keys())
        param_values = [parameter_ranges[param] for param in param_names]
        
        # 모든 가능한 매개변수 조합 생성
        from itertools import product
        for values in product(*param_values):
            current_params = {param_names[i]: values[i] for i in range(len(param_names))}
            
            # 목적 함수 평가
            score = objective_function(**current_params)
            
            # 최적 매개변수 업데이트
            if score > best_score:
                best_score = score
                best_params = current_params.copy()
                
            # 결과 기록
            self.optimization_history.append({
                'parameters': current_params.copy(),
                'score': score
            })
        
        # 최적 매개변수 및 점수 반환
        return {
            'parameters': best_params,
            'score': best_score
        }
    
    def particle_swarm_optimization(self, objective_function: Callable, 
                                    num_particles: int = 10, 
                                    num_iterations: int = 50) -> Dict[str, Any]:
        """
        입자 군집 최적화(PSO)를 사용한 매개변수 최적화
        
        매개변수:
            objective_function (Callable): 최적화할 목적 함수
            num_particles (int): 입자 수
            num_iterations (int): 반복 횟수
            
        반환값:
            Dict[str, Any]: 최적 매개변수 및 목적 함수 값
        """
        # 매개변수 공간 정의
        param_names = list(self.constraints.keys())
        param_mins = np.array([self.constraints[p]['min'] for p in param_names])
        param_maxs = np.array([self.constraints[p]['max'] for p in param_names])
        
        # PSO 초기화
        num_dimensions = len(param_names)
        
        # 입자 위치 및 속도 초기화
        positions = np.random.uniform(
            low=param_mins, 
            high=param_maxs, 
            size=(num_particles, num_dimensions)
        )
        velocities = np.random.uniform(
            low=-(param_maxs - param_mins), 
            high=(param_maxs - param_mins), 
            size=(num_particles, num_dimensions)
        )
        
        # 개인 및 전역 최적값 초기화
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([float('-inf')] * num_particles)
        
        global_best_idx = 0
        global_best_score = float('-inf')
        
        # PSO 하이퍼파라미터
        w = 0.7  # 관성 가중치
        c1 = 1.5  # 인지적 가중치 (개인 최적값)
        c2 = 1.5  # 사회적 가중치 (전역 최적값)
        
        # PSO 반복
        for iteration in range(num_iterations):
            # 각 입자의 현재 위치 평가
            for i in range(num_particles):
                # 매개변수 딕셔너리 구성
                current_params = {
                    param_names[j]: positions[i, j] for j in range(num_dimensions)
                }
                
                # 매개변수 제약 조건 적용
                for j, param in enumerate(param_names):
                    positions[i, j] = np.clip(
                        positions[i, j], 
                        self.constraints[param]['min'], 
                        self.constraints[param]['max']
                    )
                
                # 목적 함수 평가
                score = objective_function(**current_params)
                
                # 개인 최적값 업데이트
                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()
                    
                    # 전역 최적값 업데이트
                    if score > global_best_score:
                        global_best_score = score
                        global_best_idx = i
                
                # 결과 기록
                self.optimization_history.append({
                    'parameters': current_params.copy(),
                    'score': score,
                    'iteration': iteration
                })
            
            # 입자 속도 및 위치 업데이트
            r1 = np.random.random(size=(num_particles, num_dimensions))
            r2 = np.random.random(size=(num_particles, num_dimensions))
            
            velocities = (w * velocities + 
                         c1 * r1 * (personal_best_positions - positions) + 
                         c2 * r2 * (personal_best_positions[global_best_idx] - positions))
            
            # 속도 제한
            velocities = np.clip(
                velocities, 
                -(param_maxs - param_mins) * 0.1, 
                (param_maxs - param_mins) * 0.1
            )
            
            positions = positions + velocities
        
        # 최적 매개변수 구성
        best_params = {
            param_names[i]: personal_best_positions[global_best_idx][i]
            for i in range(num_dimensions)
        }
        
        # 최적 매개변수 및 점수 반환
        return {
            'parameters': best_params,
            'score': global_best_score
        }
    
    def bayesian_optimization(self, objective_function: Callable, 
                              num_initial_points: int = 5, 
                              num_iterations: int = 20) -> Dict[str, Any]:
        """
        베이지안 최적화를 사용한 매개변수 최적화
        
        매개변수:
            objective_function (Callable): 최적화할 목적 함수
            num_initial_points (int): 초기 탐색 포인트 수
            num_iterations (int): 반복 횟수
            
        반환값:
            Dict[str, Any]: 최적 매개변수 및 목적 함수 값
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
        except ImportError:
            raise ImportError("베이지안 최적화에는 scikit-learn이 필요합니다.")
        
        # 매개변수 공간 정의
        param_names = list(self.constraints.keys())
        param_mins = np.array([self.constraints[p]['min'] for p in param_names])
        param_maxs = np.array([self.constraints[p]['max'] for p in param_names])
        
        # 초기 탐색 포인트 생성
        num_dimensions = len(param_names)
        initial_points = np.random.uniform(
            low=param_mins, 
            high=param_maxs, 
            size=(num_initial_points, num_dimensions)
        )
        
        # 초기 포인트 평가
        X_observed = []
        y_observed = []
        
        for i in range(num_initial_points):
            current_params = {
                param_names[j]: initial_points[i, j] for j in range(num_dimensions)
            }
            
            score = objective_function(**current_params)
            
            X_observed.append(initial_points[i])
            y_observed.append(score)
            
            # 결과 기록
            self.optimization_history.append({
                'parameters': current_params.copy(),
                'score': score,
                'iteration': 'initial'
            })
        
        X_observed = np.array(X_observed)
        y_observed = np.array(y_observed)
        
        # 가우시안 프로세스 모델 초기화
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        
        # 획득 함수 (Expected Improvement)
        def expected_improvement(x, gp, best_y, epsilon=0.01):
            x = x.reshape(1, -1)
            mean, std = gp.predict(x, return_std=True)
            
            z = (mean - best_y - epsilon) / (std + 1e-9)
            ei = (mean - best_y - epsilon) * norm.cdf(z) + std * norm.pdf(z)
            
            return ei
        
        from scipy.stats import norm
        from scipy.optimize import minimize
        
        # 베이지안 최적화 반복
        for iteration in range(num_iterations):
            # 가우시안 프로세스 모델 학습
            gp.fit(X_observed, y_observed)
            
            # 현재 최적값
            best_idx = np.argmax(y_observed)
            best_y = y_observed[best_idx]
            
            # 획득 함수 최적화를 통한 다음 샘플링 포인트 선택
            def negative_ei(x):
                return -expected_improvement(x, gp, best_y)
            
            # 여러 시작점에서 최적화 시도
            next_points = []
            next_values = []
            
            for _ in range(10):
                x0 = np.random.uniform(param_mins, param_maxs, size=num_dimensions)
                bounds = [(param_mins[i], param_maxs[i]) for i in range(num_dimensions)]
                
                result = minimize(negative_ei, x0, bounds=bounds, method='L-BFGS-B')
                next_points.append(result.x)
                next_values.append(result.fun)
            
            # 가장 좋은 포인트 선택
            best_min_idx = np.argmin(next_values)
            next_sample = next_points[best_min_idx]
            
            # 새 포인트 평가
            current_params = {
                param_names[j]: next_sample[j] for j in range(num_dimensions)
            }
            
            score = objective_function(**current_params)
            
            # 관측 데이터 업데이트
            X_observed = np.vstack((X_observed, next_sample))
            y_observed = np.append(y_observed, score)
            
            # 결과 기록
            self.optimization_history.append({
                'parameters': current_params.copy(),
                'score': score,
                'iteration': iteration
            })
        
        # 최적 매개변수 구성
        best_idx = np.argmax(y_observed)
        best_params = {
            param_names[j]: X_observed[best_idx, j] for j in range(num_dimensions)
        }
        
        # 최적 매개변수 및 점수 반환
        return {
            'parameters': best_params,
            'score': y_observed[best_idx]
        }
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        최적화 히스토리 반환
        
        반환값:
            List[Dict[str, Any]]: 최적화 히스토리
        """
        return self.optimization_history
