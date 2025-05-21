"""
DQNAgent 클래스에 대한 단위 테스트

이 모듈은 models.dqn_agent.DQNAgent 클래스의 기능을 테스트합니다.
"""

import unittest
import numpy as np
import os
import tempfile

# 필요한 경로 추가
import sys
sys.path.append('.')

# tensorflow 임포트 실패 시 모의 객체 생성을 위한 준비
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # 모의 tf 모듈 생성
    import types
    tf = types.ModuleType('tf')
    tf.keras = types.ModuleType('tf.keras')
    tf.keras.Model = object
    tf.keras.callbacks = types.ModuleType('tf.keras.callbacks')
    tf.keras.callbacks.History = object
    sys.modules['tensorflow'] = tf
    sys.modules['tensorflow.keras'] = tf.keras

from models.dqn_agent import DQNAgent

# TensorFlow가 설치되지 않은 경우 스킵할 테스트 데코레이터
def requires_tensorflow(func):
    """TensorFlow가 필요한 테스트를 위한 데코레이터"""
    def wrapper(*args, **kwargs):
        if not TENSORFLOW_AVAILABLE:
            raise unittest.SkipTest("TensorFlow가 설치되어 있지 않습니다")
        return func(*args, **kwargs)
    return wrapper

class TestDQNAgent(unittest.TestCase):
    """
    DQNAgent 클래스의 단위 테스트
    
    이 테스트 클래스는 DQNAgent 클래스의 주요 메서드를 검증합니다:
    - 초기화 및 모델 구축
    - 타겟 모델 업데이트
    - 경험 저장 (memorize)
    - 행동 선택 (act)
    - 경험 리플레이 (replay)
    - 모델 저장 및 로드
    """
    
    @requires_tensorflow
    def setUp(self):
        """
        각 테스트 전에 실행되는 설정 메서드
        """
        self.state_size = 5
        self.action_size = 3
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            memory_size=100,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            learning_rate=0.001
        )
        
        # 테스트용 상태 및 행동 생성
        self.state = np.random.random(self.state_size)
        self.next_state = np.random.random(self.state_size)
        self.action = 1
        self.reward = 1.0
        self.done = False
    
    @requires_tensorflow
    def test_initialization(self):
        """초기화 테스트"""
        # 기본 속성 확인
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)
        self.assertEqual(self.agent.gamma, 0.95)
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertEqual(self.agent.epsilon_min, 0.01)
        self.assertEqual(self.agent.epsilon_decay, 0.995)
        self.assertEqual(self.agent.learning_rate, 0.001)
        
        # 메모리 초기화 확인
        self.assertEqual(len(self.agent.memory), 0)
        
        # 모델 초기화 확인
        self.assertIsNotNone(self.agent.model)
        self.assertIsNotNone(self.agent.target_model)
        
        # 잘못된 매개변수로 초기화 시도
        with self.assertRaises(ValueError):
            DQNAgent(state_size=0, action_size=3)
        with self.assertRaises(ValueError):
            DQNAgent(state_size=5, action_size=0)
        with self.assertRaises(ValueError):
            DQNAgent(state_size=5, action_size=3, gamma=1.5)
        with self.assertRaises(ValueError):
            DQNAgent(state_size=5, action_size=3, epsilon=2.0)
    
    @requires_tensorflow
    def test_update_target_model(self):
        """타겟 모델 업데이트 테스트"""
        # 메인 모델과 타겟 모델의 가중치가 다른지 확인 (메인 모델 학습 시뮬레이션)
        weights_before = [w.numpy() for w in self.agent.target_model.get_weights()]
        
        # 메인 모델 가중치 변경 (간단한 시뮬레이션)
        for layer in self.agent.model.layers:
            weights = layer.get_weights()
            if weights:
                noise = [np.random.normal(0, 0.1, w.shape) for w in weights]
                layer.set_weights([w + n for w, n in zip(weights, noise)])
        
        # 타겟 모델 업데이트
        self.agent.update_target_model()
        
        # 업데이트 후 가중치 비교
        weights_after = [w.numpy() for w in self.agent.target_model.get_weights()]
        
        # 가중치가 변경되었는지 확인
        for wb, wa in zip(weights_before, weights_after):
            self.assertFalse(np.array_equal(wb, wa))
        
        # 메인 모델과 타겟 모델의 가중치가 동일한지 확인
        main_weights = [w.numpy() for w in self.agent.model.get_weights()]
        target_weights = [w.numpy() for w in self.agent.target_model.get_weights()]
        
        for mw, tw in zip(main_weights, target_weights):
            self.assertTrue(np.array_equal(mw, tw))
    
    @requires_tensorflow
    def test_memorize(self):
        """경험 저장 테스트"""
        # 초기 메모리 상태 확인
        initial_memory_size = len(self.agent.memory)
        
        # 경험 저장
        self.agent.memorize(self.state, self.action, self.reward, self.next_state, self.done)
        
        # 메모리 크기 증가 확인
        self.assertEqual(len(self.agent.memory), initial_memory_size + 1)
        
        # 저장된 경험 확인
        state, action, reward, next_state, done = self.agent.memory[-1]
        self.assertTrue(np.array_equal(state, self.state))
        self.assertEqual(action, self.action)
        self.assertEqual(reward, self.reward)
        self.assertTrue(np.array_equal(next_state, self.next_state))
        self.assertEqual(done, self.done)
        
        # 유효하지 않은 행동 인덱스로 저장 시도
        with self.assertRaises(ValueError):
            self.agent.memorize(self.state, -1, self.reward, self.next_state, self.done)
        with self.assertRaises(ValueError):
            self.agent.memorize(self.state, self.action_size, self.reward, self.next_state, self.done)
    
    @requires_tensorflow
    def test_act(self):
        """행동 선택 테스트"""
        # 엡실론 = 0 (항상 최적 행동 선택)으로 설정
        self.agent.epsilon = 0
        
        # 상태 형태 변환
        state = self.state.reshape(1, -1)
        
        # Q-값 조작 (행동 1이 최적 행동이 되도록)
        action_values = np.zeros((1, self.action_size))
        action_values[0, 1] = 1.0  # 행동 1에 가장 높은 Q-값 할당
        
        # predict 메서드를 모의(Mock)하여 위에서 정의한 action_values를 반환하도록 함
        original_predict = self.agent.model.predict
        try:
            self.agent.model.predict = lambda *args, **kwargs: action_values
            
            # 행동 선택 테스트
            action = self.agent.act(state)
            self.assertEqual(action, 1)  # 가장 높은 Q-값을 가진 행동 1이 선택되어야 함
            
            # 엡실론 = 1 (항상 무작위 행동 선택)으로 설정
            self.agent.epsilon = 1.0
            actions = [self.agent.act(state) for _ in range(100)]
            
            # 모든 행동이 선택될 확률 검증 (근사적으로 각 행동의 선택 비율 확인)
            action_counts = np.zeros(self.action_size)
            for a in actions:
                action_counts[a] += 1
            
            # 각 행동이 최소 1번 이상 선택되었는지 확인
            for count in action_counts:
                self.assertGreater(count, 0)
        finally:
            # 원래 predict 메서드 복원
            self.agent.model.predict = original_predict
        
        # 잘못된 상태 형태로 행동 선택 시도
        with self.assertRaises(ValueError):
            # 형태가 (state_size + 1,)인 상태로 시도
            self.agent.act(np.random.random(self.state_size + 1))
    
    @requires_tensorflow
    def test_replay(self):
        """경험 리플레이 테스트"""
        # 메모리가 충분하지 않은 경우 (배치 크기보다 작은 경우)
        batch_size = 10
        self.agent.memory.clear()  # 메모리 비우기
        
        # 메모리에 경험 추가 (배치 크기보다 적게)
        for _ in range(batch_size - 1):
            self.agent.memorize(self.state, self.action, self.reward, self.next_state, self.done)
        
        # 메모리가 불충분한 경우 replay는 0.0을 반환해야 함
        loss = self.agent.replay(batch_size)
        self.assertEqual(loss, 0.0)
        
        # 추가 경험을 저장하여 메모리를 충분하게 만들기
        self.agent.memorize(self.state, self.action, self.reward, self.next_state, self.done)
        
        # 메모리가 충분한 경우 replay는 손실값을 반환해야 함
        # fit 메서드를 모의(Mock)하여 이력 객체를 반환하도록 함
        class MockHistory:
            def __init__(self):
                self.history = {'loss': [0.5]}
        
        original_fit = self.agent.model.fit
        try:
            self.agent.model.fit = lambda *args, **kwargs: MockHistory()
            
            # 원래 엡실론 값 저장
            original_epsilon = self.agent.epsilon
            
            # 리플레이 수행
            loss = self.agent.replay(batch_size)
            
            # 손실값이 정상적으로 반환되는지 확인
            self.assertEqual(loss, 0.5)
            
            # 엡실론이 감소했는지 확인
            self.assertLess(self.agent.epsilon, original_epsilon)
        finally:
            # 원래 fit 메서드 복원
            self.agent.model.fit = original_fit
    
    @requires_tensorflow
    def test_save_load(self):
        """모델 저장 및 로드 테스트"""
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            model_path = tmp.name
            
        try:
            # 메인 모델 가중치 변경 (간단한 시뮬레이션)
            for layer in self.agent.model.layers:
                weights = layer.get_weights()
                if weights:
                    noise = [np.random.normal(0, 0.1, w.shape) for w in weights]
                    layer.set_weights([w + n for w, n in zip(weights, noise)])
            
            # 변경된 가중치 저장
            main_weights_before = [w.numpy() for w in self.agent.model.get_weights()]
            
            # 모델 저장
            self.agent.save(model_path)
            
            # 타겟 모델 가중치 변경 (원래 저장된 가중치와 다르게)
            for layer in self.agent.model.layers:
                weights = layer.get_weights()
                if weights:
                    noise = [np.random.normal(0, 0.2, w.shape) for w in weights]
                    layer.set_weights([w + n for w, n in zip(weights, noise)])
            
            # 변경 후 가중치 확인
            main_weights_after_change = [w.numpy() for w in self.agent.model.get_weights()]
            for wb, wa in zip(main_weights_before, main_weights_after_change):
                self.assertFalse(np.array_equal(wb, wa))
            
            # 모델 로드
            self.agent.load(model_path)
            
            # 로드 후 가중치 확인
            main_weights_after_load = [w.numpy() for w in self.agent.model.get_weights()]
            for wb, wl in zip(main_weights_before, main_weights_after_load):
                self.assertTrue(np.array_equal(wb, wl))
                
            # 타겟 모델도 업데이트되었는지 확인
            target_weights = [w.numpy() for w in self.agent.target_model.get_weights()]
            for wm, wt in zip(main_weights_after_load, target_weights):
                self.assertTrue(np.array_equal(wm, wt))
        finally:
            # 임시 파일 삭제
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    @requires_tensorflow
    def test_get_q_values(self):
        """Q-값 반환 테스트"""
        # 상태 형태 변환
        state = self.state.reshape(1, -1)
        
        # Q-값 조작
        q_values = np.array([0.1, 0.5, 0.2])
        
        # predict 메서드를 모의(Mock)하여 위에서 정의한 q_values를 반환하도록 함
        original_predict = self.agent.model.predict
        try:
            self.agent.model.predict = lambda *args, **kwargs: np.array([q_values])
            
            # Q-값 얻기
            result = self.agent.get_q_values(state)
            
            # 반환된 Q-값 확인
            self.assertTrue(np.array_equal(result, q_values))
            
            # 1차원 상태 배열로도 테스트
            result = self.agent.get_q_values(self.state)
            self.assertTrue(np.array_equal(result, q_values))
        finally:
            # 원래 predict 메서드 복원
            self.agent.model.predict = original_predict
        
        # 잘못된 상태 형태로 Q-값 얻기 시도
        with self.assertRaises(ValueError):
            # 형태가 (2, state_size)인 상태로 시도
            self.agent.get_q_values(np.random.random((2, self.state_size)))

if __name__ == '__main__':
    unittest.main()
