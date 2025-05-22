"""모델 훈련 및 평가 스크립트

이 스크립트는 신경 신호 데이터에 대한 다양한 모델을 훈련하고 평가합니다.
지도 학습 모델(LSTM, CNN-LSTM, Transformer), 비지도 학습 모델(VAE, K-means, PCA),
강화학습 모델(PPO, DQN, DDPG)을 포함합니다.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           mean_absolute_error, mean_squared_error)
from typing import List, Tuple, Dict, Any, Optional, Union
import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                                       ReduceLROnPlateau, TensorBoard)

# 커스텀 모델 임포트
from models.lstm_model import LSTMModel, BidirectionalLSTMModel, EncoderDecoderLSTM
from models.cnn_lstm_model import CNNLSTM, ParallelCNNLSTM, HierarchicalCNNLSTM
from models.transformer_model import TransformerModel, ConvTransformerModel
from models.unsupervised_models import (VariationalAutoencoder, TimeSeriesKMeans, 
                                      TimeSeriesPCA, ConvolutionalVAE)
from models.ppo_model import PPOAgent
from models.dqn_agent import DQNAgent  # 이미 존재하는 DQN 에이전트 클래스

# 결과 디렉토리 설정
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 모델 저장 디렉토리 설정
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_neural_data(data_path: str) -> Dict[str, np.ndarray]:
    """
    신경 신호 데이터 로드
    
    매개변수:
        data_path (str): 데이터 디렉토리 경로
        
    반환값:
        Dict[str, np.ndarray]: 로드된 데이터
    """
    # 예시: 실제 구현에서는 실제 데이터를 로드하도록 수정
    print(f"Loading neural data from {data_path}...")
    
    # 가상 데이터 생성 (실제 데이터로 대체 필요)
    sequence_length = 100
    feature_dim = 16
    n_samples = 1000
    n_classes = 3
    
    X = np.random.randn(n_samples, sequence_length, feature_dim)
    y = np.random.randint(0, n_classes, size=n_samples)
    
    return {
        "signals": X,
        "labels": y,
        "metadata": {
            "sequence_length": sequence_length,
            "feature_dim": feature_dim,
            "n_samples": n_samples,
            "n_classes": n_classes
        }
    }


def preprocess_neural_signals(signals: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    신경 신호 전처리
    
    매개변수:
        signals (np.ndarray): 원본 신경 신호 데이터
        labels (np.ndarray): 라벨 데이터
        
    반환값:
        Tuple[np.ndarray, np.ndarray, List[str]]: (전처리된 데이터, 라벨, 특성 이름 목록)
    """
    print("Preprocessing neural signals...")
    
    # 실제 구현에서는 필터링, 정규화 등의 전처리 수행
    n_samples, sequence_length, feature_dim = signals.shape
    
    # Z-점수 정규화 (각 특성 별로)
    normalized_signals = np.zeros_like(signals)
    for i in range(feature_dim):
        mean = np.mean(signals[:, :, i])
        std = np.std(signals[:, :, i])
        normalized_signals[:, :, i] = (signals[:, :, i] - mean) / (std + 1e-8)
    
    # 특성 이름 생성
    feature_names = [f"feature_{i}" for i in range(feature_dim)]
    
    return normalized_signals, labels, feature_names


def train_supervised_models(X: np.ndarray, y: np.ndarray, model_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    지도 학습 모델 훈련
    
    매개변수:
        X (np.ndarray): 특성 데이터
        y (np.ndarray): 라벨 데이터
        model_params (Dict[str, Any]): 모델 매개변수
        
    반환값:
        Dict[str, Any]: 훈련된 모델 및 성능 지표
    """
    print("Training supervised learning models...")
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 콜백 설정
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # 결과 저장
    results = {}
    
    # 1. LSTM 모델
    print("Training LSTM model...")
    lstm_model = LSTMModel(
        sequence_length=model_params["sequence_length"],
        feature_dim=model_params["feature_dim"],
        output_dim=model_params["n_classes"],
        lstm_units=64,
        dropout_rate=0.3
    )
    
    # LSTM 모델 훈련
    lstm_history = lstm_model.train(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    # LSTM 모델 평가
    lstm_loss, lstm_mae = lstm_model.evaluate(X_test, y_test)
    lstm_preds = lstm_model.predict(X_test)
    lstm_preds_class = np.argmax(lstm_preds, axis=1) if lstm_preds.shape[1] > 1 else (lstm_preds > 0.5).astype(int)
    
    # LSTM 결과 저장
    lstm_model.save(os.path.join(MODELS_DIR, "lstm_model.h5"))
    results["lstm"] = {
        "model": lstm_model,
        "history": lstm_history,
        "loss": lstm_loss,
        "mae": lstm_mae,
        "predictions": lstm_preds,
        "metrics": {
            "accuracy": accuracy_score(y_test, lstm_preds_class),
            "precision": precision_score(y_test, lstm_preds_class, average='weighted'),
            "recall": recall_score(y_test, lstm_preds_class, average='weighted'),
            "f1": f1_score(y_test, lstm_preds_class, average='weighted')
        }
    }
    
    # 2. CNN-LSTM 모델
    print("Training CNN-LSTM model...")
    cnn_lstm_model = CNNLSTM(
        sequence_length=model_params["sequence_length"],
        feature_dim=model_params["feature_dim"],
        output_dim=model_params["n_classes"],
        cnn_filters=[32, 64],
        kernel_sizes=[3, 3],
        pool_sizes=[2, 2],
        lstm_units=[64, 32],
        dropout_rate=0.3
    )
    
    # CNN-LSTM 모델 훈련
    cnn_lstm_history = cnn_lstm_model.train(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    # CNN-LSTM 모델 평가
    cnn_lstm_loss, cnn_lstm_mae = cnn_lstm_model.evaluate(X_test, y_test)
    cnn_lstm_preds = cnn_lstm_model.predict(X_test)
    cnn_lstm_preds_class = np.argmax(cnn_lstm_preds, axis=1) if cnn_lstm_preds.shape[1] > 1 else (cnn_lstm_preds > 0.5).astype(int)
    
    # CNN-LSTM 결과 저장
    cnn_lstm_model.save(os.path.join(MODELS_DIR, "cnn_lstm_model.h5"))
    results["cnn_lstm"] = {
        "model": cnn_lstm_model,
        "history": cnn_lstm_history,
        "loss": cnn_lstm_loss,
        "mae": cnn_lstm_mae,
        "predictions": cnn_lstm_preds,
        "metrics": {
            "accuracy": accuracy_score(y_test, cnn_lstm_preds_class),
            "precision": precision_score(y_test, cnn_lstm_preds_class, average='weighted'),
            "recall": recall_score(y_test, cnn_lstm_preds_class, average='weighted'),
            "f1": f1_score(y_test, cnn_lstm_preds_class, average='weighted')
        }
    }
    
    # 3. Transformer 모델
    print("Training Transformer model...")
    transformer_model = TransformerModel(
        sequence_length=model_params["sequence_length"],
        feature_dim=model_params["feature_dim"],
        output_dim=model_params["n_classes"],
        num_transformer_blocks=2,
        num_heads=4,
        embed_dim=64,
        ff_dim=128,
        dropout_rate=0.1
    )
    
    # Transformer 모델 훈련
    transformer_history = transformer_model.train(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    # Transformer 모델 평가
    transformer_loss, transformer_mae = transformer_model.evaluate(X_test, y_test)
    transformer_preds = transformer_model.predict(X_test)
    transformer_preds_class = np.argmax(transformer_preds, axis=1) if transformer_preds.shape[1] > 1 else (transformer_preds > 0.5).astype(int)
    
    # Transformer 결과 저장
    transformer_model.save(os.path.join(MODELS_DIR, "transformer_model.h5"))
    results["transformer"] = {
        "model": transformer_model,
        "history": transformer_history,
        "loss": transformer_loss,
        "mae": transformer_mae,
        "predictions": transformer_preds,
        "metrics": {
            "accuracy": accuracy_score(y_test, transformer_preds_class),
            "precision": precision_score(y_test, transformer_preds_class, average='weighted'),
            "recall": recall_score(y_test, transformer_preds_class, average='weighted'),
            "f1": f1_score(y_test, transformer_preds_class, average='weighted')
        }
    }
    
    return results


def train_unsupervised_models(X: np.ndarray, model_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    비지도 학습 모델 훈련
    
    매개변수:
        X (np.ndarray): 특성 데이터
        model_params (Dict[str, Any]): 모델 매개변수
        
    반환값:
        Dict[str, Any]: 훈련된 모델 및 성능 지표
    """
    print("Training unsupervised learning models...")
    
    # 데이터 분할
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # 결과 저장
    results = {}
    
    # 1. VAE 모델
    print("Training VAE model...")
    vae_model = VariationalAutoencoder(
        sequence_length=model_params["sequence_length"],
        feature_dim=model_params["feature_dim"],
        latent_dim=16,
        encoder_units=[128, 64],
        decoder_units=[64, 128],
        dropout_rate=0.2
    )
    
    # VAE 모델 훈련
    vae_history = vae_model.train(
        X_train,
        epochs=50,
        batch_size=32,
        validation_data=X_test
    )
    
    # VAE 모델 평가 (재구성 오차)
    X_test_reconstructed = vae_model.reconstruct(X_test)
    reconstruction_error = np.mean(np.square(X_test - X_test_reconstructed))
    
    # 잠재 공간 인코딩
    latent_vectors = vae_model.encode(X_test)
    
    # VAE 결과 저장
    vae_model.save(
        os.path.join(MODELS_DIR, "vae_encoder.h5"),
        os.path.join(MODELS_DIR, "vae_decoder.h5"),
        os.path.join(MODELS_DIR, "vae_model.h5")
    )
    results["vae"] = {
        "model": vae_model,
        "history": vae_history,
        "latent_vectors": latent_vectors,
        "reconstruction_error": reconstruction_error
    }
    
    # 2. K-means 클러스터링
    print("Training K-means model...")
    k_means_model = TimeSeriesKMeans(
        n_clusters=model_params["n_classes"],
        flatten=True
    )
    
    # K-means 모델 훈련
    k_means_labels = k_means_model.fit_predict(X)
    
    # K-means 모델 평가 (실루엣 점수)
    kmeans_metrics = k_means_model.evaluate(X)
    
    # K-means 결과 저장
    results["kmeans"] = {
        "model": k_means_model,
        "labels": k_means_labels,
        "metrics": kmeans_metrics
    }
    
    # 3. PCA
    print("Applying PCA...")
    pca_model = TimeSeriesPCA(
        n_components=min(10, model_params["feature_dim"]),
        flatten=True
    )
    
    # PCA 적용
    X_pca = pca_model.fit_transform(X)
    
    # PCA 평가 (설명된 분산)
    variance_info = pca_model.explained_variance()
    
    # PCA 결과 저장
    results["pca"] = {
        "model": pca_model,
        "transformed_data": X_pca,
        "variance_info": variance_info
    }
    
    return results


def train_reinforcement_learning_models(env, model_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    강화학습 모델 훈련
    
    매개변수:
        env: 환경 인스턴스
        model_params (Dict[str, Any]): 모델 매개변수
        
    반환값:
        Dict[str, Any]: 훈련된 모델 및 성능 지표
    """
    print("Training reinforcement learning models...")
    
    # 환경 정보
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 행동 범위
    action_bounds = {
        param_name: (env.action_space.low[i], env.action_space.high[i])
        for i, param_name in enumerate(model_params["action_names"])
    }
    
    # 결과 저장
    results = {}
    
    # 1. PPO 모델
    print("Training PPO model...")
    ppo_agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bounds=action_bounds,
        buffer_size=2048,
        batch_size=64,
        gamma=0.99,
        lam=0.95,
        actor_hidden_sizes=[64, 64],
        critic_hidden_sizes=[64, 64],
        learning_rate=3e-4,
        clip_ratio=0.2,
        target_kl=0.01,
        value_coef=0.5,
        entropy_coef=0.01,
        train_iters=80,
        log_dir=os.path.join(RESULTS_DIR, "ppo_logs")
    )
    
    # PPO 모델 훈련
    ppo_metrics = ppo_agent.train(
        env=env,
        epochs=model_params["epochs"],
        steps_per_epoch=2048,
        max_ep_len=model_params["max_ep_len"],
        save_freq=5,
        log_freq=1,
        render=False
    )
    
    # PPO 결과 저장
    ppo_agent.save_model(os.path.join(MODELS_DIR, "ppo_model"))
    results["ppo"] = {
        "agent": ppo_agent,
        "metrics": ppo_metrics
    }
    
    # 2. DQN 모델 (이산 액션 공간에 적합)
    # 실제 구현에서는 DQN에 적합한 환경 설정 필요
    if hasattr(env.action_space, 'n'):  # 이산 액션 공간
        print("Training DQN model...")
        dqn_agent = DQNAgent(
            state_dim=state_dim,
            action_dim=env.action_space.n,
            hidden_sizes=[64, 64],
            learning_rate=1e-3,
            gamma=0.99,
            buffer_size=100000,
            batch_size=64,
            target_update_freq=1000,
            eps_start=1.0,
            eps_end=0.01,
            eps_decay=0.995,
            log_dir=os.path.join(RESULTS_DIR, "dqn_logs")
        )
        
        # DQN 모델 훈련
        dqn_metrics = dqn_agent.train(
            env=env,
            max_episodes=model_params["max_episodes"],
            max_steps=model_params["max_ep_len"],
            save_freq=10,
            log_freq=1,
            render=False
        )
        
        # DQN 결과 저장
        dqn_agent.save_model(os.path.join(MODELS_DIR, "dqn_model"))
        results["dqn"] = {
            "agent": dqn_agent,
            "metrics": dqn_metrics
        }
    
    # 3. DDPG 모델 - 이 구현은 코드/ai_models/reinforcement_learning.py에 있음
    # 실제 코드에서는 해당 모듈을 임포트하여 사용
    print("DDPG model is implemented in code/ai_models/reinforcement_learning.py")
    
    return results


def cross_validate_models(X: np.ndarray, y: np.ndarray, model_params: Dict[str, Any], n_folds: int = 10) -> Dict[str, Any]:
    """
    교차 검증 수행
    
    매개변수:
        X (np.ndarray): 특성 데이터
        y (np.ndarray): 라벨 데이터
        model_params (Dict[str, Any]): 모델 매개변수
        n_folds (int): 폴드 수
        
    반환값:
        Dict[str, Any]: 교차 검증 결과
    """
    print(f"Performing {n_folds}-fold cross-validation...")
    
    # K-폴드 교차 검증
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 모델 타입 정의
    model_types = [
        ("lstm", LSTMModel),
        ("cnn_lstm", CNNLSTM),
        ("transformer", TransformerModel)
    ]
    
    # 결과 저장
    cv_results = {model_name: {"accuracy": [], "precision": [], "recall": [], "f1": [], "mae": [], "rmse": []} 
                 for model_name, _ in model_types}
    
    # 폴드 반복
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{n_folds}")
        
        # 데이터 분할
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 콜백 설정
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True)
        ]
        
        # 각 모델 타입에 대해
        for model_name, ModelClass in model_types:
            print(f"  Training {model_name} model...")
            
            # 모델 초기화
            if model_name == "lstm":
                model = ModelClass(
                    sequence_length=model_params["sequence_length"],
                    feature_dim=model_params["feature_dim"],
                    output_dim=model_params["n_classes"],
                    lstm_units=64,
                    dropout_rate=0.3
                )
            elif model_name == "cnn_lstm":
                model = ModelClass(
                    sequence_length=model_params["sequence_length"],
                    feature_dim=model_params["feature_dim"],
                    output_dim=model_params["n_classes"],
                    cnn_filters=[32, 64],
                    kernel_sizes=[3, 3],
                    pool_sizes=[2, 2],
                    lstm_units=[64, 32],
                    dropout_rate=0.3
                )
            elif model_name == "transformer":
                model = ModelClass(
                    sequence_length=model_params["sequence_length"],
                    feature_dim=model_params["feature_dim"],
                    output_dim=model_params["n_classes"],
                    num_transformer_blocks=2,
                    num_heads=4,
                    embed_dim=64,
                    ff_dim=128,
                    dropout_rate=0.1
                )
            
            # 모델 훈련
            model.train(
                X_train, y_train,
                epochs=30,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=callbacks
            )
            
            # 모델 평가
            loss, mae = model.evaluate(X_test, y_test)
            preds = model.predict(X_test)
            preds_class = np.argmax(preds, axis=1) if preds.shape[1] > 1 else (preds > 0.5).astype(int)
            
            # 메트릭 계산
            accuracy = accuracy_score(y_test, preds_class)
            precision = precision_score(y_test, preds_class, average='weighted')
            recall = recall_score(y_test, preds_class, average='weighted')
            f1 = f1_score(y_test, preds_class, average='weighted')
            rmse = np.sqrt(mean_squared_error(y_test, preds_class))
            
            # 결과 저장
            cv_results[model_name]["accuracy"].append(accuracy)
            cv_results[model_name]["precision"].append(precision)
            cv_results[model_name]["recall"].append(recall)
            cv_results[model_name]["f1"].append(f1)
            cv_results[model_name]["mae"].append(mae)
            cv_results[model_name]["rmse"].append(rmse)
            
            print(f"    Fold {fold + 1} metrics: accuracy={accuracy:.4f}, f1={f1:.4f}, mae={mae:.4f}")
    
    # 평균 메트릭 계산
    for model_name in cv_results:
        for metric in cv_results[model_name]:
            cv_results[model_name][metric] = {
                "mean": np.mean(cv_results[model_name][metric]),
                "std": np.std(cv_results[model_name][metric]),
                "values": cv_results[model_name][metric]
            }
    
    return cv_results


def plot_results(supervised_results: Dict[str, Any], unsupervised_results: Dict[str, Any], cv_results: Dict[str, Any]):
    """
    결과 시각화
    
    매개변수:
        supervised_results (Dict[str, Any]): 지도 학습 결과
        unsupervised_results (Dict[str, Any]): 비지도 학습 결과
        cv_results (Dict[str, Any]): 교차 검증 결과
    """
    print("Plotting results...")
    
    # 1. 지도 학습 모델 성능 비교
    plt.figure(figsize=(12, 8))
    
    # 정확도, F1 점수
    metrics = ["accuracy", "f1"]
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        values = [supervised_results[model_name]["metrics"][metric] for model_name in supervised_results]
        plt.bar(supervised_results.keys(), values)
        plt.title(f"{metric.capitalize()} Score")
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
    
    # 손실, MAE
    metrics = ["loss", "mae"]
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+3)
        values = [supervised_results[model_name][metric] for model_name in supervised_results]
        plt.bar(supervised_results.keys(), values)
        plt.title(f"{metric.upper()}")
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "supervised_model_comparison.png"))
    
    # 2. 교차 검증 결과
    plt.figure(figsize=(15, 10))
    
    metrics = ["accuracy", "f1", "mae", "rmse"]
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        # 모델별 평균 및 표준편차
        models = list(cv_results.keys())
        means = [cv_results[model][metric]["mean"] for model in models]
        stds = [cv_results[model][metric]["std"] for model in models]
        
        # 바 그래프
        plt.bar(models, means, yerr=stds, capsize=10, alpha=0.7)
        plt.title(f"Cross-Validation {metric.upper()}")
        plt.grid(alpha=0.3)
        
        # y축 범위 설정
        if metric in ["accuracy", "precision", "recall", "f1"]:
            plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cross_validation_results.png"))
    
    # 3. 비지도 학습 결과
    
    # PCA 설명된 분산
    plt.figure(figsize=(10, 6))
    variance_ratio = unsupervised_results["pca"]["variance_info"]["explained_variance_ratio"]
    cumsum = np.cumsum(variance_ratio)
    
    plt.bar(range(1, len(variance_ratio) + 1), variance_ratio, alpha=0.7, label="Explained Variance")
    plt.step(range(1, len(cumsum) + 1), cumsum, where="mid", label="Cumulative Explained Variance", color="red")
    plt.axhline(y=0.95, color="green", linestyle="--", label="95% Explained Variance")
    
    plt.title("PCA Explained Variance")
    plt.xlabel("Principal Components")
    plt.ylabel("Explained Variance Ratio")
    plt.xticks(range(1, len(variance_ratio) + 1))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "pca_explained_variance.png"))
    
    # K-means 클러스터링 결과 (PCA로 차원 축소하여 시각화)
    plt.figure(figsize=(10, 8))
    X_pca = unsupervised_results["pca"]["transformed_data"]
    kmeans_labels = unsupervised_results["kmeans"]["labels"]
    
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="viridis", alpha=0.7)
    plt.title("K-means Clustering (Visualized with PCA)")
    plt.xlabel(f"PC1 ({variance_ratio[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({variance_ratio[1]:.2%} variance)")
    plt.colorbar(label="Cluster")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "kmeans_clustering.png"))
    
    # VAE 잠재 공간 시각화
    plt.figure(figsize=(10, 8))
    latent_vectors = unsupervised_results["vae"]["latent_vectors"]
    
    # 잠재 공간이 2차원 이상인 경우, 처음 2개 차원만 시각화
    plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], alpha=0.7)
    plt.title("VAE Latent Space")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "vae_latent_space.png"))


def simulate_reinforcement_learning_env():
    """
    강화학습 환경 시뮬레이션
    
    반환값:
        환경 인스턴스
    """
    # 실제 구현에서는 신경 자극 시뮬레이션 환경을 사용
    # 예시로 OpenAI Gym의 Pendulum 환경 사용
    try:
        import gym
        env = gym.make('Pendulum-v1')
        return env
    except:
        print("OpenAI Gym 환경을 로드할 수 없습니다. 환경 시뮬레이션을 건너뜁니다.")
        return None


def save_results_to_csv(supervised_results: Dict[str, Any], unsupervised_results: Dict[str, Any], 
                      cv_results: Dict[str, Any], filepath: str):
    """
    결과를 CSV 파일로 저장
    
    매개변수:
        supervised_results (Dict[str, Any]): 지도 학습 결과
        unsupervised_results (Dict[str, Any]): 비지도 학습 결과
        cv_results (Dict[str, Any]): 교차 검증 결과
        filepath (str): 저장할 파일 경로
    """
    # 지도 학습 모델 메트릭
    supervised_metrics = []
    for model_name, result in supervised_results.items():
        metrics = result["metrics"]
        supervised_metrics.append({
            "Model Type": "Supervised",
            "Model Name": model_name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1"],
            "MAE": result["mae"],
            "Loss": result["loss"]
        })
    
    # 교차 검증 결과
    cv_metrics = []
    for model_name, result in cv_results.items():
        cv_metrics.append({
            "Model Type": "Cross-Validation",
            "Model Name": model_name,
            "Accuracy": result["accuracy"]["mean"],
            "Accuracy Std": result["accuracy"]["std"],
            "F1 Score": result["f1"]["mean"],
            "F1 Score Std": result["f1"]["std"],
            "MAE": result["mae"]["mean"],
            "MAE Std": result["mae"]["std"],
            "RMSE": result["rmse"]["mean"],
            "RMSE Std": result["rmse"]["std"]
        })
    
    # 비지도 학습 결과
    unsupervised_metrics = []
    
    # VAE 재구성 오차
    unsupervised_metrics.append({
        "Model Type": "Unsupervised",
        "Model Name": "VAE",
        "Reconstruction Error": unsupervised_results["vae"]["reconstruction_error"]
    })
    
    # K-means 클러스터링 지표
    kmeans_metrics = unsupervised_results["kmeans"]["metrics"]
    unsupervised_metrics.append({
        "Model Type": "Unsupervised",
        "Model Name": "K-means",
        "Inertia": kmeans_metrics["inertia"],
        "Silhouette Score": kmeans_metrics["silhouette_score"]
    })
    
    # PCA 설명된 분산
    variance_info = unsupervised_results["pca"]["variance_info"]
    unsupervised_metrics.append({
        "Model Type": "Unsupervised",
        "Model Name": "PCA",
        "Total Explained Variance": np.sum(variance_info["explained_variance_ratio"]),
        "Components for 95% Variance": np.argmax(np.cumsum(variance_info["explained_variance_ratio"]) >= 0.95) + 1
    })
    
    # 모든 메트릭 결합
    all_metrics = supervised_metrics + cv_metrics + unsupervised_metrics
    
    # DataFrame 생성 및 저장
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")


def main():
    """메인 함수"""
    print("Starting model training and evaluation...")
    
    # 1. 데이터 로드 및 전처리
    data = load_neural_data("data/neural_recordings")
    X, y, feature_names = preprocess_neural_signals(data["signals"], data["labels"])
    
    # 모델 매개변수 설정
    model_params = {
        "sequence_length": data["metadata"]["sequence_length"],
        "feature_dim": data["metadata"]["feature_dim"],
        "n_samples": data["metadata"]["n_samples"],
        "n_classes": data["metadata"]["n_classes"],
        "epochs": 50,
        "max_ep_len": 200,
        "max_episodes": 500,
        "action_names": ["frequency", "amplitude", "pulse_width", "duration", "waveform"]
    }
    
    # 2. 지도 학습 모델 훈련
    supervised_results = train_supervised_models(X, y, model_params)
    
    # 3. 비지도 학습 모델 훈련
    unsupervised_results = train_unsupervised_models(X, model_params)
    
    # 4. 교차 검증
    cv_results = cross_validate_models(X, y, model_params)
    
    # 5. 강화학습 모델 훈련 (환경 시뮬레이션 가능한 경우)
    env = simulate_reinforcement_learning_env()
    if env is not None:
        rl_results = train_reinforcement_learning_models(env, model_params)
        env.close()
    
    # 6. 결과 시각화
    plot_results(supervised_results, unsupervised_results, cv_results)
    
    # 7. 결과 저장
    save_results_to_csv(
        supervised_results, 
        unsupervised_results, 
        cv_results,
        os.path.join(RESULTS_DIR, "model_metrics.csv")
    )
    
    print("Model training and evaluation completed successfully!")
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()