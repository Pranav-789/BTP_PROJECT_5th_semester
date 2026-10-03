import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot.subplots
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    """Loads features and targets, handles missing values, and splits data."""
    print("Loading datasets...")
    # 1. Load Data
    X = np.load('data/processed/X_embeddings.npy')
    df = pd.read_csv('data/processed/clean_router_data.csv')
    
    candidates = [
        "claude_instant", "claude_v1", "claude_v2", "code_llama_34b", 
        "gpt_35_turbo", "gpt_4", "llama_2_70b", "mistral_7b", 
        "mixtral_8x7b", "wizardlm_13b", "yi_34b"
    ]
    
    score_cols = [f"{c}_score" for c in candidates]
    token_cols = [f"{c}_tokens" for c in candidates]
    
    Y_scores = df[score_cols].values
    Y_tokens = df[token_cols].values
    
    # 2. Data Cleaning: Drop rows with NaN values in scores or tokens
    # Find rows where there is at least one NaN in scores or tokens
    nan_mask = np.isnan(Y_scores).any(axis=1) | np.isnan(Y_tokens).any(axis=1)
    
    print(f"Original data shape: {X.shape}")
    print(f"Found {nan_mask.sum()} rows with NaN values. Filtering them out...")
    
    X_clean = X[~nan_mask]
    Y_scores_clean = Y_scores[~nan_mask]
    Y_tokens_clean = Y_tokens[~nan_mask]
    
    # 3. Token Target Transformation
    Y_tokens_log = np.log1p(Y_tokens_clean)
    
    # 4. Data Splitting
    X_train, X_val, Y_scores_train, Y_scores_val, Y_tokens_train, Y_tokens_val = train_test_split(
        X_clean, Y_scores_clean, Y_tokens_log, test_size=0.2, random_state=42
    )
    
    print(f"Train set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    return X_train, X_val, Y_scores_train, Y_scores_val, Y_tokens_train, Y_tokens_val

def build_mlp_base():
    """Builds the shared base architecture for both models."""
    inputs = keras.Input(shape=(384,))
    x = layers.Dense(256, activation='gelu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(128, activation='gelu')(x)
    return inputs, x

def build_quality_model():
    """Builds the Multi-Target Regression Model for Quality Scores."""
    inputs, x = build_mlp_base()
    outputs = layers.Dense(11, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="quality_model")
    optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def build_token_model():
    """Builds the Multi-Target Regression Model for Log-Transformed Token Counts."""
    inputs, x = build_mlp_base()
    outputs = layers.Dense(11, activation='relu')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="token_model")
    optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def plot_training_history(history_quality, history_token):
    """Plots training and validation metrics in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Training History', fontsize=16)
    
    # Quality Model Loss
    axes[0, 0].plot(history_quality.history['loss'], label='Train Loss')
    axes[0, 0].plot(history_quality.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Quality Model - Loss (MSE)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Quality Model MAE
    axes[0, 1].plot(history_quality.history['mae'], label='Train MAE', color='orange')
    axes[0, 1].plot(history_quality.history['val_mae'], label='Val MAE', color='red')
    axes[0, 1].set_title('Quality Model - MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Token Model Loss
    axes[1, 0].plot(history_token.history['loss'], label='Train Loss')
    axes[1, 0].plot(history_token.history['val_loss'], label='Val Loss')
    axes[1, 0].set_title('Token Model (Log1p) - Loss (MSE)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Token Model MAE
    axes[1, 1].plot(history_token.history['mae'], label='Train MAE', color='orange')
    axes[1, 1].plot(history_token.history['val_mae'], label='Val MAE', color='red')
    axes[1, 1].set_title('Token Model (Log1p) - MAE')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig('artifacts/training_history.png')
    print("Saved training history plot to 'artifacts/training_history.png'")
    # plt.show() # Uncomment if running interactively

def main():
    X_train, X_val, Y_scores_train, Y_scores_val, Y_tokens_train, Y_tokens_val = load_and_preprocess_data()
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=8, 
        restore_best_weights=True
    )
    
    os.makedirs('artifacts', exist_ok=True)
    
    # --- Train Quality Model ---
    print("\n" + "="*50)
    print("Training Quality Model...")
    print("="*50)
    quality_model = build_quality_model()
    history_quality = quality_model.fit(
        X_train, Y_scores_train,
        validation_data=(X_val, Y_scores_val),
        epochs=40,
        batch_size=128,
        callbacks=[early_stopping],
        verbose=1
    )
    quality_model.save('artifacts/quality_model.keras')
    print("Saved quality model to 'artifacts/quality_model.keras'")
    
    # --- Train Token Model ---
    print("\n" + "="*50)
    print("Training Token Model...")
    print("="*50)
    token_model = build_token_model()
    history_token = token_model.fit(
        X_train, Y_tokens_train,
        validation_data=(X_val, Y_tokens_val),
        epochs=40,
        batch_size=128,
        callbacks=[early_stopping],
        verbose=1
    )
    token_model.save('artifacts/token_model_log.keras')
    print("Saved token model to 'artifacts/token_model_log.keras'")
    
    # --- Plot History ---
    plot_training_history(history_quality, history_token)

if __name__ == "__main__":
    main()
