import os
import glob
import logging
from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf
from core.config import settings

logger = logging.getLogger(__name__)

# Suppress noisy TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def build_evaluator_model(embedding_dim: int = 384, hidden_dim: int = 128) -> tf.keras.Model:
    """
    Builds the Multi-Task Deep Learning Evaluator Model architecture
    matching the trained UltraFeedback weights.
    """
    inputs = tf.keras.Input(shape=(embedding_dim,), name="prompt_emb")

    x = tf.keras.layers.Dense(
        hidden_dim,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(inputs)
    x = tf.keras.layers.Dropout(0.30)(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.Dropout(0.20)(x)

    outputs = {
        "helpfulness": tf.keras.layers.Dense(1, name="helpfulness")(x),
        "honesty": tf.keras.layers.Dense(1, name="honesty")(x),
        "instruction_following": tf.keras.layers.Dense(1, name="instruction_following")(x),
        "truthfulness": tf.keras.layers.Dense(1, name="truthfulness")(x),
        "fine_grained_score": tf.keras.layers.Dense(1, name="fine_grained_score")(x),
        "overall_score": tf.keras.layers.Dense(1, name="overall_score")(x),
        "log_tokens": tf.keras.layers.Dense(1, name="log_tokens")(x),
    }

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


class DeepLearningModelRegistry:
    """
    Manages loading and running inference for all 15 trained LLM evaluator models.
    """
    _instance: Optional["DeepLearningModelRegistry"] = None

    def __init__(self, weights_dir: Optional[str] = None):
        self.weights_dir = weights_dir or settings.EVALUATOR_WEIGHTS_DIR
        self.models: Dict[str, tf.keras.Model] = {}
        self.is_loaded = False

    @classmethod
    def get_instance(cls) -> "DeepLearningModelRegistry":
        if cls._instance is None:
            cls._instance = DeepLearningModelRegistry()
        return cls._instance

    def load_all_weights(self) -> Dict[str, str]:
        """
        Discovers all .weights.h5 files and initializes evaluator models in memory.
        """
        if not os.path.exists(self.weights_dir):
            logger.warning(f"Weights directory not found at: {self.weights_dir}")
            return {}

        weight_files = glob.glob(os.path.join(self.weights_dir, "*.weights.h5"))
        logger.info(f"Discovered {len(weight_files)} weight files in {self.weights_dir}")

        for filepath in weight_files:
            model_key = os.path.basename(filepath).replace(".weights.h5", "")
            try:
                model = build_evaluator_model(
                    embedding_dim=settings.EMBEDDING_DIM,
                    hidden_dim=settings.HIDDEN_DIM
                )
                model.load_weights(filepath)
                self.models[model_key] = model
                logger.info(f"Successfully loaded evaluator model for '{model_key}'")
            except Exception as e:
                logger.error(f"Failed to load weights for {model_key} from {filepath}: {e}")

        self.is_loaded = True
        return {k: f"loaded ({len(self.models)} total)" for k in self.models.keys()}

    def predict_all(self, prompt_embedding: np.ndarray) -> List[Dict]:
        """
        Runs multi-head prediction across all loaded evaluator models for a prompt embedding.
        
        Args:
            prompt_embedding: 1D or 2D numpy array of shape (384,) or (1, 384)
            
        Returns:
            List of score dicts containing metrics for each model.
        """
        if not self.is_loaded or not self.models:
            self.load_all_weights()

        if prompt_embedding.ndim == 1:
            prompt_embedding = np.expand_dims(prompt_embedding, axis=0)

        prompt_tensor = tf.constant(prompt_embedding, dtype=tf.float32)

        results = []
        for model_name, model in self.models.items():
            preds = model.predict(prompt_tensor, verbose=0)
            
            helpfulness = float(preds["helpfulness"][0][0])
            honesty = float(preds["honesty"][0][0])
            instruction_following = float(preds["instruction_following"][0][0])
            truthfulness = float(preds["truthfulness"][0][0])
            fine_grained = float(preds["fine_grained_score"][0][0])
            overall = float(preds["overall_score"][0][0])
            log_tokens = float(preds["log_tokens"][0][0])
            predicted_tokens = float(np.expm1(log_tokens))

            results.append({
                "model_name": model_name,
                "overall_score": round(overall, 4),
                "fine_grained_score": round(fine_grained, 4),
                "helpfulness": round(helpfulness, 4),
                "honesty": round(honesty, 4),
                "instruction_following": round(instruction_following, 4),
                "truthfulness": round(truthfulness, 4),
                "log_tokens": round(log_tokens, 4),
                "predicted_tokens": round(max(0.0, predicted_tokens), 1)
            })

        return results


def get_dl_registry() -> DeepLearningModelRegistry:
    return DeepLearningModelRegistry.get_instance()
