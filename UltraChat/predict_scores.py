import os
import glob
import argparse
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_evaluator_model(embedding_dim=384, hidden_dim=128):
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
        "log_tokens": tf.keras.layers.Dense(1, name="log_tokens")(x)
    }
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def main():
    parser = argparse.ArgumentParser(description="Predict model scores for a given prompt.")
    parser.add_argument("prompt", type=str, help="The prompt to evaluate.")
    args = parser.parse_args()

    prompt = args.prompt
    print(f"Loading SentenceTransformer to embed prompt...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device='cpu')
    
    prompt_emb = embedder.encode([prompt], convert_to_numpy=True, normalize_embeddings=True)
    prompt_emb = tf.constant(prompt_emb, dtype=tf.float32)
    
    weights_dir = "evaluator_weights"
    weight_files = glob.glob(os.path.join(weights_dir, "*.weights.h5"))
    
    if not weight_files:
        print(f"No weight files found in {weights_dir}/")
        return

    evaluator = build_evaluator_model()
    
    print("\n" + "="*50)
    print(f"Prompt: {prompt}")
    print("="*50 + "\n")
    
    for weight_file in weight_files:
        model_name = os.path.basename(weight_file).replace(".weights.h5", "")
        
        evaluator.load_weights(weight_file)
        
        preds = evaluator.predict(prompt_emb, verbose=0)
        
        helpfulness = preds["helpfulness"][0][0]
        honesty = preds["honesty"][0][0]
        instruction_following = preds["instruction_following"][0][0]
        truthfulness = preds["truthfulness"][0][0]
        fine_grained = preds["fine_grained_score"][0][0]
        overall = preds["overall_score"][0][0]
        log_tokens = preds["log_tokens"][0][0]
        tokens = np.expm1(log_tokens) 
        
        print(f"--- Model: {model_name} ---")
        print(f"  Overall Score      : {overall:.4f}")
        print(f"  Fine-Grained Score : {fine_grained:.4f}")
        print(f"  Helpfulness        : {helpfulness:.4f}")
        print(f"  Honesty            : {honesty:.4f}")
        print(f"  Instruction Follow : {instruction_following:.4f}")
        print(f"  Truthfulness       : {truthfulness:.4f}")
        print(f"  Predicted Tokens   : {tokens:.1f}")
        print("-" * 30 + "\n")

if __name__ == "__main__":
    main()
