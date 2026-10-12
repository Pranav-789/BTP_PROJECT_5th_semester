!pip install datasets

# --- CELL ---

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

# --- CELL ---

from datasets import load_dataset

print("Downloading and loading UltraFeedback...")
dataset = load_dataset("openbmb/UltraFeedback", split="train")

print("\n--- Raw Data Example ---")
print(f"Prompt: {dataset[0]['instruction'][:100]}...")
print(f"Number of models that answered: {len(dataset[0]['completions'])}")

# --- CELL ---

dataset

# --- CELL ---

def safe_float(val):
    try:
        return float(val) if val is not None else np.nan
    except (ValueError, TypeError):
        return np.nan

# --- CELL ---

import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    if not text:
        return 0
    # encode_ordinary bypasses special token parsing checks
    return len(tokenizer.encode_ordinary(str(text)))

# --- CELL ---

records = []
for row in dataset:
    prompt = row["instruction"]
    
    for completion in row["completions"]:
        model_name = completion.get("model")
        response = completion.get("response", "")
        annotations = completion.get("annotations", {})
        
        helpfulness = safe_float(annotations.get("helpfulness", {}).get("Rating"))
        honesty = safe_float(annotations.get("honesty", {}).get("Rating"))
        inst_follow = safe_float(annotations.get("instruction_following", {}).get("Rating"))
        truthfulness = safe_float(annotations.get("truthfulness", {}).get("Rating"))
        
        fine_grained = safe_float(completion.get("fine-grained_score"))
        overall = safe_float(completion.get("overall_score"))
        
        if np.isnan(overall) or np.isnan(fine_grained):
            continue
            
        token_count = count_tokens(response)
        
        records.append({
            "prompt": prompt,
            "model": model_name,
            "helpfulness": helpfulness,
            "honesty": honesty,
            "instruction_following": inst_follow,
            "truthfulness": truthfulness,
            "fine_grained_score": fine_grained,
            "overall_score": overall,
            "tokens": token_count,
            "log_tokens": np.log1p(token_count)
        })

df = pd.DataFrame(records)

sub_cols = ["helpfulness", "honesty", "instruction_following", "truthfulness"]
for col in sub_cols:
    df[col] = df[col].fillna(df[col].median())

# --- CELL ---

df.head(12)

# --- CELL ---

dct = sorted(
    df['model'].value_counts().to_dict().items(),
    key=lambda x: x[1],
    reverse=True
)

print(dct)
print(len(dct))

# --- CELL ---

df.info()

# --- CELL ---

df.describe()

# --- CELL ---

plt.plot(df['tokens'].sort_values().reset_index(drop=True))
plt.xlabel("Samples")
plt.ylabel("Output Tokens")
plt.show()

# --- CELL ---

# Drop the underrepresented and proprietary models
models_to_drop = ['bard', 'pythia-12b']
df = df[~df['model'].isin(models_to_drop)].copy()

active_models = df['model'].unique().tolist()
print(f"Remaining candidate LLMs: {len(active_models)}")

# --- CELL ---

df.head()

# --- CELL ---

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

unique_prompts = df['prompt'].unique().tolist()

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device='cpu')

print(f"Embedding {len(unique_prompts)} prompts...")
embeddings_matrix = embedder.encode(
    unique_prompts,
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True 
)

prompt_to_idx = {prompt: idx for idx, prompt in enumerate(unique_prompts)}

df['prompt_idx'] = df['prompt'].map(prompt_to_idx)

np.save("ultrafeedback_prompt_embeddings.npy", embeddings_matrix)

print("Mapping complete. DataFrame is ready for model-specific splitting.")
print(df[['prompt_idx', 'model']].head())

# --- CELL ---

df.head()

# --- CELL ---

model_counts = df['model'].value_counts()
print(model_counts)

# --- CELL ---

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


os.makedirs("evaluator_weights", exist_ok=True)
os.makedirs("training_plots", exist_ok=True)

tf.keras.utils.set_random_seed(42)

BATCH_SIZE = 128
EPOCHS = 100

DROP_MODELS = {"bard", "pythia-12b"}

active_models = [
    m for m in df["model"].unique()
    if m not in DROP_MODELS
]

print(f"Training {len(active_models)} models:")
print(active_models)

TARGET_COLS = [
    "helpfulness",
    "honesty",
    "instruction_following",
    "truthfulness",
    "fine_grained_score",
    "overall_score",
    "log_tokens"
]

tf_embeddings = tf.constant(
    embeddings_matrix,
    dtype=tf.float32
)

def build_tf_dataset(df_subset, shuffle=False, batch_size=128):

    target_dict = {
        col: df_subset[col].to_numpy(
            dtype=np.float32
        ).reshape(-1, 1)
        for col in TARGET_COLS
    }

    prompt_indices = df_subset["prompt_idx"].to_numpy(
        dtype=np.int32
    )

    ds = tf.data.Dataset.from_tensor_slices(
        (
            prompt_indices,
            target_dict
        )
    )

    def resolve_features(prompt_idx, targets):
        embedding = tf.gather(
            tf_embeddings,
            prompt_idx
        )

        return {
            "prompt_emb": embedding
        }, targets

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(df_subset), 20000),
            seed=42,
            reshuffle_each_iteration=True
        )

    ds = (
        ds
        .batch(batch_size)
        .map(
            resolve_features,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds

def build_evaluator_model(
    embedding_dim=384,
    hidden_dim=128
):

    inputs = tf.keras.Input(
        shape=(embedding_dim,),
        name="prompt_emb"
    )

    # First layer
    x = tf.keras.layers.Dense(
        hidden_dim,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(inputs)

    x = tf.keras.layers.Dropout(0.30)(x)

    # Second layer
    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    x = tf.keras.layers.Dropout(0.20)(x)

    # Output heads
    outputs = {
        "helpfulness": tf.keras.layers.Dense(
            1,
            name="helpfulness"
        )(x),

        "honesty": tf.keras.layers.Dense(
            1,
            name="honesty"
        )(x),

        "instruction_following": tf.keras.layers.Dense(
            1,
            name="instruction_following"
        )(x),

        "truthfulness": tf.keras.layers.Dense(
            1,
            name="truthfulness"
        )(x),

        "fine_grained_score": tf.keras.layers.Dense(
            1,
            name="fine_grained_score"
        )(x),

        "overall_score": tf.keras.layers.Dense(
            1,
            name="overall_score"
        )(x),

        "log_tokens": tf.keras.layers.Dense(
            1,
            name="log_tokens"
        )(x)
    }

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )


    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=3e-4
        ),

        loss={
            "helpfulness": "mse",
            "honesty": "mse",
            "instruction_following": "mse",
            "truthfulness": "mse",
            "fine_grained_score": "mse",
            "overall_score": "mse",
            "log_tokens": tf.keras.losses.Huber()
        }
    )

    return model



for llm_name in active_models:

    print("\n" + "=" * 60)
    print(f"Training evaluator for: {llm_name}")
    print("=" * 60)

    df_llm = df[
        df["model"] == llm_name
    ].sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print(f"Total samples: {len(df_llm)}")

    split_idx = int(len(df_llm) * 0.90)

    df_train = df_llm.iloc[:split_idx].copy()
    df_val = df_llm.iloc[split_idx:].copy()

    print(f"Train samples: {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")

    train_ds = build_tf_dataset(
        df_train,
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    val_ds = build_tf_dataset(
        df_val,
        shuffle=False,
        batch_size=BATCH_SIZE
    )

    evaluator = build_evaluator_model()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_overall_score_loss",
        patience=8,
        mode="min",
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_overall_score_loss",
        factor=0.5,
        patience=3,
        mode="min",
        min_lr=1e-6,
        verbose=1
    )

    history = evaluator.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            early_stopping,
            reduce_lr
        ],
        verbose=1
    )

    safe_name = (
        llm_name
        .replace("/", "_")
        .replace("-", "_")
    )

    weight_path = (
        f"evaluator_weights/"
        f"{safe_name}.weights.h5"
    )

    evaluator.save_weights(weight_path)

    print(f"Saved weights → {weight_path}")


    plt.figure(figsize=(16, 5))


    plt.subplot(1, 3, 1)

    plt.plot(
        history.history["overall_score_loss"],
        label="Train"
    )

    plt.plot(
        history.history["val_overall_score_loss"],
        label="Validation"
    )

    plt.title(
        f"{llm_name} - Overall Score Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()

    plt.subplot(1, 3, 2)

    plt.plot(
        history.history["fine_grained_score_loss"],
        label="Train"
    )

    plt.plot(
        history.history["val_fine_grained_score_loss"],
        label="Validation"
    )

    plt.title(
        f"{llm_name} - Fine Grained Score Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()

    plt.subplot(1, 3, 3)

    plt.plot(
        history.history["log_tokens_loss"],
        label="Train"
    )

    plt.plot(
        history.history["val_log_tokens_loss"],
        label="Validation"
    )

    plt.title(
        f"{llm_name} - Token Prediction Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.legend()

    plt.tight_layout()

    plot_path = (
        f"training_plots/"
        f"{safe_name}_history.png"
    )

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved plot → {plot_path}")

# --- CELL ---

