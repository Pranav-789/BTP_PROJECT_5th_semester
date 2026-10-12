import time
import requests
import pandas as pd
import streamlit as st

# Configure page setup
st.set_page_config(
    page_title="UltraChat | Dual-Agent LLM Router",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #2563eb, #7c3aed, #db2777);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-cache-hit {
        background-color: #dcfce7;
        color: #15803d;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-cache-miss {
        background-color: #f1f5f9;
        color: #475569;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-model {
        background-color: #ede9fe;
        color: #6d28d9;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/artificial-intelligence.png", width=70)
    st.title("Settings & Control")

    backend_url = st.text_input("Backend API URL", value="http://localhost:8000").rstrip("/")

    # Live Health Check
    health_data = None
    for target in [backend_url, backend_url.replace("localhost", "127.0.0.1")]:
        try:
            health_resp = requests.get(f"{target}/api/v1/health", timeout=8)
            if health_resp.status_code == 200:
                health_data = health_resp.json()
                backend_url = target
                break
        except Exception:
            continue

    if health_data:
        st.success("API Backend: Connected (Healthy)")
        with st.expander("System Diagnostic Details", expanded=False):
            st.write(f"**Loaded Models:** {health_data.get('evaluator_models_loaded', 0)}/15")
            st.write(f"**Local Vector Cache:** {'Active' if health_data.get('vector_cache_active') else 'Inactive'}")
            st.write(f"**Active API Keys:** {', '.join(health_data.get('available_api_keys', [])) or 'None'}")
    else:
        st.error("API Backend: Offline (Check if uvicorn is running on port 8000)")

    st.markdown("---")
    st.subheader("Routing Strategy")
    strategy_options = {
        "balanced": "Balanced (Utility + Instruction - Cost)",
        "best_quality": "Best Quality (Prioritizes High Capability)",
        "cost_optimized": "Cost Optimized (Favors Fast, Low-Token Models)",
        "instruction_focused": "Instruction Focused (Emphasizes Honesty & Truth)",
    }
    selected_strategy = st.selectbox(
        "Optimization Objective",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        index=0,
    )

    st.markdown("---")
    st.subheader("Generation Hyperparameters")
    use_cache = st.toggle("Enable Local Vector Cache", value=True)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.05)
    max_tokens = st.slider("Max Output Tokens", min_value=128, max_value=2048, value=1024, step=64)

    st.markdown("---")
    if st.button("Clear Local Semantic Cache", use_container_width=True):
        try:
            clear_res = requests.post(f"{backend_url}/api/v1/cache/clear", timeout=5)
            if clear_res.status_code == 200:
                st.success("Local Qdrant Cache Cleared!")
            else:
                st.error("Failed to clear cache.")
        except Exception as err:
            st.error(f"Error: {err}")

# App Header
st.markdown('<div class="main-header">UltraChat Intelligent Router</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "Dual-Agent Architecture: Multi-Task TensorFlow Evaluators + Local Qdrant Semantic Cache + Agent 2 Self-Correction"
    "</div>",
    unsafe_allow_html=True,
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Interactive Dual-Agent Pipeline",
        "Fast Self-Correction Loop (/ask)",
        "Neural Evaluator Benchmark (/predict-scores)",
        "Architecture & Model Roster",
    ]
)

# ---------------------------------------------------------
# TAB 1: Detailed Interactive Dual-Agent Pipeline
# ---------------------------------------------------------
with tab1:
    st.markdown("### Test Dynamic Prompt Routing")
    st.write(
        "Submits your query through the complete dual-agent pipeline: checks local semantic cache, "
        "predicts scores with 15 multi-task neural networks, routes to the Pareto-optimal LLM, and applies Agent 2 guardrails."
    )

    # Sample prompt buttons
    st.markdown("**Sample Prompts:**")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    # Initialize session state for prompt if not present
    if "prompt_input" not in st.session_state:
        st.session_state["prompt_input"] = ""

    if col_p1.button("Reasoning & Math", use_container_width=True):
        st.session_state["prompt_input"] = "Solve step-by-step: A train leaves Station A at 60 km/h. Another leaves Station B towards A at 90 km/h. They are 300 km apart. When and where do they meet?"
        st.rerun()
    if col_p2.button("Code Generation", use_container_width=True):
        st.session_state["prompt_input"] = "Write a Python asyncio program that concurrently downloads three web pages with timeout and error handling."
        st.rerun()
    if col_p3.button("Creative Writing", use_container_width=True):
        st.session_state["prompt_input"] = "Write a short cyberpunk noir monologue of a private detective in Neo-Kyoto looking for a rogue AI memory shard."
        st.rerun()
    if col_p4.button("Concept Explanation", use_container_width=True):
        st.session_state["prompt_input"] = "Explain how transformer self-attention works mathematically with queries, keys, and values in simple terms."
        st.rerun()

    user_query = st.text_area(
        "Enter Prompt:",
        key="prompt_input",
        placeholder="Type any prompt or choose one of the sample buttons above...",
        height=120,
    )

    col_force, col_submit = st.columns([2, 1])
    with col_force:
        force_model_choice = st.selectbox(
            "Optional Model Override (Bypass routing for comparison):",
            options=["None (Let Agent 1 Decide)"]
            + [
                "gpt_4",
                "gpt_3.5_turbo",
                "llama_2_70b_chat",
                "llama_2_13b_chat",
                "llama_2_7b_chat",
                "falcon_40b_instruct",
                "vicuna_33b",
                "wizardlm_70b",
                "wizardlm_13b",
                "wizardlm_7b",
                "ultralm_65b",
                "ultralm_13b",
                "mpt_30b_chat",
                "starchat",
                "alpaca_7b",
            ],
            index=0,
        )

    with col_submit:
        st.write("")
        st.write("")
        submit_btn = st.button("Route & Generate Response", type="primary", use_container_width=True)

    if submit_btn:
        if not user_query.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Executing Dual-Agent Pipeline..."):
                start_clock = time.time()
                payload = {
                    "prompt": user_query.strip(),
                    "strategy": selected_strategy,
                    "use_cache": use_cache,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "force_model": None if "None" in force_model_choice else force_model_choice,
                }
                try:
                    resp = requests.post(f"{backend_url}/api/v1/route", json=payload, timeout=60)
                    if resp.status_code == 200:
                        data = resp.json()
                        elapsed_total = round((time.time() - start_clock) * 1000, 1)

                        # Top Highlights
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("Selected Model", data.get("selected_model", "Unknown"))
                        with c2:
                            is_cached = data.get("cached", False)
                            st.metric("Cache Status", "HIT (0 LLM Cost)" if is_cached else "MISS (Live LLM)")
                        with c3:
                            st.metric("Total Latency", f"{data.get('latency_ms', elapsed_total)} ms")
                        with c4:
                            eval_status = data.get("evaluation", {}).get("status", "PASSED")
                            st.metric("Agent 2 Guardrail", eval_status)

                        st.markdown("---")

                        # Response Section
                        st.subheader("Generated Response")
                        st.markdown(data.get("response_text", ""))

                        # Agent 1 Routing Reasoning
                        st.info(f"**Agent 1 Decision Rationale:** {data.get('routing_reasoning', '')}")

                        # Agent 2 Evaluation Notes
                        eval_notes = data.get("evaluation", {}).get("evaluation_notes", "")
                        if eval_notes:
                            st.caption(f"**Agent 2 Validation Notes:** {eval_notes} (Retries: {data.get('evaluation', {}).get('retry_count', 0)})")

                        # Predictions Breakdown (if cache miss)
                        predictions = data.get("model_predictions", [])
                        if predictions:
                            st.markdown("---")
                            st.subheader("Agent 1 Multi-Task Evaluator Scores (UltraFeedback)")
                            df_preds = pd.DataFrame(predictions)

                            # Reorder for clarity
                            cols_order = [
                                "model_name",
                                "composite_score",
                                "overall_score",
                                "fine_grained_score",
                                "instruction_following",
                                "truthfulness",
                                "helpfulness",
                                "honesty",
                                "predicted_tokens",
                            ]
                            available_cols = [c for c in cols_order if c in df_preds.columns]
                            df_display = df_preds[available_cols].rename(
                                columns={
                                    "model_name": "Model",
                                    "composite_score": "Composite Utility",
                                    "overall_score": "Overall",
                                    "fine_grained_score": "Fine-Grained",
                                    "instruction_following": "Instruction",
                                    "truthfulness": "Truthfulness",
                                    "helpfulness": "Helpfulness",
                                    "honesty": "Honesty",
                                    "predicted_tokens": "Est. Tokens",
                                }
                            )

                            # Plot Composite Scores
                            chart_df = df_display.set_index("Model")[["Composite Utility"]].head(10)
                            st.bar_chart(chart_df)

                            with st.expander("View Full 15-Model Multi-Task Evaluation Table", expanded=False):
                                st.dataframe(df_display, use_container_width=True)

                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
                except Exception as ex:
                    st.error(f"Failed to communicate with API: {ex}")

# ---------------------------------------------------------
# TAB 2: Fast Self-Correction Loop (/api/v1/ask)
# ---------------------------------------------------------
with tab2:
    st.markdown("### Thesis Flowchart Self-Correction Loop (`POST /api/v1/ask`)")
    st.write(
        "Runs the automated self-correction loop: "
        "1. Check Semantic Cache $\\rightarrow$ "
        "2. Agent 1 Selects Best LLM $\\rightarrow$ "
        "3. Live Execution $\\rightarrow$ "
        "4. Agent 2 Evaluates $\\rightarrow$ "
        "5. If rejected, penalizes and eliminates model, retrying with next best candidate."
    )

    ask_prompt = st.text_area(
        "Prompt for Self-Correction Flowchart:",
        value="Explain the difference between synchronous and asynchronous programming in Python.",
        height=100,
    )
    col_retries, col_ask_btn = st.columns([1, 2])
    with col_retries:
        max_retries_input = st.number_input("Max Retries", min_value=1, max_value=5, value=3)
    with col_ask_btn:
        st.write("")
        st.write("")
        run_ask_btn = st.button("Execute /ask Flowchart", type="primary", use_container_width=True)

    if run_ask_btn:
        if not ask_prompt.strip():
            st.warning("Enter a prompt.")
        else:
            with st.spinner("Running self-correction flowchart loop..."):
                try:
                    ask_resp = requests.post(
                        f"{backend_url}/api/v1/ask",
                        json={
                            "prompt": ask_prompt.strip(),
                            "use_cache": use_cache,
                            "max_retries": max_retries_input,
                        },
                        timeout=60,
                    )
                    if ask_resp.status_code == 200:
                        ask_data = ask_resp.json()
                        a1, a2, a3 = st.columns(3)
                        with a1:
                            st.metric("Final Model Selected", ask_data.get("selected_model"))
                        with a2:
                            st.metric("Retries Needed", ask_data.get("retries", 0))
                        with a3:
                            st.metric("Served from Cache", "Yes" if ask_data.get("cached") else "No")

                        eliminated = ask_data.get("eliminated_models", [])
                        if eliminated:
                            st.warning(f"**Models Penalized & Eliminated by Agent 2:** {', '.join(eliminated)}")
                        else:
                            st.success("No models eliminated. First-attempt response passed Agent 2 validation.")

                        st.subheader("Response Output")
                        st.markdown(ask_data.get("response", ""))
                    else:
                        st.error(f"Error {ask_resp.status_code}: {ask_resp.text}")
                except Exception as ex:
                    st.error(f"Error: {ex}")

# ---------------------------------------------------------
# TAB 3: Neural Evaluator Benchmark (/api/v1/predict-scores)
# ---------------------------------------------------------
with tab3:
    st.markdown("### Predict Evaluator Scores Across 15 Models (`POST /api/v1/predict-scores`)")
    st.write(
        "Runs the query embedding through all 15 trained TensorFlow neural network weights without invoking "
        "any LLM generation APIs. Inspect the multi-head metrics predicted for each model."
    )

    pred_prompt = st.text_input(
        "Prompt to Benchmark:",
        value="Write a Python function to compute the Fibonacci sequence using memoization.",
    )

    if st.button("Evaluate All 15 Models", use_container_width=True):
        if not pred_prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Running inference across 15 TensorFlow models..."):
                try:
                    p_resp = requests.post(
                        f"{backend_url}/api/v1/predict-scores",
                        json={"prompt": pred_prompt.strip(), "strategy": selected_strategy},
                        timeout=30,
                    )
                    if p_resp.status_code == 200:
                        p_data = p_resp.json()
                        st.success(f"Evaluated {p_data.get('total_models_evaluated')} models. Top Recommended: **{p_data.get('top_recommended_model')}**")

                        predictions = p_data.get("predictions", [])
                        if predictions:
                            df_p = pd.DataFrame(predictions)
                            st.dataframe(
                                df_p.sort_values(by="composite_score", ascending=False),
                                use_container_width=True,
                            )

                            # Metric comparison chart
                            st.subheader("Composite Score Comparison")
                            st.bar_chart(df_p.set_index("model_name")["composite_score"])
                    else:
                        st.error(f"Error {p_resp.status_code}: {p_resp.text}")
                except Exception as ex:
                    st.error(f"Error: {ex}")

# ---------------------------------------------------------
# TAB 4: Architecture & Model Roster
# ---------------------------------------------------------
with tab4:
    st.markdown("### Dual-Agent System Architecture")
    st.markdown(
        """
        ```mermaid
        graph TD
            A[User Prompt] --> B[Generate 384-d Embedding]
            B --> C{Local Qdrant Semantic Cache}
            C -- Cosine >= 0.92 -- > D[Instant Cache Hit Response]
            C -- Cache Miss --> E[Agent 1: Intelligent Router]
            E --> F[15 Multi-Task TF Evaluators]
            F --> G[Pareto Strategy Scoring]
            G --> H[Candidate Model Selected]
            H --> I[LLM Executor: Gemini / Groq / HF]
            I --> J[Agent 2: Evaluator Guardrail]
            J -- Passes Validation --> K[Store in Local Vector DB & Return]
            J -- Rejection / Failure --> L[Eliminate Model & Retry Loop]
            L --> E
        ```
        """
    )

    st.markdown("---")
    st.subheader("15 Supported LLMs & Operational Tiers")
    st.markdown(
        """
        | Operational Tier | Model Identifier | Execution Target / Proxy | Cost Factor |
        | :--- | :--- | :--- | :--- |
        | **Tier 1 (Low Latency)** | `alpaca_7b` | Groq / HuggingFace | 0.10 |
        | **Tier 1 (Low Latency)** | `llama_2_7b_chat` | Groq (`llama-3.1-8b-instant`) | 0.12 |
        | **Tier 1 (Low Latency)** | `wizardlm_7b` | Groq (`llama-3.1-8b-instant`) | 0.12 |
        | **Tier 1 (Low Latency)** | `starchat` | Groq / HuggingFace | 0.15 |
        | **Tier 1 (Low Latency)** | `gpt_3.5_turbo` | Google Gemini (`gemini-2.5-flash-lite`) | 0.20 |
        | **Tier 2 (Medium)** | `llama_2_13b_chat` | Groq (`llama-3.1-8b-instant`) | 0.35 |
        | **Tier 2 (Medium)** | `ultralm_13b` | Groq (`llama-3.1-8b-instant`) | 0.35 |
        | **Tier 2 (Medium)** | `wizardlm_13b` | Groq (`llama-3.1-8b-instant`) | 0.35 |
        | **Tier 2 (Medium)** | `mpt_30b_chat` | Groq (`llama-3.3-70b-versatile`) | 0.50 |
        | **Tier 2 (Medium)** | `vicuna_33b` | Groq (`llama-3.3-70b-versatile`) | 0.55 |
        | **Tier 3 (High Quality)** | `falcon_40b_instruct` | Groq (`llama-3.3-70b-versatile`) | 0.70 |
        | **Tier 3 (High Quality)** | `ultralm_65b` | Groq (`llama-3.3-70b-versatile`) | 0.85 |
        | **Tier 3 (High Quality)** | `llama_2_70b_chat` | Groq (`llama-3.3-70b-versatile`) | 0.90 |
        | **Tier 3 (High Quality)** | `wizardlm_70b` | Groq (`llama-3.3-70b-versatile`) | 0.90 |
        | **Tier 3 (High Quality)** | `gpt_4` | Google Gemini (`gemini-2.5-flash`) | 1.00 |
        """
    )
