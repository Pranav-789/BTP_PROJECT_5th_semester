"""
Export complete MT-Bench & MMLU Benchmark Evaluation Dataset to CSV.
Extracts input prompts, routed decisions, quality scores, predicted tokens,
and baseline comparisons across all 64 standardized queries.
"""

import os
import json
import csv

def generate_benchmark_csv():
    json_path = os.path.join(os.path.dirname(__file__), "..", "UltraChat", "benchmark_results.json")
    json_path = os.path.abspath(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["detailed_query_records"]
    print(f"Loaded {len(records)} benchmark queries from {json_path}")

    fieldnames = [
        "query_id",
        "benchmark",
        "category",
        "input_prompt",
        "selected_model_balanced",
        "dispatched_tier",
        "router_utility_score",
        "output_quality_score",
        "instruction_following_score",
        "helpfulness_score",
        "truthfulness_score",
        "predicted_output_tokens",
        "cost_factor",
        "projected_cost_per_query_usd",
        "gpt4_baseline_quality",
        "quality_retention_pct",
        "cost_savings_vs_gpt4_pct",
        "selected_model_best_quality",
        "best_quality_score",
        "selected_model_cost_optimized",
        "cost_optimized_quality_score",
        "selected_model_instruction_focused",
        "gpt35_baseline_quality",
        "llama2_7b_baseline_quality",
        "evaluation_verdict"
    ]

    rows = []
    for r in records:
        b_strat = r["strategies"]["balanced"]
        bq_strat = r["strategies"]["best_quality"]
        co_strat = r["strategies"]["cost_optimized"]
        if_strat = r["strategies"]["instruction_focused"]
        baselines = r["baselines"]

        gpt4_q = baselines["gpt_4"]["overall"]
        q_ret = (b_strat["overall_quality"] / gpt4_q) * 100 if gpt4_q else 100.0
        cost_factor = b_strat["cost_factor"]
        cost_sav = (1.0 - cost_factor) * 100.0
        
        # GPT-4 reference rate is $30.00 / 1k queries = $0.030 per query
        cost_per_query = cost_factor * 0.030

        tier = b_strat["tier"]
        if "Tier 3" in tier:
            verdict = "Tier 3 Frontier Dispatched (High Complexity / Precision Task)"
        elif "Tier 2" in tier:
            verdict = "Tier 2 Mid-Range Dispatched (Intermediate Technical Depth)"
        else:
            verdict = "Tier 1 Lightweight Dispatched (Frugal Routing - 80% Cost Savings)"

        rows.append({
            "query_id": r["id"],
            "benchmark": r["benchmark"],
            "category": r["category"],
            "input_prompt": r["prompt"],
            "selected_model_balanced": b_strat["selected_model"],
            "dispatched_tier": b_strat["tier"],
            "router_utility_score": round(b_strat["utility"], 4),
            "output_quality_score": round(b_strat["overall_quality"], 4),
            "instruction_following_score": round(b_strat["instruction_score"], 4),
            "helpfulness_score": round(b_strat["helpfulness"], 4),
            "truthfulness_score": round(b_strat["truthfulness"], 4),
            "predicted_output_tokens": round(b_strat["predicted_tokens"], 1),
            "cost_factor": round(cost_factor, 3),
            "projected_cost_per_query_usd": f"${cost_per_query:.5f}",
            "gpt4_baseline_quality": round(gpt4_q, 4),
            "quality_retention_pct": f"{q_ret:.2f}%",
            "cost_savings_vs_gpt4_pct": f"{cost_sav:.1f}%",
            "selected_model_best_quality": bq_strat["selected_model"],
            "best_quality_score": round(bq_strat["overall_quality"], 4),
            "selected_model_cost_optimized": co_strat["selected_model"],
            "cost_optimized_quality_score": round(co_strat["overall_quality"], 4),
            "selected_model_instruction_focused": if_strat["selected_model"],
            "gpt35_baseline_quality": round(baselines["gpt_3.5_turbo"]["overall"], 4),
            "llama2_7b_baseline_quality": round(baselines["llama_2_7b_chat"]["overall"], 4),
            "evaluation_verdict": verdict
        })

    # Save to docs/ and UltraChat/
    dest_paths = [
        os.path.join(os.path.dirname(__file__), "BENCHMARK_EVALUATION_DATASET.csv"),
        os.path.join(os.path.dirname(__file__), "..", "UltraChat", "benchmark_evaluation_results.csv")
    ]

    for p in dest_paths:
        p_abs = os.path.abspath(p)
        try:
            with open(p_abs, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Generated CSV: {p_abs}")
        except PermissionError:
            print(f"[WARNING] Could not save to '{p_abs}': File is locked (likely open in Microsoft Excel or another viewer). Close it and re-run.")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV to '{p_abs}': {e}")

if __name__ == "__main__":
    generate_benchmark_csv()
