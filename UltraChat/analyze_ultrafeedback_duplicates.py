"""
UltraFeedback Duplicate Query Analysis & Exact-Match Caching Simulation
Calculates duplicate query occurrences and evaluates caching feasibility.
"""

import os
import sys
import json
import csv
import re
from collections import Counter

# Set UTF-8 encoding for standard output
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

def run_analysis():
    print("=" * 60)
    print("Loading openbmb/UltraFeedback dataset...")
    print("=" * 60)
    
    from datasets import load_dataset
    
    split_name = "train"
    dataset = load_dataset("openbmb/UltraFeedback", split=split_name)
    total_records = len(dataset)
    print(f"Loaded {total_records} records from UltraFeedback ({split_name} split).")

    # Step 2: Extract instructions and filter missing/empty
    valid_queries = []
    excluded_count = 0

    for i, item in enumerate(dataset):
        instruction = item.get("instruction")
        if instruction is None:
            excluded_count += 1
            continue
        text = str(instruction)
        if len(text.strip()) == 0:
            excluded_count += 1
            continue
        valid_queries.append(text)

    total_valid = len(valid_queries)

    # Step 3: Exact duplicate queries analysis
    exact_counter = Counter(valid_queries)
    unique_exact = len(exact_counter)
    duplicate_occurrences = total_valid - unique_exact
    
    # Distinct query texts whose frequency is > 1
    distinct_duplicated_types = sum(1 for q, count in exact_counter.items() if count > 1)
    appearing_exactly_twice = sum(1 for q, count in exact_counter.items() if count == 2)
    appearing_three_plus = sum(1 for q, count in exact_counter.items() if count >= 3)
    
    most_frequent = exact_counter.most_common(10)

    # Step 4: Normalized query analysis
    # remove leading/trailing whitespace and collapse internal repeated whitespace
    norm_queries = [re.sub(r'\s+', ' ', q.strip()) for q in valid_queries]
    norm_counter = Counter(norm_queries)
    unique_norm = len(norm_counter)
    duplicate_norm_occurrences = total_valid - unique_norm
    norm_cache_hit_rate = (duplicate_norm_occurrences / total_valid * 100) if total_valid > 0 else 0.0

    # Step 5: Simulate sequential exact-match caching
    cache_set = set()
    cache_hits = 0
    cache_misses = 0

    for q in valid_queries:
        if q in cache_set:
            cache_hits += 1
        else:
            cache_misses += 1
            cache_set.add(q)

    cache_hit_rate = (cache_hits / total_valid * 100) if total_valid > 0 else 0.0
    cache_miss_rate = (cache_misses / total_valid * 100) if total_valid > 0 else 0.0
    llm_call_reduction = cache_hit_rate

    # Step 6: Save results to JSON
    json_results = {
        "dataset_name": "openbmb/UltraFeedback",
        "dataset_split": split_name,
        "total_records": total_records,
        "valid_queries": total_valid,
        "excluded_empty_or_missing_queries": excluded_count,
        "exact_match_metrics": {
            "unique_exact_queries": unique_exact,
            "duplicate_occurrences": duplicate_occurrences,
            "distinct_duplicated_query_types": distinct_duplicated_types,
            "queries_appearing_exactly_twice": appearing_exactly_twice,
            "queries_appearing_3_plus_times": appearing_three_plus,
            "top_10_most_frequent_queries": [
                {"query": q, "count": count} for q, count in most_frequent if count > 1
            ]
        },
        "sequential_cache_simulation": {
            "total_queries_processed": total_valid,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate_pct": round(cache_hit_rate, 4),
            "cache_miss_rate_pct": round(cache_miss_rate, 4),
            "potential_reduction_in_llm_calls_pct": round(llm_call_reduction, 4)
        },
        "normalized_query_metrics": {
            "unique_normalized_queries": unique_norm,
            "duplicate_normalized_occurrences": duplicate_norm_occurrences,
            "normalized_cache_hit_rate_pct": round(norm_cache_hit_rate, 4)
        }
    }

    # Save to root and docs
    json_path = "ultrafeedback_duplicate_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    # Step 7: Save CSV of duplicated exact query texts and counts
    csv_path = "ultrafeedback_duplicate_queries.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_text", "occurrence_count"])
        # sort by count desc, then alphabetically
        dup_items = [(q, count) for q, count in exact_counter.items() if count > 1]
        dup_items.sort(key=lambda x: x[1], reverse=True)
        for q, count in dup_items:
            writer.writerow([q, count])

    # Copy files to docs/ as well for organizational neatness
    docs_dir = "docs"
    if os.path.exists(docs_dir):
        with open(os.path.join(docs_dir, "ultrafeedback_duplicate_analysis.json"), "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        with open(os.path.join(docs_dir, "ultrafeedback_duplicate_queries.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["query_text", "occurrence_count"])
            for q, count in dup_items:
                writer.writerow([q, count])

    # Step 8: Print the exact final report formatted as requested
    print()
    print("ULTRAFEEDBACK DUPLICATE QUERY ANALYSIS")
    print(f"Dataset split: {split_name}")
    print(f"Total records: {total_records}")
    print(f"Valid queries: {total_valid}")
    print(f"Unique exact queries: {unique_exact}")
    print(f"Duplicate occurrences: {duplicate_occurrences}")
    print(f"Distinct duplicated query types: {distinct_duplicated_types}")
    print(f"Queries appearing exactly twice: {appearing_exactly_twice}")
    print(f"Queries appearing 3+ times: {appearing_three_plus}")
    print()
    print("SEQUENTIAL EXACT-MATCH CACHE SIMULATION")
    print(f"Cache hits: {cache_hits}")
    print(f"Cache misses: {cache_misses}")
    print(f"Cache hit rate: {cache_hit_rate:.2f}%")
    print(f"Cache miss rate: {cache_miss_rate:.2f}%")
    print(f"Potential reduction in LLM calls: {llm_call_reduction:.2f}%")
    print()
    print("NORMALIZED QUERY ANALYSIS")
    print(f"Unique normalized queries: {unique_norm}")
    print(f"Duplicate normalized occurrences: {duplicate_norm_occurrences}")
    print(f"Normalized cache hit rate: {norm_cache_hit_rate:.2f}%")
    print()
    print("Output files:")
    print("ultrafeedback_duplicate_analysis.json")
    print("ultrafeedback_duplicate_queries.csv")
    print()

if __name__ == "__main__":
    run_analysis()
