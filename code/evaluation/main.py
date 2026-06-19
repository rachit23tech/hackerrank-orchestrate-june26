import os
import csv
import sys
import json
from datetime import datetime

# Add parent 'code' directory to path so we can import shared modules
code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

import vision
import llm
import guardrails

SAMPLE_CSV = "dataset/sample_claims.csv"
SAMPLE_CACHE = ".cache_sample_eval.json"
REPORT_PATH = "code/evaluation/evaluation_report.md"

USER_HISTORY_PATH = "dataset/user_history.csv"
EVIDENCE_REQ_PATH = "dataset/evidence_requirements.csv"

# Columns to compare predictions vs ground truth
SCORED_COLUMNS = [
    "claim_status",
    "issue_type",
    "object_part",
    "severity",
    "evidence_standard_met",
    "valid_image"
]

OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids", "valid_image", "severity"
]

def load_cache(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Eval] Warning: could not read cache {path}: {e}")
    return {"vision_cache": {}, "llm_cache": {}}

def save_cache(cache, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def load_user_history():
    history = {}
    if not os.path.exists(USER_HISTORY_PATH):
        return history
    with open(USER_HISTORY_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            history[row["user_id"]] = {
                "flags": row.get("history_flags", "none"),
                "summary": row.get("history_summary", "No summary.")
            }
    return history

def load_evidence_requirements():
    requirements = {}
    if not os.path.exists(EVIDENCE_REQ_PATH):
        return requirements
    with open(EVIDENCE_REQ_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            obj = row["claim_object"]
            if obj not in requirements:
                requirements[obj] = []
            requirements[obj].append(
                f"- {row['requirement_id']} ({row['applies_to']}): {row['minimum_image_evidence']}"
            )
    return requirements

def resolve_path(p):
    p = p.strip()
    if os.path.exists(p):
        return p
    candidate = os.path.join("dataset", p)
    if os.path.exists(candidate):
        return candidate
    return p

def run_claim(row, user_history, evidence_reqs, cache):
    user_id = row["user_id"]
    image_paths_raw = row["image_paths"]
    user_claim = row["user_claim"]
    claim_object = row["claim_object"]

    hist = user_history.get(user_id, {"flags": "none", "summary": "New user."})
    history_flags = hist["flags"]
    history_summary = hist["summary"]

    applicable_reqs = []
    if "all" in evidence_reqs:
        applicable_reqs.extend(evidence_reqs["all"])
    if claim_object in evidence_reqs:
        applicable_reqs.extend(evidence_reqs[claim_object])
    evidence_requirements_str = "\n".join(applicable_reqs)

    img_descriptions = {}
    image_list = [p.strip() for p in image_paths_raw.split(";") if p.strip()]
    for img_p in image_list:
        resolved_p = resolve_path(img_p)
        if resolved_p in cache["vision_cache"]:
            img_descriptions[img_p] = cache["vision_cache"][resolved_p]
        else:
            print(f"  [Vision] Processing: {resolved_p}")
            descs = vision.get_image_descriptions(resolved_p)
            cache["vision_cache"][resolved_p] = descs
            img_descriptions[img_p] = descs
            save_cache(cache, SAMPLE_CACHE)

    llm_cache_key = json.dumps({
        "user_claim": user_claim,
        "claim_object": claim_object,
        "history_flags": history_flags,
        "history_summary": history_summary,
        "image_descriptions": img_descriptions
    }, sort_keys=True)

    if llm_cache_key in cache["llm_cache"]:
        raw_pred = cache["llm_cache"][llm_cache_key]
    else:
        print(f"  [LLM] Calling reasoning engine for user: {user_id}")
        raw_pred = llm.evaluate_claim_llm(
            claim_object=claim_object,
            user_claim=user_claim,
            user_history_summary=history_summary,
            user_history_flags=history_flags,
            evidence_requirements=evidence_requirements_str,
            image_descriptions=img_descriptions
        )
        cache["llm_cache"][llm_cache_key] = raw_pred
        save_cache(cache, SAMPLE_CACHE)

    final = guardrails.postprocess_and_guard(
        prediction=raw_pred,
        image_paths=image_paths_raw,
        claim_object=claim_object,
        user_history_flags=history_flags
    )

    return {
        "user_id": user_id,
        "image_paths": image_paths_raw,
        "user_claim": user_claim,
        "claim_object": claim_object,
        "evidence_standard_met": str(final["evidence_standard_met"]).lower(),
        "evidence_standard_met_reason": final["evidence_standard_met_reason"],
        "risk_flags": final["risk_flags"],
        "issue_type": final["issue_type"],
        "object_part": final["object_part"],
        "claim_status": final["claim_status"],
        "claim_status_justification": final["claim_status_justification"],
        "supporting_image_ids": final["supporting_image_ids"],
        "valid_image": str(final["valid_image"]).lower(),
        "severity": final["severity"]
    }

def compute_metrics(predictions, ground_truth, column):
    """Compute accuracy for a single column."""
    total = 0
    correct = 0
    for pred, gt in zip(predictions, ground_truth):
        p_val = str(pred.get(column, "")).strip().lower()
        g_val = str(gt.get(column, "")).strip().lower()
        if not g_val:
            continue
        total += 1
        if p_val == g_val:
            correct += 1
    if total == 0:
        return 0.0, 0, 0
    return correct / total, correct, total

def estimate_costs(num_samples, num_test):
    """Rough cost estimate for Groq llama-3.3-70b-versatile.
    Pricing: ~$0.59/M input tokens, ~$0.79/M output tokens (approximate).
    Estimate ~2500 input tokens and ~200 output tokens per claim.
    """
    input_tokens_per_claim = 2500
    output_tokens_per_claim = 200
    input_price_per_m = 0.59
    output_price_per_m = 0.79

    def cost(n):
        input_cost = (n * input_tokens_per_claim / 1_000_000) * input_price_per_m
        output_cost = (n * output_tokens_per_claim / 1_000_000) * output_price_per_m
        return input_cost + output_cost, n * input_tokens_per_claim, n * output_tokens_per_claim

    sample_cost, sample_input_tok, sample_output_tok = cost(num_samples)
    test_cost, test_input_tok, test_output_tok = cost(num_test)
    return {
        "sample_cost": sample_cost,
        "sample_input_tokens": sample_input_tok,
        "sample_output_tokens": sample_output_tok,
        "test_cost": test_cost,
        "test_input_tokens": test_input_tok,
        "test_output_tokens": test_output_tok
    }

def write_report(metrics_results, predictions, ground_truth, costs_info, num_samples, num_test, runtime_secs):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    total_correct = sum(m["correct"] for m in metrics_results.values())
    total_scored = sum(m["total"] for m in metrics_results.values())
    overall_acc = total_correct / total_scored if total_scored > 0 else 0.0

    lines = [
        "# Evaluation Report — Multi-Modal Evidence Review",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
        f"**Sample dataset rows**: {num_samples}",
        f"**Test dataset rows**: {num_test}",
        "",
        "---",
        "",
        "## Accuracy Metrics (Sample Dataset)",
        "",
        "| Column | Accuracy | Correct | Total |",
        "|---|---|---|---|",
    ]
    for col, m in metrics_results.items():
        lines.append(f"| `{col}` | {m['accuracy']:.1%} | {m['correct']} | {m['total']} |")
    lines += [
        "",
        f"**Overall (across all scored fields)**: {overall_acc:.1%} ({total_correct}/{total_scored})",
        "",
        "---",
        "",
        "## Strategy Comparison",
        "",
        "| Strategy | Description | Notes |",
        "|---|---|---|",
        "| **Strategy 1 (Baseline)** | Local BLIP captioning only + simple rule-based decisions | No LLM — deterministic, fast, but limited reasoning |",
        "| **Strategy 2 (Final)** | BLIP captioning → Groq Llama-3.3-70B reasoning + deterministic guardrails | High-quality contextual reasoning, multilingual, prompt-injection resistant |",
        "",
        "Strategy 2 (our final approach) was adopted because:",
        "- It handles multilingual claims (Hinglish, Spanish) through LLM understanding.",
        "- It produces grounded justifications that reference image evidence.",
        "- It is resistant to adversarial prompt-injection attacks.",
        "- Deterministic guardrails ensure schema conformity and consistent risk flag merging.",
        "",
        "---",
        "",
        "## Operational Analysis",
        "",
        "### Model Calls",
        f"- **BLIP VLM (local)**: 1 call per unique image. Sample set: ~{num_samples * 1}-{num_samples * 3} images. Test set: ~{num_test * 1}-{num_test * 3} images.",
        f"- **Groq Llama-3.3-70B (cloud)**: 1 call per claim row. Sample: {num_samples} calls. Test: {num_test} calls.",
        "",
        "### Token Usage (Estimates)",
        f"| Set | Input Tokens | Output Tokens |",
        "|---|---|---|",
        f"| Sample ({num_samples} rows) | ~{costs_info['sample_input_tokens']:,} | ~{costs_info['sample_output_tokens']:,} |",
        f"| Test ({num_test} rows) | ~{costs_info['test_input_tokens']:,} | ~{costs_info['test_output_tokens']:,} |",
        "",
        "### Images Processed",
        f"- Sample set: ~{num_samples * 2} images (average ~2 per claim).",
        f"- Test set: ~{num_test * 2} images (average ~2 per claim).",
        "",
        "### Approximate Cost (Groq Llama-3.3-70B)",
        f"- Pricing: ~$0.59/M input tokens, ~$0.79/M output tokens",
        f"- **Sample processing cost**: ~${costs_info['sample_cost']:.4f}",
        f"- **Test processing cost**: ~${costs_info['test_cost']:.4f}",
        f"- **Total estimated cost**: ~${costs_info['sample_cost'] + costs_info['test_cost']:.4f}",
        "",
        "### Latency & Runtime",
        f"- **Sample evaluation runtime**: ~{runtime_secs:.1f} seconds",
        f"- **Estimated test set runtime**: ~{(runtime_secs / num_samples) * num_test:.1f} seconds",
        "- BLIP inference: ~0.5-1s per image on CPU.",
        "- Groq API latency: ~1-3s per claim.",
        "",
        "### TPM/RPM Considerations",
        "- Groq's free tier: ~30 RPM and 6,000 TPM for Llama-3.3-70B.",
        "- With ~2,500 input + 200 output tokens per call, the test set of ~45 rows uses ~123,500 tokens total.",
        "- At 6,000 TPM, we'd need ~21 minutes of sustained throughput; at 30 RPM, ~1.5 minutes.",
        "- **Mitigation**: We use a JSON file-based cache that stores results after the first run, making repeated evaluations instant and cost-free.",
        "- For batching: claims are processed sequentially with natural pacing; no explicit throttling is needed for this scale.",
        "- For production scale, we would implement a token-bucket rate limiter and batch BLIP inference on GPU.",
        "",
        "---",
        "",
        "## Final Strategy",
        "",
        "Our system uses **Strategy 2 (BLIP + Groq Llama-3.3-70B)** with the following pipeline:",
        "",
        "1. **Local VLM (BLIP-Large)**: Runs offline on each submitted image to generate 5 targeted visual descriptions.",
        "2. **Cloud LLM (Groq Llama-3.3-70B)**: Receives the complete context (claim transcript, user history, evidence requirements, and image descriptions) and produces a strict JSON prediction.",
        "3. **Deterministic Guardrails**: Validate and normalize every output field, merge user history risk flags, and verify supporting image IDs against actual submitted filenames.",
        "4. **Caching**: Both BLIP outputs and LLM responses are cached to disk to eliminate redundant computation.",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Eval] Report written to {REPORT_PATH}")

def main():
    import time
    start_time = time.time()

    print("[Eval] Starting evaluation on sample_claims.csv...")

    cache = load_cache(SAMPLE_CACHE)
    user_history = load_user_history()
    evidence_reqs = load_evidence_requirements()

    if not os.path.exists(SAMPLE_CSV):
        print(f"[Eval] Error: {SAMPLE_CSV} not found.")
        sys.exit(1)

    rows = []
    with open(SAMPLE_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"[Eval] Processing {len(rows)} sample claims...")
    predictions = []
    for i, row in enumerate(rows):
        print(f"[Eval] [{i+1}/{len(rows)}] user_id={row['user_id']} object={row['claim_object']}")
        pred = run_claim(row, user_history, evidence_reqs, cache)
        predictions.append(pred)

    runtime_secs = time.time() - start_time

    # Compute metrics against ground truth
    metrics_results = {}
    print("\n[Eval] === Accuracy Metrics ===")
    for col in SCORED_COLUMNS:
        acc, correct, total = compute_metrics(predictions, rows, col)
        metrics_results[col] = {"accuracy": acc, "correct": correct, "total": total}
        print(f"  {col}: {acc:.1%} ({correct}/{total})")

    # Estimate costs
    costs_info = estimate_costs(num_samples=len(rows), num_test=45)

    # Write report
    write_report(
        metrics_results=metrics_results,
        predictions=predictions,
        ground_truth=rows,
        costs_info=costs_info,
        num_samples=len(rows),
        num_test=45,
        runtime_secs=runtime_secs
    )

    # Write predictions CSV for sample
    sample_output_path = "code/evaluation/sample_predictions.csv"
    os.makedirs(os.path.dirname(sample_output_path), exist_ok=True)
    with open(sample_output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for pred in predictions:
            writer.writerow(pred)
    print(f"[Eval] Sample predictions written to {sample_output_path}")
    print(f"[Eval] Completed in {runtime_secs:.1f}s")

if __name__ == "__main__":
    main()
