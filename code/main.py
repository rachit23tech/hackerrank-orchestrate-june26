import os
import csv
import json
import argparse
import sys

# Add the 'code' directory to python path
code_dir = os.path.dirname(os.path.abspath(__file__))
if code_dir not in sys.path:
    sys.path.append(code_dir)

import vision
import llm
import guardrails

DEFAULT_INPUT = "dataset/claims.csv"
DEFAULT_OUTPUT = "output.csv"
DEFAULT_CACHE = ".cache_predictions.json"
USER_HISTORY_PATH = "dataset/user_history.csv"
EVIDENCE_REQ_PATH = "dataset/evidence_requirements.csv"

OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids", "valid_image", "severity"
]

def load_cache(cache_path):
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Cache] Warning: {e}")
    return {"vision_cache": {}, "llm_cache": {}}

def save_cache(cache, cache_path):
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"[Cache] Error saving: {e}")

def load_user_history():
    history = {}
    if not os.path.exists(USER_HISTORY_PATH):
        return history
    with open(USER_HISTORY_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            history[row["user_id"]] = {
                "flags": row.get("history_flags", "none"),
                "summary": row.get("history_summary", "No history.")
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
                f"- Requirement {row['requirement_id']} ({row['applies_to']}): {row['minimum_image_evidence']}"
            )
    return requirements

def resolve_image_path(p):
    p = p.strip()
    if os.path.exists(p):
        return p
    candidate = os.path.join("dataset", p)
    if os.path.exists(candidate):
        return candidate
    return p

def process_single_claim(row, user_history, evidence_reqs, cache, cache_path):
    user_id = row["user_id"]
    image_paths_raw = row["image_paths"]
    user_claim = row["user_claim"]
    claim_object = row["claim_object"]

    history_info = user_history.get(user_id, {"flags": "none", "summary": "New user, no prior history."})
    history_flags = history_info["flags"]
    history_summary = history_info["summary"]

    applicable_reqs = []
    if "all" in evidence_reqs:
        applicable_reqs.extend(evidence_reqs["all"])
    if claim_object in evidence_reqs:
        applicable_reqs.extend(evidence_reqs[claim_object])
    evidence_requirements_str = "\n".join(applicable_reqs)

    # Vision: BLIP captioning (cached per image)
    img_descriptions = {}
    image_list = [p.strip() for p in image_paths_raw.split(";") if p.strip()]
    cache_modified = False
    for img_p in image_list:
        resolved_p = resolve_image_path(img_p)
        if resolved_p in cache["vision_cache"]:
            img_descriptions[img_p] = cache["vision_cache"][resolved_p]
        else:
            print(f"[Main] Running local VLM on image: {resolved_p}")
            descs = vision.get_image_descriptions(resolved_p)
            cache["vision_cache"][resolved_p] = descs
            img_descriptions[img_p] = descs
            cache_modified = True

    if cache_modified:
        save_cache(cache, cache_path)

    # LLM reasoning (cached per claim)
    llm_cache_key = json.dumps({
        "user_claim": user_claim,
        "claim_object": claim_object,
        "history_flags": history_flags,
        "history_summary": history_summary,
        "image_descriptions": img_descriptions
    }, sort_keys=True)

    if llm_cache_key in cache["llm_cache"]:
        print(f"[Main] Cache hit for user: {user_id}")
        raw_prediction = cache["llm_cache"][llm_cache_key]
    else:
        print(f"[Main] Calling reasoning engine for user: {user_id}")
        raw_prediction = llm.evaluate_claim_llm(
            claim_object=claim_object,
            user_claim=user_claim,
            user_history_summary=history_summary,
            user_history_flags=history_flags,
            evidence_requirements=evidence_requirements_str,
            image_descriptions=img_descriptions
        )
        cache["llm_cache"][llm_cache_key] = raw_prediction
        save_cache(cache, cache_path)

    # Guardrails
    final_pred = guardrails.postprocess_and_guard(
        prediction=raw_prediction,
        image_paths=image_paths_raw,
        claim_object=claim_object,
        user_history_flags=history_flags
    )

    return {
        "user_id": user_id,
        "image_paths": image_paths_raw,
        "user_claim": user_claim,
        "claim_object": claim_object,
        "evidence_standard_met": str(final_pred["evidence_standard_met"]).lower(),
        "evidence_standard_met_reason": final_pred["evidence_standard_met_reason"],
        "risk_flags": final_pred["risk_flags"],
        "issue_type": final_pred["issue_type"],
        "object_part": final_pred["object_part"],
        "claim_status": final_pred["claim_status"],
        "claim_status_justification": final_pred["claim_status_justification"],
        "supporting_image_ids": final_pred["supporting_image_ids"],
        "valid_image": str(final_pred["valid_image"]).lower(),
        "severity": final_pred["severity"]
    }

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Evidence Review Pipeline")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(f"[Main] Starting pipeline. Input: {args.input}, Output: {args.output}")

    cache = load_cache(args.cache)
    user_history = load_user_history()
    evidence_reqs = load_evidence_requirements()

    if not os.path.exists(args.input):
        print(f"[Main] Error: Input file {args.input} not found.")
        sys.exit(1)

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.limit:
        print(f"[Main] Limiting processing to first {args.limit} rows.")
        rows = rows[:args.limit]

    print(f"[Main] Loaded {len(rows)} claims to evaluate.")

    results = []
    for i, row in enumerate(rows):
        print(f"[Main] [{i+1}/{len(rows)}] Processing claim for user {row['user_id']}...")
        res = process_single_claim(row, user_history, evidence_reqs, cache, args.cache)
        results.append(res)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for res in results:
            writer.writerow(res)

    print(f"[Main] Pipeline completed successfully. Output written to {args.output}")

if __name__ == "__main__":
    main()
