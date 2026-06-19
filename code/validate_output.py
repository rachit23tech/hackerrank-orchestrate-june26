"""
Validate output.csv for schema conformity and allowed values.
Run: python code/validate_output.py --output output.csv --input dataset/claims.csv
"""
import csv
import sys
import argparse

REQUIRED_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids", "valid_image", "severity"
]

ALLOWED_STATUSES = {"supported", "contradicted", "not_enough_information"}
ALLOWED_ISSUES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown"
}
ALLOWED_SEVERITIES = {"none", "low", "medium", "high", "unknown"}
ALLOWED_RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
    "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required"
}

def validate(output_path, input_path):
    errors = []
    warnings = []

    # Load input to check row count
    with open(input_path, "r", encoding="utf-8") as f:
        input_rows = list(csv.DictReader(f))

    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        output_cols = reader.fieldnames or []
        output_rows = list(reader)

    # 1. Column check
    for col in REQUIRED_COLUMNS:
        if col not in output_cols:
            errors.append(f"MISSING COLUMN: '{col}'")

    extra_cols = [c for c in output_cols if c not in REQUIRED_COLUMNS]
    if extra_cols:
        warnings.append(f"EXTRA COLUMNS (ignored by evaluator): {extra_cols}")

    # 2. Column order check
    expected_order = REQUIRED_COLUMNS
    actual_order = [c for c in output_cols if c in REQUIRED_COLUMNS]
    if actual_order != expected_order:
        errors.append(f"COLUMN ORDER MISMATCH.\n  Expected: {expected_order}\n  Got:      {actual_order}")

    # 3. Row count check
    if len(output_rows) != len(input_rows):
        errors.append(f"ROW COUNT MISMATCH: output has {len(output_rows)} rows, input has {len(input_rows)} rows.")

    # 4. Per-row validation
    for i, row in enumerate(output_rows):
        rownum = i + 2  # 1-indexed with header

        # claim_status
        status = str(row.get("claim_status", "")).strip().lower()
        if status not in ALLOWED_STATUSES:
            errors.append(f"Row {rownum}: invalid claim_status='{status}'")

        # issue_type
        issue = str(row.get("issue_type", "")).strip().lower()
        if issue not in ALLOWED_ISSUES:
            errors.append(f"Row {rownum}: invalid issue_type='{issue}'")

        # severity
        severity = str(row.get("severity", "")).strip().lower()
        if severity not in ALLOWED_SEVERITIES:
            errors.append(f"Row {rownum}: invalid severity='{severity}'")

        # evidence_standard_met
        esm = str(row.get("evidence_standard_met", "")).strip().lower()
        if esm not in {"true", "false"}:
            errors.append(f"Row {rownum}: invalid evidence_standard_met='{esm}' (must be 'true' or 'false')")

        # valid_image
        vi = str(row.get("valid_image", "")).strip().lower()
        if vi not in {"true", "false"}:
            errors.append(f"Row {rownum}: invalid valid_image='{vi}' (must be 'true' or 'false')")

        # risk_flags
        risk_raw = str(row.get("risk_flags", "")).strip()
        if risk_raw and risk_raw.lower() != "none":
            for flag in risk_raw.split(";"):
                flag = flag.strip().lower()
                if flag and flag not in ALLOWED_RISK_FLAGS:
                    errors.append(f"Row {rownum}: invalid risk_flag='{flag}'")

        # supporting_image_ids
        supp = str(row.get("supporting_image_ids", "")).strip()
        if not supp:
            errors.append(f"Row {rownum}: supporting_image_ids is empty (should be 'none' if no images)")

    # Summary
    print(f"\n{'='*60}")
    print(f"VALIDATION REPORT: {output_path}")
    print(f"{'='*60}")
    print(f"  Rows in output:  {len(output_rows)}")
    print(f"  Rows in input:   {len(input_rows)}")
    print(f"  Errors:          {len(errors)}")
    print(f"  Warnings:        {len(warnings)}")

    if warnings:
        print(f"\n⚠️  WARNINGS:")
        for w in warnings:
            print(f"   - {w}")

    if errors:
        print(f"\n❌ ERRORS:")
        for e in errors:
            print(f"   - {e}")
        print(f"\n{'='*60}")
        print("VALIDATION FAILED. Fix the above errors before submitting.")
        sys.exit(1)
    else:
        print(f"\n✅ VALIDATION PASSED. {output_path} is submission-ready.")

def main():
    parser = argparse.ArgumentParser(description="Validate output.csv against schema requirements.")
    parser.add_argument("--output", default="output.csv", help="Path to the output predictions CSV")
    parser.add_argument("--input", default="dataset/claims.csv", help="Path to the input claims CSV")
    args = parser.parse_args()
    validate(args.output, args.input)

if __name__ == "__main__":
    main()
