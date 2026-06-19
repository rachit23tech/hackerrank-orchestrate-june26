import os

ALLOWED_STATUSES = {"supported", "contradicted", "not_enough_information"}
ALLOWED_ISSUES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown"
}
ALLOWED_SEVERITIES = {"none", "low", "medium", "high", "unknown"}

CAR_PARTS = {
    "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
    "headlight", "taillight", "fender", "quarter_panel", "body", "unknown"
}
LAPTOP_PARTS = {
    "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base", "body", "unknown"
}
PACKAGE_PARTS = {
    "box", "package_corner", "package_side", "seal", "label", "contents", "item", "unknown"
}

ALLOWED_RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
    "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required"
}

def get_valid_parts(claim_object: str) -> set:
    if claim_object == "car":
        return CAR_PARTS
    elif claim_object == "laptop":
        return LAPTOP_PARTS
    elif claim_object == "package":
        return PACKAGE_PARTS
    return {"unknown"}

def postprocess_and_guard(
    prediction: dict,
    image_paths: str,
    claim_object: str,
    user_history_flags: str
) -> dict:
    """Enforce strict allowed value constraints and consistency rules on prediction dictionary."""
    # Create a copy to avoid mutating the input
    output = prediction.copy()
    
    # 1. Parse and validate boolean fields
    for bool_field in ["evidence_standard_met", "valid_image"]:
        val = output.get(bool_field, False)
        if isinstance(val, str):
            output[bool_field] = val.strip().lower() in ["true", "1", "yes"]
        else:
            output[bool_field] = bool(val)

    # 2. Normalize and check claim_status
    status = str(output.get("claim_status", "not_enough_information")).strip().lower()
    if status not in ALLOWED_STATUSES:
        status = "not_enough_information"
    output["claim_status"] = status

    # 3. Normalize and check issue_type
    issue = str(output.get("issue_type", "unknown")).strip().lower()
    if issue not in ALLOWED_ISSUES:
        issue = "unknown"
    output["issue_type"] = issue

    # 4. Normalize and check severity
    severity = str(output.get("severity", "unknown")).strip().lower()
    if severity not in ALLOWED_SEVERITIES:
        severity = "unknown"
    # Consistency rule: if claim_status is contradicted or not_enough_information, or issue is none, severity should typically be low or none unless specified.
    # We let LLM decide but normalise string.
    output["severity"] = severity

    # 5. Normalize and check object_part based on object type
    part = str(output.get("object_part", "unknown")).strip().lower()
    valid_parts = get_valid_parts(claim_object)
    if part not in valid_parts:
        part = "unknown"
    output["object_part"] = part

    # 6. Normalize and validate supporting_image_ids
    # Extract valid image IDs from the image_paths
    valid_img_ids = set()
    if image_paths:
        for p in image_paths.split(";"):
            p = p.strip()
            if p:
                img_id = os.path.splitext(os.path.basename(p))[0]
                valid_img_ids.add(img_id)

    supporting = str(output.get("supporting_image_ids", "none")).strip()
    if not supporting or supporting.lower() in ["none", "null", "false"]:
        output["supporting_image_ids"] = "none"
    else:
        support_ids = [s.strip() for s in supporting.split(";") if s.strip()]
        filtered_ids = [s for s in support_ids if s in valid_img_ids]
        if filtered_ids:
            output["supporting_image_ids"] = ";".join(filtered_ids)
        else:
            output["supporting_image_ids"] = "none"

    # 7. Normalize and validate risk_flags, merging with user history risk
    risk_val = str(output.get("risk_flags", "none")).strip()
    risk_set = set()
    
    if risk_val and risk_val.lower() != "none":
        for r in risk_val.split(";"):
            r = r.strip().lower()
            if r in ALLOWED_RISK_FLAGS:
                risk_set.add(r)
                
    # Merge history flags
    if user_history_flags and user_history_flags.lower() != "none":
        for f in user_history_flags.split(";"):
            f = f.strip().lower()
            if f in ALLOWED_RISK_FLAGS:
                risk_set.add(f)
                
    # Clean up empty or none values
    risk_set.discard("none")
    risk_set.discard("")
    
    if not risk_set:
        output["risk_flags"] = "none"
    else:
        output["risk_flags"] = ";".join(sorted(list(risk_set)))
        
    # Ensure evidence_standard_met_reason and claim_status_justification are string
    output["evidence_standard_met_reason"] = str(output.get("evidence_standard_met_reason", "")).strip()
    output["claim_status_justification"] = str(output.get("claim_status_justification", "")).strip()

    return output
