import streamlit as st
import pandas as pd
import os
import json
import sys
import csv
import base64
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClaimVerify AI – Evidence Review Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main { background: #0f1117; }
.block-container { padding: 1.5rem 2.5rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1d2e 0%, #141627 100%);
    border-right: 1px solid #2a2d3e;
}
[data-testid="stSidebar"] * { color: #e0e0f0 !important; }

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #1e2130 0%, #262a3f 100%);
    border: 1px solid #3a3f5c;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    margin-bottom: 0.6rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.5);
}
.metric-value {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #7c6fff, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.78rem;
    color: #8890b5;
    font-weight: 500;
    margin-top: 0.4rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Status badges */
.badge-supported { background: #1a472a; color: #4ade80; border: 1px solid #166534; }
.badge-contradicted { background: #4a1919; color: #f87171; border: 1px solid #7f1d1d; }
.badge-not_enough_information { background: #3a3010; color: #fbbf24; border: 1px solid #78350f; }
.badge-true { background: #1a3a4a; color: #38bdf8; border: 1px solid #0c4a6e; }
.badge-false { background: #3a2010; color: #fb923c; border: 1px solid #7c2d12; }
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}

/* Claim card */
.claim-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #1e2235 100%);
    border: 1px solid #2e3354;
    border-radius: 14px;
    padding: 1.4rem;
    margin: 0.8rem 0;
}
.claim-transcript {
    background: #111320;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.84rem;
    color: #c0c8e8;
    line-height: 1.6;
    max-height: 200px;
    overflow-y: auto;
}
.image-caption {
    background: #0d1020;
    border: 1px solid #2a2d4a;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    font-size: 0.78rem;
    color: #8890b5;
    line-height: 1.5;
    margin-top: 0.4rem;
}
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e0e0f0;
    margin: 1.4rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #3a3f5c;
}
.sub-header {
    font-size: 0.85rem;
    font-weight: 600;
    color: #a0a8cc;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.4rem;
}
.risk-tag {
    display: inline-block;
    background: #2d1a0d;
    border: 1px solid #7c3f0d;
    color: #fb923c;
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    font-size: 0.72rem;
    font-weight: 500;
    margin: 0.1rem 0.1rem;
}
.logo-text {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #7c6fff, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ─────────────────────────────────────────────────────────
def make_badge(value, badge_type=None):
    if badge_type is None:
        badge_type = str(value).lower().replace(" ", "_")
    cls = f"badge-{badge_type}"
    return f'<span class="badge {cls}">{value}</span>'

def load_image_b64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

def resolve_path(p):
    p = p.strip()
    if os.path.exists(p):
        return p
    c = os.path.join("dataset", p)
    if os.path.exists(c):
        return c
    return p

def load_output_csv(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def load_sample_ground_truth():
    path = "dataset/sample_claims.csv"
    if not os.path.exists(path):
        return {}
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt[row["user_id"] + "|" + row["image_paths"]] = row
    return gt

def load_cache(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"vision_cache": {}, "llm_cache": {}}

# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="logo-text">🔍 ClaimVerify AI</div>', unsafe_allow_html=True)
    st.caption("Multi-Modal Evidence Review System")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Overview & Metrics", "🔎 Claim Explorer", "🧪 Live Playground"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Data**")
    output_csv_path = st.text_input("Predictions CSV", value="output.csv", label_visibility="collapsed")
    sample_cache_path = st.text_input("Cache file", value=".cache_sample_eval.json", label_visibility="collapsed")
    st.divider()
    st.caption("Powered by BLIP-Large + Groq Llama-3.3-70B")

# ── Load data ────────────────────────────────────────────────────────
predictions = load_output_csv(output_csv_path)
sample_gt = load_sample_ground_truth()
cache = load_cache(sample_cache_path)

# ── PAGE: Overview & Metrics ─────────────────────────────────────────
if "Overview" in page:
    st.markdown('<div class="section-header">📊 Dashboard Overview</div>', unsafe_allow_html=True)

    if predictions is None:
        st.warning(f"No predictions file found at `{output_csv_path}`. Run `python code/main.py` first.", icon="⚠️")
    else:
        df = pd.DataFrame(predictions)

        # KPI row
        total = len(df)
        supported = (df["claim_status"] == "supported").sum()
        contradicted = (df["claim_status"] == "contradicted").sum()
        not_enough = (df["claim_status"] == "not_enough_information").sum()
        high_risk = df["risk_flags"].apply(lambda x: "user_history_risk" in str(x) or "manual_review_required" in str(x)).sum()

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, label, value in [
            (c1, "Total Claims", total),
            (c2, "✅ Supported", supported),
            (c3, "❌ Contradicted", contradicted),
            (c4, "❓ Not Enough Info", not_enough),
            (c5, "🚩 High-Risk Flags", high_risk)
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Distribution Charts</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="sub-header">Claim Status</div>', unsafe_allow_html=True)
            status_counts = df["claim_status"].value_counts()
            st.bar_chart(status_counts, color="#7c6fff", height=220)

        with col2:
            st.markdown('<div class="sub-header">Issue Type</div>', unsafe_allow_html=True)
            issue_counts = df["issue_type"].value_counts().head(8)
            st.bar_chart(issue_counts, color="#a78bfa", height=220)

        with col3:
            st.markdown('<div class="sub-header">Severity</div>', unsafe_allow_html=True)
            sev_counts = df["severity"].value_counts()
            st.bar_chart(sev_counts, color="#c084fc", height=220)

        col4, col5 = st.columns(2)
        with col4:
            st.markdown('<div class="sub-header">Claim Object Type</div>', unsafe_allow_html=True)
            obj_counts = df["claim_object"].value_counts()
            st.bar_chart(obj_counts, color="#818cf8", height=200)

        with col5:
            st.markdown('<div class="sub-header">Evidence Standard Met</div>', unsafe_allow_html=True)
            ev_counts = df["evidence_standard_met"].value_counts()
            st.bar_chart(ev_counts, color="#6366f1", height=200)

        # Sample accuracy vs ground truth
        if sample_gt:
            st.markdown('<div class="section-header">📈 Accuracy vs Ground Truth (Sample)</div>', unsafe_allow_html=True)
            scored_cols = ["claim_status", "issue_type", "object_part", "severity", "evidence_standard_met", "valid_image"]
            sample_preds = load_output_csv("code/evaluation/sample_predictions.csv")
            if sample_preds:
                sample_df = pd.DataFrame(sample_preds)
                sample_rows = []
                with open("dataset/sample_claims.csv", "r", encoding="utf-8") as f:
                    sample_rows = list(csv.DictReader(f))
                acc_data = {}
                for col in scored_cols:
                    correct = sum(
                        1 for p, g in zip(sample_preds, sample_rows)
                        if str(p.get(col, "")).lower() == str(g.get(col, "")).lower()
                        and str(g.get(col, "")).strip() != ""
                    )
                    total_col = sum(1 for g in sample_rows if str(g.get(col, "")).strip() != "")
                    acc_data[col] = round(correct / total_col * 100, 1) if total_col > 0 else 0
                acc_df = pd.DataFrame({"Field": list(acc_data.keys()), "Accuracy (%)": list(acc_data.values())})
                acc_df = acc_df.set_index("Field")
                st.bar_chart(acc_df, color="#34d399", height=280)

        # Raw table
        st.markdown('<div class="section-header">📋 Predictions Table</div>', unsafe_allow_html=True)
        display_cols = ["user_id", "claim_object", "claim_status", "issue_type", "object_part", "severity", "evidence_standard_met", "valid_image", "risk_flags"]
        st.dataframe(
            df[display_cols],
            use_container_width=True,
            height=300
        )

# ── PAGE: Claim Explorer ─────────────────────────────────────────────
elif "Explorer" in page:
    st.markdown('<div class="section-header">🔎 Claim Explorer</div>', unsafe_allow_html=True)

    if predictions is None:
        st.warning(f"No predictions file found at `{output_csv_path}`. Run `python code/main.py` first.", icon="⚠️")
    else:
        df = pd.DataFrame(predictions)

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_obj = st.selectbox("Object Type", ["All"] + sorted(df["claim_object"].unique().tolist()))
        with col2:
            filter_status = st.selectbox("Claim Status", ["All"] + sorted(df["claim_status"].unique().tolist()))
        with col3:
            filter_severity = st.selectbox("Severity", ["All"] + sorted(df["severity"].unique().tolist()))

        filtered = df.copy()
        if filter_obj != "All":
            filtered = filtered[filtered["claim_object"] == filter_obj]
        if filter_status != "All":
            filtered = filtered[filtered["claim_status"] == filter_status]
        if filter_severity != "All":
            filtered = filtered[filtered["severity"] == filter_severity]

        st.caption(f"Showing {len(filtered)} of {len(df)} claims")

        # Claim detail view
        if len(filtered) > 0:
            claim_labels = [f"[{i+1}] {row['user_id']} — {row['claim_object']} — {row['claim_status']}"
                           for i, row in filtered.iterrows()]
            selected_label = st.selectbox("Select a claim to inspect:", claim_labels)
            sel_idx = claim_labels.index(selected_label)
            claim = filtered.iloc[sel_idx]

            st.markdown(f"""<div class="claim-card">""", unsafe_allow_html=True)
            col1, col2 = st.columns([1.5, 2])

            with col1:
                st.markdown(f'<div class="sub-header">User & Object</div>', unsafe_allow_html=True)
                st.markdown(f"**User:** `{claim['user_id']}` | **Object:** `{claim['claim_object']}`")
                st.markdown("**Claim Status:**", unsafe_allow_html=True)
                st.markdown(make_badge(claim['claim_status']), unsafe_allow_html=True)

                cols_a, cols_b = st.columns(2)
                with cols_a:
                    st.metric("Issue Type", claim.get("issue_type", "unknown"))
                    st.metric("Severity", claim.get("severity", "unknown"))
                with cols_b:
                    st.metric("Object Part", claim.get("object_part", "unknown"))
                    st.metric("Evidence Met", claim.get("evidence_standard_met", "false"))

                risk_flags_str = claim.get("risk_flags", "none")
                if risk_flags_str and risk_flags_str != "none":
                    st.markdown('<div class="sub-header" style="margin-top:0.8rem">Risk Flags</div>', unsafe_allow_html=True)
                    flags_html = " ".join([f'<span class="risk-tag">{f}</span>' for f in risk_flags_str.split(";")])
                    st.markdown(flags_html, unsafe_allow_html=True)

                st.markdown(f'<div class="sub-header" style="margin-top:0.8rem">Evidence Reason</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="image-caption">{claim.get("evidence_standard_met_reason", "")}</div>', unsafe_allow_html=True)

                st.markdown(f'<div class="sub-header" style="margin-top:0.8rem">Justification</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="image-caption">{claim.get("claim_status_justification", "")}</div>', unsafe_allow_html=True)

            with col2:
                st.markdown(f'<div class="sub-header">Claim Transcript</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="claim-transcript">{claim["user_claim"].replace(" | ", "<br>")}</div>', unsafe_allow_html=True)

                # Images
                img_paths = [p.strip() for p in claim["image_paths"].split(";") if p.strip()]
                if img_paths:
                    st.markdown(f'<div class="sub-header" style="margin-top:0.8rem">Submitted Images & Visual Descriptions</div>', unsafe_allow_html=True)
                    img_cols = st.columns(min(len(img_paths), 3))
                    for i, img_p in enumerate(img_paths):
                        resolved = resolve_path(img_p)
                        img_id = os.path.splitext(os.path.basename(img_p))[0]
                        with img_cols[i % 3]:
                            if os.path.exists(resolved):
                                st.image(resolved, caption=f"Image ID: {img_id}", use_container_width=True)
                            else:
                                st.info(f"Image not found: {img_p}")

                            # Get BLIP descriptions from cache
                            descs = cache["vision_cache"].get(resolved, {})
                            if descs:
                                desc_lines = "\n".join([f"• {v}" for v in descs.values()])
                                st.markdown(f'<div class="image-caption"><b>BLIP Descriptions:</b><br>{desc_lines}</div>',
                                           unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

# ── PAGE: Live Playground ─────────────────────────────────────────────
elif "Playground" in page:
    st.markdown('<div class="section-header">🧪 Live Claim Playground</div>', unsafe_allow_html=True)
    st.caption("Enter a custom claim and run the full verification pipeline interactively.")

    with st.form("playground_form"):
        col1, col2 = st.columns(2)
        with col1:
            pg_claim_object = st.selectbox("Claim Object", ["car", "laptop", "package"])
            pg_user_id = st.text_input("User ID (optional)", value="user_test_001")
        with col2:
            pg_user_history = st.text_input("User History Summary (optional)",
                                            value="First-time claimant with no prior history.")
            pg_history_flags = st.text_input("History Flags (e.g. none or user_history_risk)", value="none")

        pg_user_claim = st.text_area(
            "Claim Conversation",
            value="Customer: I dropped my laptop and now the screen has a crack. | Support: Can you upload a photo?",
            height=120
        )
        pg_image_path = st.text_input("Image Path (relative to repo root, e.g. dataset/images/sample/case_009/img_1.jpg)",
                                      value="dataset/images/sample/case_009/img_1.jpg")

        submit_btn = st.form_submit_button("🚀 Run Verification Pipeline", use_container_width=True)

    if submit_btn:
        code_dir = os.path.join(os.getcwd(), "code")
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)

        with st.spinner("Running local BLIP vision model…"):
            try:
                import vision
                resolved_img = resolve_path(pg_image_path)
                image_descs = vision.get_image_descriptions(resolved_img)
                st.success("✅ Visual descriptions generated!")

                if os.path.exists(resolved_img):
                    st.image(resolved_img, caption="Submitted Image", width=350)

                desc_lines = "\n".join([f"• **{k}**: {v}" for k, v in image_descs.items()])
                st.markdown(f'<div class="image-caption"><b>BLIP Descriptions:</b><br>{desc_lines}</div>',
                           unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Vision model error: {e}")
                image_descs = {pg_image_path: {"unconditional": f"Error: {e}"}}

        with st.spinner("Reasoning with Groq Llama-3.3-70B…"):
            try:
                import llm
                import guardrails

                evidence_reqs_str = (
                    "The claimed object and relevant part should be visible clearly enough to inspect the claimed condition."
                )
                raw_pred = llm.evaluate_claim_llm(
                    claim_object=pg_claim_object,
                    user_claim=pg_user_claim,
                    user_history_summary=pg_user_history,
                    user_history_flags=pg_history_flags,
                    evidence_requirements=evidence_reqs_str,
                    image_descriptions={pg_image_path: image_descs}
                )
                final_pred = guardrails.postprocess_and_guard(
                    prediction=raw_pred,
                    image_paths=pg_image_path,
                    claim_object=pg_claim_object,
                    user_history_flags=pg_history_flags
                )

                st.markdown('<div class="section-header">🎯 Verification Results</div>', unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    status = final_pred.get("claim_status", "unknown")
                    st.markdown(f"**Claim Status:** {make_badge(status)}", unsafe_allow_html=True)
                    st.markdown(f"**Issue Type:** `{final_pred.get('issue_type', 'unknown')}`")
                with col2:
                    st.markdown(f"**Object Part:** `{final_pred.get('object_part', 'unknown')}`")
                    st.markdown(f"**Severity:** `{final_pred.get('severity', 'unknown')}`")
                with col3:
                    ev = final_pred.get("evidence_standard_met", False)
                    ev_str = str(ev).lower()
                    st.markdown(f"**Evidence Met:** {make_badge(ev_str, ev_str)}", unsafe_allow_html=True)
                    st.markdown(f"**Valid Image:** `{final_pred.get('valid_image', 'unknown')}`")

                st.markdown(f"**Supporting Images:** `{final_pred.get('supporting_image_ids', 'none')}`")

                risk_flags = final_pred.get("risk_flags", "none")
                if risk_flags != "none":
                    flags_html = " ".join([f'<span class="risk-tag">{f}</span>' for f in risk_flags.split(";")])
                    st.markdown(f"**Risk Flags:** {flags_html}", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="claim-card" style="margin-top: 1rem;">
                    <div class="sub-header">Evidence Reason</div>
                    <div class="claim-transcript">{final_pred.get('evidence_standard_met_reason', '')}</div>
                    <div class="sub-header" style="margin-top: 0.8rem">Justification</div>
                    <div class="claim-transcript">{final_pred.get('claim_status_justification', '')}</div>
                </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"LLM/Guardrails error: {e}")
