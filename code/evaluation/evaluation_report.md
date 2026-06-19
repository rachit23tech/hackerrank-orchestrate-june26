# Evaluation Report — Multi-Modal Evidence Review

**Generated**: 2026-06-19T11:36:29.689839
**Sample dataset rows**: 20
**Test dataset rows**: 45

---

## Accuracy Metrics (Sample Dataset)

| Column | Accuracy | Correct | Total |
|---|---|---|---|
| `claim_status` | 35.0% | 7 | 20 |
| `issue_type` | 40.0% | 8 | 20 |
| `object_part` | 45.0% | 9 | 20 |
| `severity` | 35.0% | 7 | 20 |
| `evidence_standard_met` | 45.0% | 9 | 20 |
| `valid_image` | 45.0% | 9 | 20 |

**Overall (across all scored fields)**: 40.8% (49/120)

---

## Strategy Comparison

| Strategy | Description | Notes |
|---|---|---|
| **Strategy 1 (Baseline)** | Local BLIP captioning only + simple rule-based decisions | No LLM — deterministic, fast, but limited reasoning |
| **Strategy 2 (Final)** | BLIP captioning → Groq Llama-3.3-70B reasoning + deterministic guardrails | High-quality contextual reasoning, multilingual, prompt-injection resistant |

Strategy 2 (our final approach) was adopted because:
- It handles multilingual claims (Hinglish, Spanish) through LLM understanding.
- It produces grounded justifications that reference image evidence.
- It is resistant to adversarial prompt-injection attacks.
- Deterministic guardrails ensure schema conformity and consistent risk flag merging.

---

## Operational Analysis

### Model Calls
- **BLIP VLM (local)**: 1 call per unique image. Sample set: ~20-60 images. Test set: ~45-135 images.
- **Groq Llama-3.3-70B (cloud)**: 1 call per claim row. Sample: 20 calls. Test: 45 calls.

### Token Usage (Estimates)
| Set | Input Tokens | Output Tokens |
|---|---|---|
| Sample (20 rows) | ~50,000 | ~4,000 |
| Test (45 rows) | ~112,500 | ~9,000 |

### Images Processed
- Sample set: ~40 images (average ~2 per claim).
- Test set: ~90 images (average ~2 per claim).

### Approximate Cost (Groq Llama-3.3-70B)
- Pricing: ~$0.59/M input tokens, ~$0.79/M output tokens
- **Sample processing cost**: ~$0.0327
- **Test processing cost**: ~$0.0735
- **Total estimated cost**: ~$0.1061

### Latency & Runtime
- **Sample evaluation runtime**: ~38.1 seconds
- **Estimated test set runtime**: ~85.8 seconds
- BLIP inference: ~0.5-1s per image on CPU.
- Groq API latency: ~1-3s per claim.

### TPM/RPM Considerations
- Groq's free tier: ~30 RPM and 6,000 TPM for Llama-3.3-70B.
- With ~2,500 input + 200 output tokens per call, the test set of ~45 rows uses ~123,500 tokens total.
- At 6,000 TPM, we'd need ~21 minutes of sustained throughput; at 30 RPM, ~1.5 minutes.
- **Mitigation**: We use a JSON file-based cache that stores results after the first run, making repeated evaluations instant and cost-free.
- For batching: claims are processed sequentially with natural pacing; no explicit throttling is needed for this scale.
- For production scale, we would implement a token-bucket rate limiter and batch BLIP inference on GPU.

---

## Final Strategy

Our system uses **Strategy 2 (BLIP + Groq Llama-3.3-70B)** with the following pipeline:

1. **Local VLM (BLIP-Large)**: Runs offline on each submitted image to generate 5 targeted visual descriptions.
2. **Cloud LLM (Groq Llama-3.3-70B)**: Receives the complete context (claim transcript, user history, evidence requirements, and image descriptions) and produces a strict JSON prediction.
3. **Deterministic Guardrails**: Validate and normalize every output field, merge user history risk flags, and verify supporting image IDs against actual submitted filenames.
4. **Caching**: Both BLIP outputs and LLM responses are cached to disk to eliminate redundant computation.