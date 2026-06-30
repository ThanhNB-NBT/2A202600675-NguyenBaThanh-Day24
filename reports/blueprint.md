# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh vien:** Nguyen Ba Thanh  
**Ngay:** 2026-06-30

## Guard Stack Pipeline

| Layer | Tool | Latency P95 | Failure Action |
|---|---|---:|---|
| PII Detection | Local Presidio-compatible regex | <10ms | Reject + log |
| Topic/Jailbreak | Local input rail / NeMo-ready hook | <300ms | 503 + reason |
| RAG Pipeline | Day 18-compatible shim | <2000ms | Fallback |
| Output Check | Local output rail / NeMo-ready hook | <300ms | Block + log |

## CI Gates

- RAGAS faithfulness >= 0.75 on the 50-question set.
- Adversarial suite pass rate >= 90% (18/20) before merge to main.
- P95 total guard latency < 500ms.
- No `# TODO` remains in `src/phase_*.py`.

## Monitoring

| Metric | Current Lab Value | Alert Threshold | Action |
|---|---:|---:|---|
| P95 total guard latency | 0.13ms | >500ms | Profile rail layer |
| Adversarial pass rate | 20/20 | <18/20 | Add attack pattern |
| Worst RAGAS metric | faithfulness | <0.70 | Inspect bottom-10 |
| Dominant failure distribution | factual | n/a | Tune retrieval for that set |

## Production Notes

The lab stack keeps the production gates simple: run Phase A to catch quality
regressions, Phase C to block unsafe inputs, and latency checks to stop slow
guardrails from shipping. In a real deployment, the local keyword rail should
be replaced by the prepared NeMo Guardrails config, but the same JSON report
shape and CI gate can stay unchanged.

## Lab Results

| Result | Value |
|---|---:|
| RAGAS avg score | 1.0 |
| Worst RAGAS metric | faithfulness |
| Dominant failure distribution | factual |
| Cohen kappa | 0.545455 |
| Adversarial pass rate | 20 / 20 |
| Guard P95 latency | 0.13ms |
