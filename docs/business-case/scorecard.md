# Use-case scorecard — FSI / transaction foundation model

> Worked example, scored retrospectively.

- **Accelerator:** `cloudera-forge-fsi-foundational-model`
- **Sourcing input:** analyst & partner drivers (NVIDIA TFM blueprint; VAST partnership)
  + competitive gap (foundation-model fraud stories)
- **Type:** greenfield
- **Business KPI moved:** fraud detection lift (AUC / average precision) at constant
  review budget

## Weighted score

| Criterion | Weight | Score (1–5) | Weighted | Notes |
|-----------|:------:|:-----------:|:--------:|-------|
| Business value / KPI impact | 0.25 | 4 | 1.00 | Embedding lift over raw features is the headline, measured live |
| Data availability | 0.20 | 4 | 0.80 | Public transaction data + fetched TFM checkpoint; VAST target needed |
| Technical feasibility | 0.20 | 4 | 0.80 | Full GPU path proven; DEMO-FALLBACK mode de-risks demos |
| Repeatability across accounts | 0.20 | 4 | 0.80 | Any FSI + VAST/Cloudera AI account; GPU runtime required |
| Strategic / competitive fit | 0.15 | 5 | 0.75 | NVIDIA + VAST + Cloudera joint story |
| **Total** | **1.00** | | **4.15 / 5** | ≥ 4.0 advances |

**Decision:** 🟢 Greenlight
**Knockout triggered?** no
