---
name: permutation-validation
version: "2.0.0"
description: |
  DEPRECATED — merged into causal-impact-campaign v2.0.0.
  Use the causal-impact-campaign skill instead, which now includes all permutation
  validation methodology (effect-size comparison, code templates, SCA permutation protocol,
  Cloud Run scaling, and common pitfalls).
  Trigger words that previously matched this skill ("permutation test", "validate p-value",
  "empirical significance") now route to causal-impact-campaign.
deprecated: true
replaced_by: causal-impact-campaign
---

# DEPRECATED — Use causal-impact-campaign instead

This skill has been merged into `causal-impact-campaign` v2.0.0 (Step 5: Validate).

All content is now in the unified skill:
- Effect-size comparison methodology (Abadie 2010, Linden 2018, Young 2019)
- `generate_random_dates()` and `compute_permutation_pvalue()` code templates
- SCA permutation protocol (per-mode selection, log/raw mix)
- Cloud Run scaling guidance
- Common pitfalls (masking inflation, covariate count, NaN handling)
