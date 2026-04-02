# permutation-validation
[![GitHub release](https://img.shields.io/github/v/release/wan-huiyan/permutation-validation)](https://github.com/wan-huiyan/permutation-validation/releases) [![Claude Code](https://img.shields.io/badge/Claude_Code-skill-orange)](https://claude.com/claude-code) [![license](https://img.shields.io/github/license/wan-huiyan/permutation-validation)](LICENSE) [![last commit](https://img.shields.io/github/last-commit/wan-huiyan/permutation-validation)](https://github.com/wan-huiyan/permutation-validation/commits)

Validate causal inference model results using empirical permutation tests.

A Claude Code skill that checks whether a causal impact p-value (from BSTS, tfcausalimpact, CausalPy, etc.) is trustworthy by running the same model on many random intervention dates and comparing the real effect to the resulting null distribution.

## What it does

- Generates random (fake) intervention dates with proper exclusion zones around the real treatment
- Runs the exact same model specification on each random date (same covariates, masking, seasonality)
- Computes an empirical permutation p-value: proportion of random dates producing effects >= the real effect
- Provides a GO/NO-GO recommendation with an interpretation table
- Reports model-based and permutation p-values side by side for honest stakeholder communication

## When to use

- After getting a model-based p-value < 0.10 that you plan to present to stakeholders
- When you have done covariate engineering or pre-period masking (both inflate overfit risk)
- When the number of covariates is > 3 relative to the training period
- When comparing model specifications to find the most robust one
- When a BSTS or CausalPy result "seems too good" and you want an empirical sanity check

## Quick start

Install the skill, then ask Claude Code to validate your results:

```
You: Run a permutation test on my BSTS results

Claude: I'll run a permutation test to validate your BSTS causal impact estimate.
       Let me set up 50 random intervention dates, excluding a buffer zone around
       your real treatment date, and run the same model spec on each one...

       Results:
       Model-based (BSTS):    p = 0.040  (internal model uncertainty)
       Permutation (N=50):    p = 0.059  (empirical -- 3/50 random dates matched)

       Interpretation: MODERATE -- the effect is unusual but not extreme under the
       null distribution. Present with honest framing: "~6% chance a random window
       would show an effect this large."
```

## Installation

```bash
claude install-skill github:wan-huiyan/permutation-validation
```

## Key insights encoded in this skill

- **Masking inflates overconfidence.** Wide pre-period masks (e.g., Nov-Jan) make the model think data is smoother than it is. Narrower masks (e.g., BF week to Jan 5) produce more honest permutation p-values.
- **Too many covariates = overfitting.** Decomposed retail covariates (7 components) can overfit training data. 3-5 orthogonal covariates outperform 7+ on permutation robustness.
- **Weather covariates improve discrimination.** Exogenous covariates like temperature help the model explain variance at random dates, making the real intervention stand out more clearly.
- **Pre-BF spikes should NOT be masked.** Revenue spikes before Black Friday teach the model that spikes can happen naturally, preventing it from treating every post-period spike as "significant."

## Scaling with Cloud Run

For N=50+ permutations, use GCP Cloud Run Jobs for parallel execution. Each task picks one random date via `CLOUD_RUN_TASK_INDEX`, runs the model, and writes the result to GCS. Typical cost: ~$0.15/permutation (~$7.50 for 50 runs).

## Limitations

- Designed for time series causal inference (BSTS, tfcausalimpact, CausalPy). Not for A/B test design, general bootstrap CIs, cross-validation, or statistical power analysis.
- Requires sufficient pre-intervention data to generate meaningful random dates (at least 180 days recommended).
- Permutation tests with N < 30 have limited resolution -- the minimum achievable p-value is 1/N.

## Related skills

- [causal-impact-campaign](https://github.com/wan-huiyan/causal-impact-campaign) -- Measure causal impact of marketing campaigns using Bayesian structural time series. Use this skill first, then validate with permutation-validation.
- [cloud-run-batch-experiment](https://github.com/wan-huiyan/cloud-run-batch-experiment) -- Deploy batch experiments on GCP Cloud Run Jobs. Pairs with this skill for scaling permutation runs to 50-100+ parallel tasks.

## License

MIT
