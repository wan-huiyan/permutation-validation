---
name: permutation-validation
version: "1.1.0"
description: |
  Validate causal inference model results using empirical permutation tests. Use this skill whenever
  the user wants to check if a causal impact p-value is trustworthy, run a permutation or
  randomization test, validate BSTS/CausalPy/tfcausalimpact results against random intervention
  dates, detect model overfitting in time series causal inference, or assess how robust a causal
  estimate is. Trigger on phrases like "permutation test", "validate p-value", "is this result
  overfit", "run random dates", "empirical significance", "how robust is this result", "placebo
  test with random dates", "are my BSTS results reliable", "the p-value seems too good",
  "validate my causal impact analysis", or "randomization inference". Also trigger when the user
  has completed a causal impact analysis and wants to stress-test their findings before presenting
  to stakeholders. This skill is specifically about comparing a real causal effect estimate against
  a null distribution built from fake intervention dates — it is NOT for general hypothesis testing,
  bootstrap confidence intervals, or A/B test power analysis.
  NOT for: A/B test design, general bootstrap CIs, cross-validation for prediction models,
  or statistical power analysis.
input: |
  - A causal impact model specification (covariates, masking, seasonality settings)
  - Time series data (CSV or DataFrame) with target metric and covariates
  - The real intervention date and post-period length
  - The model-based p-value to validate (e.g., BSTS p=0.04)
output: |
  - Permutation p-value (proportion of random dates producing effects >= real effect)
  - Distribution plot of null effects vs real effect
  - Comparison table: model p-value vs permutation p-value
  - GO/NO-GO recommendation for presenting the result
---

# Permutation Validation for Causal Inference

This skill validates causal impact estimates by running the same model specification on many
random (fake) intervention dates and comparing the real effect to the resulting null distribution.
The core insight: a model-based p-value (like BSTS p=0.04) can be misleadingly low if the model
is overfit to the data — the permutation test is the honest empirical check.

## Why This Matters

Model-based p-values from BSTS (tfcausalimpact) reflect the model's *internal* uncertainty, which
can be overconfident when:
- The model has many covariates relative to the training period
- Pre-period masking removes high-variance segments, making the model think data is "cleaner" than it is
- Seasonality settings (e.g., nseasons=14) absorb variance that should stay in the residuals

A permutation test asks a fundamentally different question: "If I pick a random 4-day window and
run the exact same model, how often do I get an effect estimate as large as the real one?"

If the answer is "often" (permutation p > 0.15), then the model-based p-value is not trustworthy —
you're seeing a pattern that the model would find anywhere in the data.

### Why compare EFFECT SIZES, not p-values (CRITICAL)

The permutation test MUST compare absolute effect sizes (`|abs_eff|`), NOT model p-values.
This follows the established causal inference literature:

- **Abadie et al. (2010 JASA, 2021 JEL)**: Synthetic control permutation uses the post/pre MSPE
  *ratio* (an effect-size statistic), not model p-values
- **Linden (2018, J Eval Clin Pract)**: ITS permutation compares the *magnitude* of trend changes
- **Young (2019, QJE)**: Model p-values inflate under misspecification; randomization inference
  using effect estimates is robust

**The mechanism**: A model p-value = effect / estimated_uncertainty. When uncertainty is systematically
underestimated (e.g., BSTS VI/HMC/Prophet all show 35-55% FPR on daily retail data), the p-value is
distorted at every permuted date equally. Comparing p-values therefore inherits the model's FPR.
Effect-size comparison is immune because the inflation cancels in the relative ranking.

**Empirical evidence**: In one engagement, comparing p-values showed 0/15 specs passing permutation.
Switching to effect-size comparison on the same data recovered spec 348 at perm p=0.032.
The difference is not statistical noise — it's a fundamental methodological choice.

## When to Use

- After getting a model-based p-value < 0.10 that you plan to present to stakeholders
- When you've done covariate engineering or pre-period masking (both inflate overfit risk)
- When the number of covariates is > 3 relative to the training period
- When comparing model specifications to find the most robust one

## The Permutation Test Protocol

### Step 1: Define the Null Experiment

```python
import numpy as np
import pandas as pd

def generate_random_dates(df, real_intervention_date, post_period_days, n_permutations=50,
                          min_pre_days=180, min_post_days=None, seed=42):
    """Generate random intervention dates for the permutation test.

    Key constraints:
    - Each random date must leave at least min_pre_days before it
    - Each random date must leave at least post_period_days after it
    - Exclude dates within 2x post_period_days of the real intervention
      (to avoid contamination from the actual treatment effect)
    """
    if min_post_days is None:
        min_post_days = post_period_days

    rng = np.random.default_rng(seed)

    dates = pd.to_datetime(df['date'])
    earliest = dates.min() + pd.Timedelta(days=min_pre_days)
    latest = dates.max() - pd.Timedelta(days=min_post_days)

    # Exclusion zone around the real intervention
    exclusion_start = real_intervention_date - pd.Timedelta(days=2 * post_period_days)
    exclusion_end = real_intervention_date + pd.Timedelta(days=2 * post_period_days)

    candidates = dates[(dates >= earliest) & (dates <= latest) &
                       ~((dates >= exclusion_start) & (dates <= exclusion_end))]

    if len(candidates) < n_permutations:
        print(f"Warning: only {len(candidates)} candidate dates available, "
              f"requested {n_permutations}")
        n_permutations = len(candidates)

    selected = rng.choice(candidates, size=n_permutations, replace=False)
    return sorted(pd.to_datetime(selected))
```

### Step 2: Run Each Permutation

For each random date, run the **exact same model specification** as the real analysis:
- Same covariates
- Same masking logic (but shifted to the random date's context)
- Same seasonality settings (nseasons, etc.)
- Same pre-period length calculation

```python
from causalimpact import CausalImpact

def run_single_permutation(df, fake_date, post_days, covariates, nseasons=7,
                           mask_config=None):
    """Run one permutation with the same model spec as the real analysis.

    Returns the absolute cumulative effect estimate.
    """
    pre_start = df['date'].min()
    pre_end = fake_date - pd.Timedelta(days=1)
    post_end = fake_date + pd.Timedelta(days=post_days - 1)

    # Apply masking if configured (shift mask windows relative to fake_date)
    analysis_df = df[['date', TARGET] + covariates].copy()
    if mask_config:
        analysis_df = apply_mask(analysis_df, mask_config, fake_date)

    # Build the model input
    ci_df = analysis_df.set_index('date')[[TARGET] + covariates]
    pre_period = [str(pre_start.date()), str(pre_end.date())]
    post_period = [str(fake_date.date()), str(post_end.date())]

    try:
        ci = CausalImpact(ci_df, pre_period, post_period, nseasons=nseasons)
        summary = ci.summary_data
        return abs(summary['average']['abs_effect'] * post_days)
    except Exception as e:
        print(f"  Permutation {fake_date.date()} failed: {e}")
        return None
```

### Step 3: Compute the Permutation P-value

```python
def compute_permutation_pvalue(real_effect, null_effects):
    """Two-sided permutation p-value.

    p = (number of null effects >= |real_effect|) / total permutations

    This is deliberately conservative — it counts how many random dates
    produced effects AS LARGE OR LARGER than the real one.
    """
    valid = [e for e in null_effects if e is not None]
    n_extreme = sum(1 for e in valid if e >= abs(real_effect))
    p = n_extreme / len(valid)
    return p, len(valid)
```

### Step 4: Interpret the Results

| Permutation p | Interpretation | Recommendation |
|---|---|---|
| < 0.05 | Strong: real effect is very unlikely under null | Present with confidence |
| 0.05 - 0.10 | Moderate: effect is unusual but not extreme | Present with honest framing ("~X% chance a random window would show this") |
| 0.10 - 0.20 | Weak: model may be finding patterns everywhere | Investigate model spec, try fewer covariates |
| > 0.20 | Not robust: model-based p-value is overconfident | Do NOT present model-based p-value as reliable |

### Step 5: Report Both P-values

Always report model-based and permutation p-values side by side:

```
Model-based (BSTS):    p = 0.040  (internal model uncertainty)
Permutation (N=50):    p = 0.059  (empirical — 2/50 random dates matched)
```

The permutation p-value is the one that should drive the headline claim.

## Scaling with Cloud Run

For N=50+ permutations, running locally takes hours. Use Cloud Run Jobs for parallelism:

```bash
# One job per permutation date
gcloud run jobs execute permutation-test \
  --region europe-west2 \
  --tasks 50 \
  --set-env-vars "SPEC_ID=A2.4,N_PERMUTATIONS=50" \
  --timeout 3600
```

Each task reads `CLOUD_RUN_TASK_INDEX` to pick its assigned random date, runs the model, and
writes the effect estimate to GCS. A collector script then aggregates results.

Typical cost: ~$0.15 per permutation on Cloud Run (2 vCPU, 4GB RAM, ~2min each).
50 permutations ≈ $7.50, 100 permutations ≈ $15.

## Common Pitfalls

### 1. Masking inflates overconfidence
Wide pre-period masks (e.g., mask_nov_jan = Nov 1 to Jan 31) remove high-variance data, making
the model think the data is smoother than it really is. This produces tight credible intervals
and low model-based p-values — but the permutation test reveals the truth.

**Fix:** Use narrower masks (e.g., mask_xmas = BF week to Jan 5 only). In our experience,
switching from mask_nov_jan to mask_xmas moved permutation p from >0.15 to 0.059-0.098.

### 2. Too many covariates = overfitting
Decomposed retail covariates (7 separate holiday components) can overfit the training data.
The model learns to explain every bump, leaving no residual variance — which makes ANY
post-period deviation look "significant."

**Fix:** Use fewer, more orthogonal covariates. In practice, 3-5 covariates outperform 7+
on permutation robustness.

### 3. Weather covariates improve discrimination
Exogenous covariates like temperature and precipitation help the model explain variance at
random dates (which tend to fall in unremarkable weather periods), making the real intervention
date stand out more clearly. In our testing, adding weather moved permutation p from 0.098 to 0.059.

### 4. Pre-BF spikes should NOT be masked
Revenue spikes before Black Friday (z-scores +10 to +13) are informative — they teach the model
that revenue CAN spike for non-promo reasons. Masking them makes the model think spikes are
abnormal, which inflates the significance of any post-period spike.

## Placebo Test (Complementary Validation)

Run the full model on a pre-intervention window where no campaign occurred:

```python
# Use the 4 days before the real intervention as a "fake" post-period
placebo_start = real_intervention_date - pd.Timedelta(days=8)
placebo_end = real_intervention_date - pd.Timedelta(days=5)
```

A good model should show p > 0.20 on the placebo test (no false positive). If the placebo test
shows p < 0.10, the model is finding effects where none exist — likely overfit.

## SCA Permutation Validation (Specification Curve Analysis)

When running a full SCA (hundreds of specs across covariate bundles and pre-period modes),
permutation-validate the **top N specs** rather than all specs to keep compute manageable.

**Recommended approach:**
- Select top 50 specs by p-value from the SCA results
- Run 10 permutation shuffles per spec = 500 parallel tasks on Cloud Run
- Add a `perm_p` column to the SCA detail table
- Flag any spec where perm_p > 0.10 as potentially spurious

**Why top-N instead of all specs:**
- Full SCA × full permutations (e.g., 448 × 50 = 22,400 tasks) is prohibitively expensive
- The bottom-ranked specs (p > 0.30) are already non-significant — permutation adds nothing
- The top specs are the ones you'd actually report to stakeholders

**Short pre-period caveat:**
When the best specs cluster in a short pre-period mode (e.g., 54-day post-holiday window),
the permutation space is smaller (fewer candidate fake intervention dates). This reduces
statistical power of the permutation test itself. Use at least 10 shuffles and interpret
conservatively. If perm_p = 0.10 with 10 shuffles, that's 1/10 — borderline. With 50 shuffles
and perm_p = 0.08, that's 4/50 — more robust.

**Pre-period mode comparison:**
Run permutation on the top 3-5 specs from EACH pre-period mode (not just the best overall).
If the short pre-period specs pass permutation AND the full-period specs show directional
consistency (positive effect, even if higher p), that's convergent validity. If the short
pre-period specs pass but the full-period specs show opposite direction, that's a red flag.

**Prior empirical results from retail SCA:**
- Sale signals alone: passed BSTS p-value (0.033) but FAILED permutation (0.140)
- Sale signals + external Trends: passed both (BSTS 0.011, perm 0.080)
- External Trends alone: strongest pass (BSTS 0.008, perm 0.040)
- Lesson: external exogenous signals carry robust causal information; endogenous sale signals
  need external support to pass permutation validation

## Composability

- Use after `causal-impact-campaign` to validate findings
- Pairs with `cloud-run-batch-experiment` for scaling permutation runs
- Feed results into stakeholder deliverables with honest dual-p-value framing
- For SCA: run after `generate_sca_specs()` produces results, before client presentation
