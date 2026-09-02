# Probabilistic tools

The `sibyl/forecasting/probabilistic_tools/` package and the tool-runner that exposes it to
forecaster models.


---

## Agent-facing detail (relocated from `AGENTS.md`, 2026-08-26)

The sections below were inline in `AGENTS.md`, which is loaded on every coding-agent
invocation. They are reference material rather than always-true rules, so they live here
now and are read on demand.

Some of this overlaps the prose above — it was written for a different audience and is
kept verbatim rather than merged, so nothing is silently lost. Reconciling the two is
worthwhile follow-up, but it is an editorial pass, not a mechanical one.
### `sibyl/forecasting/probabilistic_tools/`

Reusable probability math — pooling, Beta-Binomial Bayes, percentile → parametric fits (normal/lognormal/Student-t), declared-vs-math consistency checks, Dirichlet CIs, Neg-Bin/Poisson discrete percentiles, exponential/Weibull survival, Gamma-conjugate hazard. `prob_event_before`, `linear_pool` / `log_pool` / `satopaa_extremize`, `beta_binomial_update`, `cdf_at_threshold`, `dirichlet_with_other` are wired into `tool_runner` dispatch. (`poisson_at_least_one` is exported and used inside `mc_discrete.py` / `survival.py`, but is NOT itself dispatched by `tool_runner`.)

Newly-added math (Workstreams D1-D3):

- **Noisy-OR** (`aggregation.py` `noisy_or`): rare-binary decomposition `1 − ∏(1 − pᵢ)` for combining independent failure-mode probabilities. Exported from the package, but NOT currently dispatched by `tool_runner` (no references in `tool_runner.py`) — it is a callable available for future wiring, not an active dispatch path. `TODO(noisy-or-wiring)`: either add a binary Noisy-OR dispatch (when a forecaster declares independent sub-event probabilities) or leave as a library-only helper.
- **Mixture-of-normals** (`mixtures.py`): `MixtureOfNormals` / `MixtureComponent` types, `mixture_cdf`, `fit_mixture_from_percentiles` (multi-start L-BFGS-B with single-normal fallback), and `percentiles_to_metaculus_cdf_via_mixture` (constraint-enforced `PCHIP_CDF_POINTS`-point CDF). The library itself is preserved but currently dormant — the `NumericStructured.mixture_components` schema slot and the router branch that consumed it were removed 2026-07-08 after zero prod fires; percentiles+PCHIP outperformed the mixture path in every benchmark.
- **Gamma waiting-time, conditional-given-survival**: `gamma_prob_event_before` with elapsed-window split (`survival_distributions.py`) — covers the missing waiting-time fitter alongside the existing exponential / Weibull / Gamma-hazard variants.

