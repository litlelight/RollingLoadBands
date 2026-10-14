# Superseded interpretation of legacy resampling labels

**Applies to:** legacy results and reports from E0--E29 kept for audit in this repository, including report and CSV fields such as `95% CI`, `ci_lower_pct`, `ci_upper_pct`, and probability-like tail labels.

**Current authoritative interpretation:** The revised manuscript and Supplementary Materials treat the numerical 2.5th--97.5th percentile endpoints from resampling stored daily indicators or selected daily losses as **conditional descriptive resampling percentile ranges**. They condition on the archived forecasts, realized evaluation calendar and selected decision rules. These ranges **are not** calibrated confidence intervals for the fixed 2025 observed annual total and **do not** provide a guaranteed confidence statement about future-year performance.

The primary paired cost ranges shown in manuscript Table 9 and Figures 4--5 originate from the retained 5,000-replicate seven-day circular-block resampling under `results/ISO_NE_REVISION_V1.0/`. The separately archived E28 annual circular, annual non-circular, and quarter-stratified sensitivities can have slightly different numerical endpoints and should not be silently substituted for the primary run.

**Provenance rule:** Older report and CSV titles and values are preserved for exact historical traceability. This notice supersedes their *inferential wording*, **not** their underlying numerical records. For detailed assumptions and differences in resampling designs, consult manuscript Section 4.4 and Supplementary Sections S7, S13--S15. E30 and E31 add independent follow-up sensitivity evidence.

**Version note:** The immutable v1.3.0 GitHub tag / [Zenodo DOI 10.5281/zenodo.23368356](https://doi.org/10.5281/zenodo.23368356) remains unchanged. These notices were added later to the current `main` branch as documentation updates. Earlier v1.2.0 [DOI 10.5281/zenodo.23064483](https://doi.org/10.5281/zenodo.23064483) identifies the E0--E29 record only.
