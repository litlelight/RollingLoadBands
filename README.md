# Rolling Empirically Calibrated Simultaneous Bands for Next-Day Load Profiles and Cost-Sensitive Peak-Risk Decision Support — Reproducibility Package

This reproducibility package accompanies the revised manuscript. It contains the frozen processed ISO-NE and NYISO model inputs, source code, configurations, selected earlier-revision result stores required by the second-round analyses, and the complete E24--E29 reviewer analyses.

**Package version: 1.2.0.** Any earlier v1.1.0 archival identifiers refer to the preceding revision and do not include the E24--E29 analyses. The archival DOI for version 1.2.0 should be added to the repository metadata and manuscript after the corresponding deposit is completed.

## Contents

- `outputs/01_data_audit/`: four frozen profile/feature Parquet inputs, feature manifest, and preprocessing audits.
- `experiments/`, `configs/`, and the three root runners: E0--E19, E20--E23, and E24--E29 implementations with frozen settings.
- `results/ISO_NE_REVISION_V1.0/` and `results/ISO_NE_REVISION_V1.1_CLOSURE/`: selected preceding results, including the frozen out-of-sample prediction/band stores and 2025 decision cubes required for a direct E24--E29 rerun.
- `results/ROUND2_REVISION_V1/`: complete E24--E29 CSV/Parquet traces, including the 365-day decision-matrix audit and NYISO target audit.
- `reports/`: human-readable experiment reports and earlier figures.
- `MANIFEST.sha256`, `verify_package.py`, and `generate_manifest.py`: byte-level distribution checks. The manifest covers all distributed files except itself.

To keep the repository compact, regenerable E0--E23 auxiliary diagnostic and runtime tables are not redistributed here. This does **not** remove any E24--E29 file or change any included scientific result value. The earlier runners regenerate the omitted auxiliaries when their raw-input prerequisites are available. The frozen predecessor stores listed above are distributed so that E24--E29 can be rerun without first repeating the complete E0--E19 workflow.

## Verify before running

From the extracted repository root:

```bash
python verify_package.py
```

The verifier checks every listed byte hash, rejects unlisted distributed files, confirms the nine E24 frozen-input fingerprints and the E24--E29 output hashes, and checks the curated per-file size ceiling. Do not rerun experiments before this check: runners may replace result files with regenerated outputs and thereby invalidate the distribution manifest.

## Environment

Python 3.12 is recommended; the recorded run used Python 3.12.14. Install `requirements.txt` in a fresh virtual environment. The recorded GPU setup used PyTorch `2.11.0+cu128`; CPU operation is supported but may be substantially slower.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduce the experiments

The first two commands regenerate the preceding revision. E24--E29 can also run directly from the shipped frozen prediction/band stores:

```bash
python run_all_revision.py --config configs/revision_v1.yaml
python run_e20_e23.py --config configs/reviewer_closure_v1.yaml
python run_review_round2.py --config configs/review_round2_v1.yaml --nyiso-raw-dir /path/to/2025-nyiso-monthly-zips
```

The third command requires the twelve **official** NYISO monthly archives for 2025, named `20250101pal_csv.zip` through `20251201pal_csv.zip`, for E29's timestamp audit. Put them in one external directory and pass that directory to `--nyiso-raw-dir`; alternatively use the ignored default directory `data/raw_nyiso/`. Raw operator ZIPs are not redistributed. The E24--E28 calculations use included frozen inputs and do not alter the point-model specification. The second-round runner was re-executed in a clean staged copy; all scientific result files matched the original byte-for-byte. Only E24 path metadata and `RUN_STATUS.json` changed to portable relative paths.

For raw-input preprocessing rather than the supplied frozen inputs, follow [`DATA.md`](DATA.md) and run:

```bash
python experiments/e0_preprocess.py --downloads /path/to/downloads --output outputs/01_data_audit
```

Operator archives can be revised after retrieval, so newly rebuilt Parquet bytes need not equal the frozen fingerprints. The full E0--E19 run took about 37 minutes on the recorded workstation; E20--E23 took about 75 seconds and E24--E29 about 23 seconds. These are run records, not hardware-independent benchmarks.

## Evaluation and interpretation

2020--2023 serve as development diagnostics, 2024 as validation/selection, 2025 as retrospective sequential evaluation, and 2026H1 as a partial-year follow-up. The retained 90-day calibration window is inherited, not a proven validation optimum. The 2024 joint window/decision and ACI choices were formulated after 2025 had been inspected; their 2025 comparisons are retrospective. Conditional block-resampling ranges are not confidence intervals for the fixed realized annual cost or guarantees of future-year performance. See [`EXPECTED_RESULTS.md`](EXPECTED_RESULTS.md) and `reports/ROUND2_REVISION_V1/ROUND2_EXPERIMENT_REPORT.md` for checkpoints and limitations.

## Data and third-party terms

Raw ISO-NE and NYISO operator archives are not redistributed. See [`DATA.md`](DATA.md) for official data sources, reconstruction requirements, and the additional raw files needed for a complete E29 rerun.

No open-source software license is granted by this package. The manuscript's publication license does not automatically apply to code or data. ISO-NE and NYISO source archives retain their providers' terms; this package does not relicense or redistribute raw operator data. Processed inputs are provided for reproducibility, subject to the notices in [`LICENSE_AND_DATA_TERMS.md`](LICENSE_AND_DATA_TERMS.md) and [`DATA.md`](DATA.md).

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). After the version 1.2.0 archival deposit is published, add the resulting DOI to the repository metadata and use that fixed archived version in the manuscript's Data Availability statement.
