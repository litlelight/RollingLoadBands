# Data sources and reconstruction

Raw archives are not redistributed in this compact package. The frozen processed inputs required by the model pipelines are included under `outputs/01_data_audit/`.

## Data rights and conditions of use

ISO-NE and NYISO retain all rights they may have in their source data and website content. Neither the package's software status nor the manuscript's publication license relicenses those third-party materials. Inclusion of frozen processed model inputs is not a sublicense or transfer of ownership. Users are responsible for checking the providers' current terms and applicable law before using or redistributing source data or derived files.

- ISO-NE source content is governed by the provider's [Legal and Privacy terms](https://www.iso-ne.com/legal-privacy) and the terms attached to the relevant data service.
- NYISO source data should be obtained from the official [Real-Time Actual Load archive](https://mis.nyiso.com/public/P-58Blist.htm) and used subject to NYISO's applicable notices and terms.
- This package makes no representation that provider terms remain unchanged after the release date.

See [`LICENSE_AND_DATA_TERMS.md`](LICENSE_AND_DATA_TERMS.md) for the separation between software, manuscript, and third-party data rights.

## ISO New England

- Official source: [ISO-NE System Loads in EEI Format](https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/sys-load-eei-fmt)
- Study coverage: 2003-01-01 through 2026-06-30.
- Resolution used by the study: 24 EEI load positions per day.
- The preprocessing implementation expects the downloaded annual text files to be present inside two local archives named `my_reports.zip` and `my_reports (1).zip`. Together they must cover 2003 through 2026H1.

Documented source handling:

- One malformed trailing 2014 fragment is removed only after the complete matching record is confirmed in the 2015 source file.
- The zero placeholder at 2015-03-08 hour-ending 02 is replaced with the adjacent-position mean, 12,378 MW.
- Raw archives are never modified.

## New York ISO

- Official source: [NYISO Real-Time Actual Load archive](https://mis.nyiso.com/public/P-58Blist.htm)
- Study coverage: January 2018 through December 2025.
- Required canonical archive names: `YYYYMM01pal_csv.zip`.
- Expected archive count: 96 monthly ZIP files.
- Duplicate browser downloads with names such as `(1)` are ignored by the parser.

The seven missing daily files from 2021-04-01 through 2021-04-07 are treated as an official archive gap and are not filled. Because the feature definition requires 56 days of history, the first valid post-gap target is 2021-06-03.

NYISO system load is constructed from the eleven named zones in each P-58B file. EDT and EST observations are mapped to distinct UTC instants before local-hour aggregation. Spring-forward days are normalized by interpolation of the missing local hour; fall-back repeated hours are averaged.

## Frozen processed inputs

| File | Purpose |
| --- | --- |
| `ISO_NE_daily_profile.parquet` | ISO-NE 24-position daily load profiles |
| `ISO_NE_features.parquet` | ISO-NE 233 predictors and 24 targets |
| `NYISO_daily_profile.parquet` | NYISO system-level 24-hour daily profiles |
| `NYISO_features.parquet` | NYISO 233 predictors and 24 targets |
| `feature_manifest.csv` | Predictor definitions and forecast-origin availability |

The complete quality-control record is in the remaining CSV files and `data_audit_report.md` in the same directory.

## Additional raw files for the E29 timestamp audit

The second-round runner uses the twelve official NYISO monthly archives for
January--December 2025 to count off-five-minute-grid observations. Keep the
canonical names `20250101pal_csv.zip` through `20251201pal_csv.zip` in one
directory outside the distributed package (or in the ignored `data/raw_nyiso/`
directory). Then pass its path to `run_review_round2.py --nyiso-raw-dir`.
Those ZIP bytes are not part of this repository. E24--E28 and the stored E29
CSV audit results can be inspected without downloading them, but a full E29
rerun requires them. The audit uses the same equal-observation local-hour target
definition as the manuscript; it does not construct duration-weighted energy.


## Third-round E30/E31 evidence

E30 is a retrospective NYISO raw-timestamp aggregation sensitivity; it requires **all 96** official monthly `YYYYMM01pal_csv.zip` archives covering 2018--2025 for a full rerun. Their hashes are recorded in `results/ROUND3_MINOR_REVISION_V1/E30/raw_zip_sha256.csv`, but the original operator archives themselves are **not** distributed. The alternative integrates a forward-held observation for at most 30 minutes across UTC hours, without bridging longer gaps (at most five minutes is used to close adjacent hour/dataset edges), and reuses the same local-day/DST standardization. It is not asserted to equal a physical duration-weighted meter series.

E30's machine-readable hourly/day/event/forecast/band/decision outputs are distributed under `results/ROUND3_MINOR_REVISION_V1/E30/`. E31 reuses the frozen E26 `quality_metrics.csv` already in this package; it requires no raw data or model rerun to regenerate its audit table.
