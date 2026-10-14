"""Expose the already-frozen 2024 ACI selection values beside 2025 results."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "work" / "pydeps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(root: Path, output: Path) -> None:
    source = root / "results" / "ROUND2_REVISION_V1" / "E26" / "quality_metrics.csv"
    quality = pd.read_csv(source)
    selected = quality.loc[quality.family.eq("aci_projected") & quality.period.isin([2024, 2025])].copy()
    selected["eta"] = selected.method.str.removeprefix("aci_eta").astype(float)
    if len(selected) != 24:
        raise AssertionError("Expected two systems, six ACI steps, and two periods")
    validation = selected.loc[selected.period.eq(2024), ["dataset", "eta", "days", "joint_coverage", "mean_width_mw"]]
    validation = validation.rename(columns={
        "days": "validation_days", "joint_coverage": "validation_joint_coverage",
        "mean_width_mw": "validation_width_mw",
    })
    evaluation = selected.loc[selected.period.eq(2025), [
        "dataset", "eta", "days", "joint_coverage", "mean_width_mw", "interval_score_mw"
    ]].rename(columns={
        "days": "evaluation_days", "joint_coverage": "evaluation_joint_coverage",
        "mean_width_mw": "evaluation_width_mw", "interval_score_mw": "evaluation_interval_score_mw",
    })
    result = validation.merge(evaluation, on=["dataset", "eta"], validate="one_to_one")
    result["validation_absolute_gap_to_90pp"] = 100 * (result.validation_joint_coverage - 0.9).abs()
    result["selected"] = False
    for system, group in result.groupby("dataset"):
        winner = group.sort_values([
            "validation_absolute_gap_to_90pp", "validation_width_mw", "eta"
        ]).index[0]
        result.loc[winner, "selected"] = True
        if not np.isclose(result.at[winner, "eta"], 0.02):
            raise AssertionError(f"Unexpected ACI step selected for {system}")
    output.mkdir(parents=True, exist_ok=True)
    result = result.sort_values(["dataset", "eta"]).reset_index(drop=True)
    result.to_csv(output / "aci_2024_selection_and_2025_results.csv", index=False, float_format="%.10f")
    explanation = """# E31 ACI validation-period selection audit

No model or ACI path was rerun. This table joins the frozen 2024 and 2025
rows in E26/quality_metrics.csv. The selected step is the smallest absolute
2024 joint-coverage gap to 90%, then the narrower 2024 mean width. The table
lists 2025 evaluation values only after the 2024 selection columns.

The choice of this selection protocol itself was formulated retrospectively
after inspection of the 2025 record, as already disclosed in the manuscript.
"""
    (output / "E31_REPORT.md").write_text(explanation, encoding="utf-8")
    manifest = pd.DataFrame([
        {"input": str(source.relative_to(root)), "sha256": sha256(source)},
        {"input": "aci_2024_selection_and_2025_results.csv",
         "sha256": sha256(output / "aci_2024_selection_and_2025_results.csv")},
    ])
    manifest.to_csv(output / "E31_HASHES.csv", index=False)


if __name__ == "__main__":
    run(ROOT, ROOT / "results" / "ROUND3_MINOR_REVISION_V1" / "E31")
