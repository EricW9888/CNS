"""Create the compact, public-complete frozen EXP-011 record."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.provenance import sha256_file


EXP = ROOT / "experiments/EXP-011-acceptance-convergence"
SPEC_PATH = EXP / "specification.json"
METRICS_PATH = EXP / "local-metrics.json"
MANIFEST_PATH = EXP / "archive-manifest.json"
VISIBILITY_PATH = EXP / "visibility-diagnosis.json"
CANONICAL_PATH = EXP / "canonical-local-result.json"
OUTPUT = EXP / "record.json"


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf8"))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf8"))
    visibility = json.loads(VISIBILITY_PATH.read_text(encoding="utf8"))
    canonical = json.loads(CANONICAL_PATH.read_text(encoding="utf8"))
    conditions = spec["conditions"]
    physical = {
        condition: metrics["measurements"]["combined"][condition]
        ["production_to_reference"]["physical_light"]["max_abs_difference"]
        for condition in conditions
    }
    final_local = {}
    for condition in conditions:
        row = metrics["gate_rows"][condition]
        final_local[condition] = {
            "combined_production_to_reference_5pct": row["combined_production_to_reference"],
            "optical_production_to_reference_5pct": row["optical_production_to_reference"],
            "temporal_0.5_to_0.25ms_5pct": row["temporal_0.5_to_0.25ms"],
            "neural_half_step_5pct": row["neural_half_step"],
            "trends_passed": row["trends"],
            "controls_passed": row["controls"],
            "physical_light_1e-4_passed": row["production_reference_physical_light_1e-4"],
        }
    implementation_files = [
        "src/acceptance_integral.py", "src/compound_eye.py", "src/retinotopic_eye.py",
        "src/retinotopy_diagnostics.py", "src/sensory_chain.py",
        "scripts/render_exp011_sensors.py", "scripts/preflight_exp011.py",
        "scripts/analyze_exp011_canonical_metrics.py", "tests/test_acceptance_integral.py",
    ]
    implementation_sha256 = {path: sha256_file(ROOT / path) for path in implementation_files}
    independent_mass = visibility["worst_case_split_and_independent_mass"]
    public_tag = "exp011-canonical-archives-2026-09-14"
    public_asset = "exp011-canonical-archives-direct.tar"
    record = {
        "experiment_id": spec["experiment_id"],
        "status": "negative_partial_numerical_result",
        "question": spec["question"],
        "public_scope": "Canonical scientific record. All material assumptions, methods, failures, corrections, diagnostics, provenance, and exact raw archive access needed to evaluate this result are included here or in the public CNS release asset.",
        "predecessor": {
            "experiment": spec["predecessor"],
            "public_base_commit": spec["predecessor_commit"],
            "frozen_predecessors_unchanged": True,
        },
        "frozen_method": {
            "optical": spec["optical_method"],
            "quadrature_normalization": spec["quadrature_normalization"],
            "levels": spec["ordered_levels"],
            "candidate": spec["candidate"],
            "temporal_contract": spec["temporal_contract"],
            "translation_execution_path": spec["execution_path_resolution"],
            "visibility_not_neural_input": True,
            "no_biology_scene_threshold_or_downstream_changes": True,
        },
        "predeclared_criteria": {
            "maximum_local_stage_relative_error": spec["acceptance"]["maximum_local_stage_relative_error"],
            "maximum_physical_light_absolute_error": spec["acceptance"]["maximum_physical_light_absolute_error"],
            "trend": spec["acceptance"]["trend"],
            "candidate_gate": spec["acceptance"]["candidate_gate"],
            "static_and_disconnection_max_abs": spec["acceptance"]["static_and_disconnection_max_abs"],
            "determinism": spec["acceptance"]["determinism"],
        },
        "initial_verifier_failure": canonical,
        "visibility_diagnosis": {
            "record": "visibility-diagnosis.json",
            "verifier_failure_preserved": True,
            "all_offenders_mixed": visibility["all_offenders_mixed"],
            "archive_integrity_except_blocked_mass_bounds": visibility["metadata_shape_source_digest_checks_except_blocked_mass_bounds"],
            "independent_high_order_mass_physical": all(
                0.0 <= row["independent_dense_mass_256x4096"] <= 1.0 for row in independent_mass),
            "interpretation": "Finite split-quadrature blocked-mass diagnostic overshoot; the pole-adjacent finite-evaluator explanation remains an inference, not a biological or physical discontinuity claim.",
        },
        "local_analysis": {
            "record": "local-metrics.json",
            "candidate_gate_pairs_corrected": metrics["implementation"]["candidate_gate_pairs"],
            "final_local_stages_and_pools_passed": all(
                all(metrics["gate_rows"][condition][key].values())
                for condition in conditions
                for key in ("combined_production_to_reference", "optical_production_to_reference", "temporal_0.5_to_0.25ms", "neural_half_step")
            ),
            "trends_passed": all(metrics["gate_rows"][condition]["trends"] for condition in conditions),
            "controls_passed": metrics["control_gate_passed"],
            "archive_manifest_sha256_verified": metrics["archive_manifest"]["all_sha256_verified"],
            "streaming_equivalence_prefixes_passed": all(row["passed"] for row in metrics["streaming_correlator_equivalence_prefixes"]),
            "physical_light_max_abs_by_condition": physical,
            "physical_light_gate_passed_by_condition": {
                condition: metrics["gate_rows"][condition]["production_reference_physical_light_1e-4"]
                for condition in conditions
            },
            "final_local_gate_passed": metrics["candidate_local_gate_passed"],
            "interpretation": "The final local numerical chain meets the declared relative convergence criteria, but the accepted production/reference optical light remains outside the frozen 1e-4 requirement in every condition.",
        },
        "result": {
            "classification": "negative_partial_numerical_result",
            "physical_light_gate_failed_all_conditions": True,
            "no_stronger_biological_claim": True,
            "downstream_evaluation": False,
            "LPi_HS_H2_DNp15": "unevaluated",
        },
        "interpretation_and_limits": [
            "This result concerns numerical convergence of the provisional full-eye optical/local sensory interface, not biological validation.",
            "The failed blocked_mass bound is retained as a visible verifier failure and does not support a clean archive-integrity claim.",
            "No anatomy registration, motor calibration, behavior, learning, or downstream causal claim follows.",
            "The finest reference is a predeclared finite numerical reference, not ground truth.",
        ],
        "provenance": {
            "specification_sha256": sha256_file(SPEC_PATH),
            "archive_manifest_sha256": sha256_file(MANIFEST_PATH),
            "archive_count": manifest["archive_count"],
            "archive_total_bytes": manifest["total_bytes"],
            "archive_bundle_sha256": manifest["release_storage"]["bundle"]["sha256"],
            "public_archive_release": {
                "repository": "EricW9888/CNS",
                "release_tag": public_tag,
                "asset_name": public_asset,
                "asset_url": f"https://github.com/EricW9888/CNS/releases/download/{public_tag}/{public_asset}",
            },
            "implementation_sha256": implementation_sha256,
        },
        "artifacts": {
            "specification": "specification.json",
            "canonical_initial_result": "canonical-local-result.json",
            "visibility_diagnosis": "visibility-diagnosis.json",
            "local_metrics": "local-metrics.json",
            "archive_manifest": "archive-manifest.json",
            "summary_figure": "figures/EXP-011-acceptance-convergence.png",
        },
    }
    OUTPUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"output": str(OUTPUT), "status": record["status"],
                      "physical_light": physical}, indent=2))


if __name__ == "__main__":
    main()
