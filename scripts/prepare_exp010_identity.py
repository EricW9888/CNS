"""Verify released male body/column tables; do not infer a facet/body matching."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
from urllib.request import urlopen
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.provenance import sha256_file

BASE = "https://raw.githubusercontent.com/reiserlab/male-drosophila-visual-system-connectome-code/dbafc73124b5c96e96429cdf2a89d067cae841bc/"
SOURCES = [
    ("ME_assigned_columns.csv", BASE+"results/exchange/ME_assigned_columns.csv", "649140bab99503d13136e7cbe60372d981e728505d5c8db0b82f0ef33cfd8c8a"),
    ("supplementary_tables.zip", "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-025-08746-0/MediaObjects/41586_2025_8746_MOESM4_ESM.zip", "ef34f64c74e157fdec06312e3a1d6e567ae48a383944dd99a3f854e7ba593c93"),
    ("coordinate-systems.md", BASE+"docs/coordinate-systems.md", "d88398773d1ae7b883e0ce1cbf074408ad342c78fa65a4b00915a25f66889c0e")]
TARGET = ROOT / "experiments/EXP-010-retinotopic-refinement/identity-evidence.json"


def summarize_assignments(columns, alignment, annotations):
    if columns.bodyId.duplicated().any() or annotations.bodyId.duplicated().any():
        raise ValueError("body axes must be unique")
    joined = columns.merge(annotations[["bodyId", "type", "instance"]], on="bodyId", how="left", validate="one_to_one")
    result = {"column_assignments": {"rows": len(columns), "exact_current_body_ids": int(joined.type.notna().sum()),
              "exact_type_agreement": int((joined.neuron_type == joined.type).sum()), "types": columns.neuron_type.value_counts().sort_index().to_dict()}}
    by_id = annotations.set_index("bodyId")
    mi = columns[columns.neuron_type == "Mi1"].set_index("bodyId")
    coverage = {}
    for subtype in "abcd":
        entries = alignment[["Mi1 bodyId", f"T4{subtype} bodyId", "Used in analysis"]].dropna()
        ids = entries[f"T4{subtype} bodyId"].astype("int64")
        mn = entries["Mi1 bodyId"].astype("int64")
        current = by_id.reindex(ids)
        coverage[f"T4{subtype}"] = {"published_assignments": len(entries), "unique_T4_ids": int(ids.nunique()),
                                   "exact_current_body_ids": int(current.type.notna().sum()),
                                   "exact_current_subtype": int((current.type == f"T4{subtype}").sum()),
                                   "Mi1_ids_with_released_hex_addresses": int(mn.isin(mi.index).sum()),
                                   "rows_used_in_published_analysis": int((entries["Used in analysis"] == 1).sum()),
                                   "current_instances_R": int(current.instance.str.endswith("_R", na=False).sum())}
    result["published_Mi1_T4_alignment"] = coverage
    t = annotations[annotations.type.str.fullmatch("T[45][abcd]", na=False)]
    result["annotation_table_T4_T5"] = {"rows": len(t), "nonmissing_assignedOlHex1": int(t.assignedOlHex1.notna().sum()),
                                        "nonmissing_assignedOlHex2": int(t.assignedOlHex2.notna().sum())}
    result["individual_measured_facet_to_MaleCNS_mapping"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--write-evidence", action="store_true")
    parser.add_argument("--check-evidence", action="store_true")
    args = parser.parse_args()
    if args.write_evidence and (TARGET.exists() or args.check_evidence):
        raise ValueError("evidence records cannot be overwritten")
    folder = ROOT / "data/exp010_identity"
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for name, url, digest in SOURCES:
        path = folder / name
        if args.download and not path.exists():
            with urlopen(url, timeout=30) as response:
                path.write_bytes(response.read())
        if sha256_file(path) != digest:
            raise ValueError(f"primary source differs: {name}")
        files.append({"path": path.relative_to(ROOT).as_posix(), "url": url, "bytes": path.stat().st_size, "sha256": digest, "redistributed": False})
    registry = json.loads((ROOT / "data/manifest.json").read_text())
    known = registry["sources"] + [f for m in registry["materializations"] for f in m["files"]]
    for filename in ("data/body-annotations.feather", "data/optic-column-type-assignments-v1.0.xlsx"):
        entry = next(e for e in known if e["path"] == filename)
        if sha256_file(ROOT/filename) != entry["sha256"]:
            raise ValueError(f"frozen identity input differs: {filename}")
        files.append({k: entry[k] for k in ("path", "bytes", "sha256")})
    columns = pd.read_csv(folder / "ME_assigned_columns.csv")
    with zipfile.ZipFile(folder / "supplementary_tables.zip") as z:
        alignment = pd.read_excel(io.BytesIO(z.read("Sup_Table_4_Mi1_T4_alignment_final.xlsx")))
    result = {"experiment_id": "EXP-010-retinotopic-refinement", "source_files": files,
              **summarize_assignments(columns, alignment, pd.read_feather(ROOT/"data/body-annotations.feather")),
              "registration_boundary": "Exact current right-eye body IDs and source-assigned Mi1-to-T4/hex correspondences are available. They are anatomical assignment estimates, not measured receptive fields. Zhao's released lens/FAFB indices use another specimen/coordinate system. A template-space registration is not an individual facet identity crosswalk. No left/T5 individual eye registration is established here.",
              "coordinate_evidence": "Male optic-lobe author docs define hex1/hex2 with an arbitrary exterior origin and P/Q origin at hex1=18,hex2=19. MaleCNS downloads expose native 8nm skeletons and JRC2018 template transforms; no validated eye-grid registration follows from an origin shift alone.",
              "fresh_release_inspection": {"repository_commit": "67767d2233657983993ff6c2be48e836a935863c",
                "column_workbook_url": "https://raw.githubusercontent.com/flyconnectome/2025malecns/67767d2233657983993ff6c2be48e836a935863c/supplemental_data/optic-column-type-assignments-v1.0.xlsx",
                "column_workbook_sha256": "d4af1cacb751036f7e84bfecc9bec79ca010066ac066559c29b566003ec080d3"}}
    text = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.check_evidence and json.loads(TARGET.read_text()) != result:
        raise ValueError("identity evidence replay differs")
    if args.write_evidence:
        TARGET.write_text(text, encoding="utf8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
