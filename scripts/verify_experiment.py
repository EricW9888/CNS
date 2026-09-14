"""Check frozen artifacts, then delegate declared replay or historical invariants."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_exp009_retinotopy import frozen_entry_digest
from src.provenance import sha256_file


def pointer(document, path):
    if not path.startswith("/"):
        raise ValueError("evidence paths must be JSON pointers")
    for token in path[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        document = document[int(token)] if isinstance(document, list) else document[token]
    return document


def repository_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or Path(relative).is_absolute():
        raise ValueError("artifact path outside repository")
    return path


def evidence_assertion(value, assertion):
    if "equals" in assertion:
        expected = assertion["equals"]
        if isinstance(value, bool) or isinstance(expected, bool):
            return type(value) is type(expected) and value == expected
        return value == expected
    if "greater_than" in assertion:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > assertion["greater_than"]
    if "length" in assertion:
        return len(value) == assertion["length"]
    raise ValueError("unknown evidence assertion")


def valid_test_reference(test):
    file, *selector = test.split("::")
    path = repository_path(file)
    if not path.is_file():
        return False
    if not selector:
        return True
    # These registry references use top-level test functions, not parametrized
    # node IDs or classes. Check the actual symbol without importing tests/data.
    return len(selector) == 1 and any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == selector[0]
                                     for n in ast.parse(path.read_text()).body)


def validate_registry(registry=None, claims_document=None):
    if registry is None:
        registry = json.loads((ROOT/"experiments/registry.json").read_text())
    entries = registry["experiments"]
    ids = [e["experiment_id"] for e in entries]
    records = list((ROOT/"experiments").glob("*/record.json"))
    if len(set(ids)) != len(ids) or {e["record"] for e in entries} != {p.relative_to(ROOT).as_posix() for p in records}:
        raise ValueError("registry does not exactly cover experiment records")
    for entry in entries:
        record = json.loads(repository_path(entry["record"]).read_text())
        if (record["experiment_id"], record["status"]) != (entry["experiment_id"], entry["status"]):
            raise ValueError("registry status/identity drift")
        if record.get("question", entry["question"]) != entry["question"]:
            raise ValueError("registry question differs from record")
        if entry["predecessor"] is not None and entry["predecessor"] not in ids:
            raise ValueError("unknown predecessor")
        for artifact in (entry["report"], entry["record"], entry["primary_figure"]):
            if artifact is not None and not repository_path(artifact).is_file():
                raise ValueError(f"missing registered artifact: {artifact}")
        if entry["canonical_verify_command"] != f"python scripts/verify_experiment.py {entry['experiment_id']}":
            raise ValueError("noncanonical verification command")
        if len(entry["frozen_commit"]) != 40 or not entry["result"]:
            raise ValueError("freeze/result metadata incomplete")
        if entry["verification"]["kind"] not in ("exact_replay", "historical_invariants"):
            raise ValueError("unknown verification kind")
        if entry["verification"]["kind"] == "exact_replay":
            command = entry["verification"]["replay_command"]
            if not repository_path(command[0]).is_file() or "--check-record" not in command or any(flag in command for flag in ("--write-record", "--promote-figure")):
                raise ValueError("verification must use an existing read-only record replay")
        for test in entry["verification"]["tests"]:
            if not valid_test_reference(test):
                raise ValueError("missing evidence test")
    if claims_document is None:
        claims_document = json.loads((ROOT/"experiments/evidence.json").read_text())
    claims = claims_document["claims"]
    if len({c["claim_id"] for c in claims}) != len(claims):
        raise ValueError("duplicate claim IDs")
    by_id = {e["experiment_id"]: e for e in entries}
    for entry in entries:
        visited, current = set(), entry["experiment_id"]
        while current is not None:
            if current in visited:
                raise ValueError("cyclic predecessor history")
            visited.add(current)
            current = by_id[current]["predecessor"]
    for claim in claims:
        entry = by_id[claim["experiment_id"]]
        record = json.loads(repository_path(entry["record"]).read_text())
        if not evidence_assertion(pointer(record, claim["value_pointer"]), {"equals": claim["value"]}):
            raise ValueError("claim value differs from record")
        for evidence in claim["evidence"]:
            if not evidence_assertion(pointer(record, evidence["pointer"]), evidence):
                raise ValueError(f"claim evidence fails: {claim['claim_id']}/{evidence['pointer']}")
        for test in claim["tests"]:
            if not valid_test_reference(test):
                raise ValueError("missing claim test")
    return by_id


def hash_entries(document):
    if isinstance(document, dict):
        if isinstance(document.get("path"), str) and "sha256" in document:
            yield document
        for value in document.values():
            yield from hash_entries(value)
    elif isinstance(document, list):
        for value in document:
            yield from hash_entries(value)


def ignored_local_artifact(relative):
    """Only absent ignored downloads/results may be omitted in CI checks."""
    return relative.startswith(("data/", "results/")) and subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative], cwd=ROOT,
        check=False).returncode == 0


def integrity(entry, *, allow_missing=False):
    # Verify exact Git bytes, allowing only checkout newline conversion for
    # record.json (all historical record attributes already declare text/LF).
    for name in (entry["record"], entry["primary_figure"]):
        if name is None:
            continue
        frozen = subprocess.check_output(["git", "show", f"{entry['frozen_commit']}:{name}"], cwd=ROOT)
        current = repository_path(name).read_bytes()
        if name.endswith("record.json"):
            current = current.replace(b"\r\n", b"\n")
        if current != frozen:
            raise ValueError(f"frozen artifact changed: {name}")
    if entry["primary_figure"]:
        with Image.open(repository_path(entry["primary_figure"])) as image:
            image.verify()
    folder = repository_path(entry["record"]).parent
    record = json.loads((folder/"record.json").read_text())
    specification = folder/"specification.json"
    documents = [record]
    if "identity_evidence_sha256" in record:
        identity = folder/"identity-evidence.json"
        if sha256_file(identity) != record["identity_evidence_sha256"]:
            raise ValueError("identity evidence fingerprint differs from record")
        documents.append(json.loads(identity.read_text()))
    if specification.exists():
        documents.append(json.loads(specification.read_text()))
        expected = record.get("specification_sha256", record.get("metrics", {}).get("specification_sha256"))
        if expected and sha256_file(specification) != expected:
            raise ValueError("specification fingerprint differs from record")
    checked, missing = set(), set()
    for document in documents:
        for source in hash_entries(document):
            path = repository_path(source["path"])
            if not path.exists():
                if not allow_missing or not ignored_local_artifact(source["path"]):
                    raise FileNotFoundError(f"required source absent: {source['path']}")
                missing.add(source["path"])
                continue
            if frozen_entry_digest(source) != source["sha256"]:
                raise ValueError(f"source fingerprint differs: {source['path']}")
            checked.add(source["path"])
        # Historical names differ; preserve their explicit LF-source policy.
        for obj in (document, document.get("metrics", {})):
            for key in ("implementation_sha256", "implementation_LF_sha256"):
                for path, digest in obj.get(key, {}).items():
                    if frozen_entry_digest({"path": path}) != digest:
                        raise ValueError(f"implementation fingerprint differs: {path}")
                    checked.add(path)
    return {"experiment_id": entry["experiment_id"], "frozen_artifacts": "passed",
            "source_files_checked": len(checked), "missing_local_artifacts": sorted(missing), "replay": "not_executed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id", nargs="?")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--integrity-only", action="store_true", help="CI/partial-install check; never claims a replay")
    parser.add_argument("--sensors", type=Path, default=ROOT/"results/exp008_eye_final")
    parser.add_argument("--reference", type=Path, default=ROOT/"results/exp010_reference")
    args = parser.parse_args()
    entries = validate_registry()
    if args.all == bool(args.experiment_id):
        parser.error("select one experiment or --all")
    selected = list(entries.values()) if args.all else [entries[args.experiment_id]]
    for entry in selected:
        result = integrity(entry, allow_missing=args.integrity_only)
        print(json.dumps(result), flush=True)
        if args.integrity_only:
            continue
        subprocess.run([sys.executable, "-m", "pytest", "-q", *entry["verification"]["tests"]], cwd=ROOT, check=True)
        if entry["verification"]["kind"] == "historical_invariants":
            print("Historical artifact/invariant verification only; no fresh body or model replay claimed.", flush=True)
            continue
        command = [token.replace("{sensors}", str(args.sensors)).replace("{reference}", str(args.reference)) for token in entry["verification"]["replay_command"]]
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
        # Recheck immutable results after delegated replay; it must not write
        # canonical records or promote a replacement historical figure.
        integrity(entry)
        print(f"Exact record replay and frozen artifact verification passed: {entry['experiment_id']}", flush=True)


if __name__ == "__main__":
    main()
