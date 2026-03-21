#!/usr/bin/env python3
"""Prepare repository content for initial commit.

Features:
- File scanning and categorization
- .gitignore effectiveness checks
- Suspicious file detection
- Security checklist checks
- Manifest and summary report generation
- Optional interactive exclusion flow
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bootstrap_common import (
    has_tty,
    human_size,
    is_binary_file,
    repo_root,
    run_cmd,
    setup_logger,
    system_snapshot,
    utc_now,
    write_json,
    write_text,
)


SENSITIVE_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]"),
    re.compile(r"(?i)password\s*[:=]\s*['\"].{4,}['\"]"),
    re.compile(r"(?i)token\s*[:=]\s*['\"][A-Za-z0-9_\-\.]{12,}['\"]"),
]

REQUIRED_FILES = [
    ".gitignore",
    ".gitattributes",
    ".pre-commit-config.yaml",
    "README.md",
    "LICENSE",
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
]

ALWAYS_EXCLUDE_PATTERNS = [
    ".git/**",
    "f1_racing_env/**",
    ".venv/**",
    "__pycache__/**",
    "*.pyc",
    "logs/**",
    "tmp/**",
    "temp/**",
]

PRUNE_DIRS = {
    ".git",
    "f1_racing_env",
    ".venv",
    "__pycache__",
    "node_modules",
    "logs",
    "tmp",
    "temp",
}

GENERATED_ARTIFACT_PATTERNS = [
    "data/processed/**",
    "data/tracks/**/extracted/**",
    "data/tracks/**/visualizations/**",
    "data/tracks/**/analysis/**",
    "models/checkpoints/**",
    "experiments/**/checkpoints/**",
]


@dataclass
class ScanItem:
    path: str
    category: str
    size_bytes: int
    binary: bool
    suspicious: list[str]


def normalize(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def git_ignored(root: Path, rel_path: str) -> bool:
    result = run_cmd(["git", "check-ignore", "-q", rel_path], cwd=root)
    return result.returncode == 0


def categorize(rel_path: str) -> str:
    if rel_path.startswith("docs/") or rel_path.endswith(".md"):
        return "documentation"
    if rel_path.startswith("tests/") or rel_path.startswith("test"):
        return "tests"
    if rel_path.startswith("configs/") or rel_path.endswith((".json", ".yaml", ".yml", ".toml", ".ini")):
        return "configuration"
    if rel_path.startswith("scripts/"):
        return "scripts"
    if rel_path.startswith(("utils/", "environments/", "models/", "training/", "control/", "planning/", "perception/", "vehicle_dynamics/")):
        return "source"
    if rel_path.startswith(".github/"):
        return "ci"
    if rel_path.startswith("data/"):
        return "data"
    return "other"


def python_header_ok(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:6]
    except OSError:
        return False
    joined = "\n".join(lines)
    return "#!/usr/bin/env python" in joined or '"""' in joined or "'''" in joined


def find_missing_inits(root: Path) -> list[str]:
    missing: list[str] = []
    for current, dirs, files in os.walk(root):
        dirs[:] = [item for item in dirs if item not in PRUNE_DIRS]
        directory = Path(current)
        rel = normalize(directory, root)
        if rel.startswith(("data", "notebooks")):
            continue
        has_py = any(name.endswith(".py") and name != "__init__.py" for name in files)
        if has_py and not (directory / "__init__.py").exists():
            missing.append(rel)
    return sorted(missing)


def scan_repository(root: Path, logger) -> dict[str, Any]:
    include_items: list[ScanItem] = []
    ignored_should_be: list[str] = []
    warnings: list[str] = []
    sensitive_hits: list[dict[str, Any]] = []

    for current, dirs, files in os.walk(root):
        dirs[:] = [item for item in dirs if item not in PRUNE_DIRS]
        base = Path(current)
        for filename in files:
            path = base / filename
            rel = normalize(path, root)

            if matches_any(rel, ALWAYS_EXCLUDE_PATTERNS):
                continue

            size = path.stat().st_size
            binary = is_binary_file(path)
            suspicious: list[str] = []

            if size > 10 * 1024 * 1024:
                suspicious.append("large_file")
            if binary:
                suspicious.append("binary")
            if rel.endswith((".tmp", ".bak")):
                suspicious.append("temporary")

            should_ignore = matches_any(rel, GENERATED_ARTIFACT_PATTERNS)
            is_ignored = git_ignored(root, rel)
            if should_ignore and not is_ignored:
                ignored_should_be.append(rel)

            if is_ignored:
                continue

            category = categorize(rel)
            include_items.append(ScanItem(rel, category, size, binary, suspicious))

            if not binary and size < 2 * 1024 * 1024:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    text = ""
                for pattern in SENSITIVE_PATTERNS:
                    match = pattern.search(text)
                    if match:
                        sensitive_hits.append({"file": rel, "pattern": pattern.pattern, "sample": match.group(0)[:80]})

    category_counts = Counter(item.category for item in include_items)
    total_size = sum(item.size_bytes for item in include_items)

    missing_required = [file for file in REQUIRED_FILES if not (root / file).exists()]
    if missing_required:
        warnings.append(f"Missing required files: {', '.join(missing_required)}")

    missing_inits = find_missing_inits(root)
    if missing_inits:
        warnings.append(f"Missing __init__.py in: {', '.join(missing_inits[:15])}")

    py_header_missing: list[str] = []
    for item in include_items:
        if item.path.endswith(".py"):
            if not python_header_ok(root / item.path):
                py_header_missing.append(item.path)
    if py_header_missing:
        warnings.append(f"Python header/docstring missing in {len(py_header_missing)} files")

    grouped = defaultdict(list)
    for item in include_items:
        grouped[item.category].append(
            {
                "path": item.path,
                "size_bytes": item.size_bytes,
                "binary": item.binary,
                "suspicious": item.suspicious,
            }
        )

    report = {
        "timestamp": utc_now(),
        "system": system_snapshot(),
        "summary": {
            "total_files": len(include_items),
            "total_size_bytes": total_size,
            "total_size_human": human_size(total_size),
            "category_counts": dict(category_counts),
        },
        "issues": {
            "missing_required": missing_required,
            "should_be_ignored_but_not": ignored_should_be,
            "sensitive_hits": sensitive_hits,
            "python_header_missing": py_header_missing,
            "missing_inits": missing_inits,
            "warnings": warnings,
        },
        "manifest": dict(grouped),
    }
    logger.info("Prepared manifest: %d files, %s", len(include_items), human_size(total_size))
    return report


def interactive_exclusions(report: dict[str, Any]) -> set[str]:
    if not has_tty():
        return set()

    flat = []
    for category, items in report["manifest"].items():
        for item in items:
            flat.append((category, item["path"], item["size_bytes"]))

    print("\nInteractive review mode")
    print("Enter comma-separated indexes to exclude, or press Enter to continue.\n")
    for idx, (_, path, size) in enumerate(flat[:200], start=1):
        print(f"[{idx:03d}] {path} ({human_size(size)})")
    if len(flat) > 200:
        print(f"... truncated listing ({len(flat)} total files)")

    raw = input("Exclude indexes: ").strip()
    if not raw:
        return set()

    excluded: set[str] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token.isdigit():
            continue
        idx = int(token)
        if 1 <= idx <= len(flat):
            excluded.add(flat[idx - 1][1])
    return excluded


def apply_exclusions(report: dict[str, Any], exclusions: set[str]) -> dict[str, Any]:
    if not exclusions:
        return report

    updated_manifest: dict[str, list[dict[str, Any]]] = {}
    for category, items in report["manifest"].items():
        updated_manifest[category] = [item for item in items if item["path"] not in exclusions]

    total_files = sum(len(items) for items in updated_manifest.values())
    total_size = sum(item["size_bytes"] for items in updated_manifest.values() for item in items)

    report["manifest"] = updated_manifest
    report["summary"]["total_files"] = total_files
    report["summary"]["total_size_bytes"] = total_size
    report["summary"]["total_size_human"] = human_size(total_size)
    report["issues"]["warnings"].append(f"User excluded {len(exclusions)} files in interactive mode")
    report["interactive_exclusions"] = sorted(exclusions)
    return report


def generate_checklist(report: dict[str, Any]) -> list[dict[str, Any]]:
    issues = report["issues"]
    checks = [
        {
            "name": "Required files present",
            "ok": len(issues["missing_required"]) == 0,
            "detail": "All required files found" if not issues["missing_required"] else ", ".join(issues["missing_required"]),
        },
        {
            "name": "No sensitive data detected",
            "ok": len(issues["sensitive_hits"]) == 0,
            "detail": "No key/password patterns found" if not issues["sensitive_hits"] else f"{len(issues['sensitive_hits'])} potential hits",
        },
        {
            "name": "No generated artifacts staged",
            "ok": len(issues["should_be_ignored_but_not"]) == 0,
            "detail": "Gitignore patterns effective" if not issues["should_be_ignored_but_not"] else f"{len(issues['should_be_ignored_but_not'])} files need ignore rules",
        },
        {
            "name": "Python package init files",
            "ok": len(issues["missing_inits"]) == 0,
            "detail": "All package directories contain __init__.py" if not issues["missing_inits"] else f"Missing in {len(issues['missing_inits'])} directories",
        },
        {
            "name": "Python header/docstring check",
            "ok": len(issues["python_header_missing"]) == 0,
            "detail": "All Python files contain header/docstring" if not issues["python_header_missing"] else f"{len(issues['python_header_missing'])} files need review",
        },
        {
            "name": "Large file check",
            "ok": report["summary"]["total_size_bytes"] <= 500 * 1024 * 1024,
            "detail": f"Manifest size {report['summary']['total_size_human']}",
        },
    ]
    return checks


def save_reports(root: Path, report: dict[str, Any], checklist: list[dict[str, Any]], output_dir: Path, logger) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "initial_commit_manifest.json"
    checklist_path = output_dir / "pre_commit_checklist.json"
    summary_path = output_dir / "initial_commit_summary.md"

    write_json(report_path, report)
    write_json(checklist_path, {"timestamp": utc_now(), "checks": checklist})

    lines = [
        "# Initial Commit Preparation Summary",
        "",
        f"- Generated: {utc_now()}",
        f"- Total files: {report['summary']['total_files']}",
        f"- Total size: {report['summary']['total_size_human']}",
        "",
        "## Category Breakdown",
    ]
    for category, count in sorted(report["summary"]["category_counts"].items()):
        lines.append(f"- {category}: {count}")

    lines.extend(["", "## Checklist"])
    for check in checklist:
        status = "PASS" if check["ok"] else "WARN"
        lines.append(f"- [{status}] {check['name']}: {check['detail']}")

    if report["issues"]["warnings"]:
        lines.extend(["", "## Warnings"])
        for warning in report["issues"]["warnings"]:
            lines.append(f"- {warning}")

    write_text(summary_path, "\n".join(lines) + "\n")
    logger.info("Wrote reports: %s, %s, %s", report_path, checklist_path, summary_path)


def prepare_repository(
    dry_run: bool = False,
    interactive: bool = False,
    output_dir: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = repo_root()
    logger = setup_logger("prepare_initial_commit")
    logger.info("Scanning repository: %s", root)

    report = scan_repository(root, logger)

    if interactive:
        excluded = interactive_exclusions(report)
        if excluded:
            report = apply_exclusions(report, excluded)

    checklist = generate_checklist(report)
    out_dir = output_dir or (root / "logs" / "bootstrap")
    save_reports(root, report, checklist, out_dir, logger)

    if dry_run:
        logger.info("Dry-run mode enabled: no staging or commit actions were performed.")

    return report, checklist


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare repository files for initial commit")
    parser.add_argument("--dry-run", action="store_true", help="Preview only; do not modify repository")
    parser.add_argument("--interactive", action="store_true", help="Interactive review and manual exclusions")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/bootstrap"),
        help="Directory for manifest/checklist reports",
    )
    args = parser.parse_args()

    report, checklist = prepare_repository(
        dry_run=args.dry_run,
        interactive=args.interactive,
        output_dir=args.output_dir,
    )

    critical = 0
    if report["issues"]["sensitive_hits"]:
        critical += 1
    if report["issues"]["missing_required"]:
        critical += 1

    print(json.dumps({"summary": report["summary"], "checklist": checklist[:6]}, indent=2))
    return 1 if critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
