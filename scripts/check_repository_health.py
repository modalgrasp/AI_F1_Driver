#!/usr/bin/env python3
"""Repository health monitor with scoring and actionable recommendations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bootstrap_common import is_binary_file, repo_root, run_cmd, setup_logger, utc_now, write_json, write_text


SECRET_RE = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]"),
    re.compile(r"(?i)password\s*[:=]\s*['\"].{4,}['\"]"),
]

REQUIRED_DIRS = [
    "configs",
    "docs",
    "environments",
    "models",
    "scripts",
    "tests",
    "training",
    "utils",
    "vehicle_dynamics",
]


def health_checks(root: Path, quick: bool) -> dict[str, Any]:
    score = 100
    issues: list[dict[str, Any]] = []
    recommendations: list[str] = []

    for rel in REQUIRED_DIRS:
        if not (root / rel).exists():
            score -= 5
            issues.append({"severity": "high", "kind": "missing_directory", "path": rel})

    # Secrets scan (text files only, first 1 MB)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith((".git/", "f1_racing_env/", ".venv/")):
            continue
        if is_binary_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:1024 * 1024]
        except OSError:
            continue
        for pattern in SECRET_RE:
            if pattern.search(text):
                score -= 15
                issues.append({"severity": "critical", "kind": "potential_secret", "path": rel})
                recommendations.append("Remove secrets and rotate exposed keys immediately.")
                break

    # Large files check
    for path in root.rglob("*"):
        if path.is_file() and path.stat().st_size > 50 * 1024 * 1024:
            rel = path.relative_to(root).as_posix()
            if not rel.startswith(".git/"):
                score -= 4
                issues.append({"severity": "medium", "kind": "large_file", "path": rel, "size": path.stat().st_size})

    # Git hygiene
    status = run_cmd(["git", "status", "--porcelain"], cwd=root)
    if status.returncode != 0:
        score -= 20
        issues.append({"severity": "high", "kind": "git_status_failed"})
    elif status.stdout.strip():
        score -= 8
        issues.append({"severity": "medium", "kind": "working_tree_dirty", "count": len(status.stdout.splitlines())})
        recommendations.append("Commit or stash pending changes to keep repository clean.")

    # Dependency vulnerability quick check
    if not quick:
        audit = run_cmd(["python", "-m", "pip", "audit"], cwd=root)
        if audit.returncode != 0 and "No known vulnerabilities" not in (audit.stdout + audit.stderr):
            score -= 6
            issues.append({"severity": "medium", "kind": "dependency_vulnerabilities", "detail": (audit.stdout + audit.stderr)[:400]})

    # Basic docs link check
    readme = root / "README.md"
    if readme.exists():
        content = readme.read_text(encoding="utf-8", errors="ignore")
        refs = re.findall(r"\(([^)]+\.md)\)", content)
        broken = [ref for ref in refs if not (root / ref).exists()]
        if broken:
            score -= 3
            issues.append({"severity": "low", "kind": "broken_doc_links", "count": len(broken), "examples": broken[:10]})

    score = max(0, min(100, score))
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"

    return {
        "timestamp": utc_now(),
        "score": score,
        "grade": grade,
        "issues": issues,
        "recommendations": sorted(set(recommendations)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check repository health and generate score")
    parser.add_argument("--quick", action="store_true", help="Skip slower checks")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/bootstrap"))
    args = parser.parse_args()

    logger = setup_logger("check_repository_health")
    root = repo_root()
    report = health_checks(root, quick=args.quick)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "repository_health_report.json"
    md_path = args.output_dir / "repository_health_report.md"
    write_json(json_path, report)

    lines = [
        "# Repository Health Report",
        "",
        f"- Score: {report['score']}/100 ({report['grade']})",
        f"- Generated: {report['timestamp']}",
        "",
        "## Issues",
    ]
    if report["issues"]:
        for issue in report["issues"]:
            lines.append(f"- [{issue.get('severity','info').upper()}] {issue.get('kind')} {issue.get('path','')}")
    else:
        lines.append("- No issues found")

    lines.extend(["", "## Recommendations"])
    if report["recommendations"]:
        for rec in report["recommendations"]:
            lines.append(f"- {rec}")
    else:
        lines.append("- None")

    write_text(md_path, "\n".join(lines) + "\n")
    logger.info("Health score: %d", report["score"])
    print(json.dumps(report, indent=2))
    return 0 if report["score"] >= 90 else 1


if __name__ == "__main__":
    raise SystemExit(main())
