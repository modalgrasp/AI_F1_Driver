#!/usr/bin/env python3
"""Generate comprehensive Phase 1 completion reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bootstrap_common import repo_root, run_cmd, setup_logger, system_snapshot, utc_now, write_json, write_text


def safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect_package_versions(packages: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        code = f"import importlib; m=importlib.import_module('{package}'); print(getattr(m, '__version__', 'unknown'))"
        result = run_cmd([sys.executable, "-c", code], cwd=repo_root())
        versions[package] = result.stdout.strip() if result.returncode == 0 else "not-installed"
    return versions


def code_stats(root: Path) -> dict[str, Any]:
    py_files = list(root.rglob("*.py"))
    total_lines = 0
    for path in py_files:
        try:
            total_lines += len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
        except OSError:
            pass
    return {
        "python_files": len(py_files),
        "python_lines": total_lines,
        "total_files": sum(1 for path in root.rglob("*") if path.is_file()),
    }


def git_info(root: Path) -> dict[str, Any]:
    commit = run_cmd(["git", "rev-parse", "HEAD"], cwd=root)
    count = run_cmd(["git", "rev-list", "--count", "HEAD"], cwd=root)
    remotes = run_cmd(["git", "remote", "-v"], cwd=root)
    tags = run_cmd(["git", "tag", "--list"], cwd=root)
    return {
        "head": commit.stdout if commit.returncode == 0 else "unknown",
        "commit_count": int(count.stdout) if count.returncode == 0 and count.stdout.isdigit() else 0,
        "remotes": remotes.stdout.splitlines() if remotes.returncode == 0 else [],
        "tags": tags.stdout.splitlines() if tags.returncode == 0 else [],
    }


def build_payload(root: Path) -> dict[str, Any]:
    tests = safe_json(root / "logs" / "bootstrap" / "phase1_test_report.json")
    commit = safe_json(root / "logs" / "bootstrap" / "initial_commit_report.json")
    remote = safe_json(root / "logs" / "bootstrap" / "remote_setup_report.json")
    health = safe_json(root / "logs" / "bootstrap" / "repository_health_report.json")

    packages = collect_package_versions(["torch", "numpy", "gymnasium", "pytest"])

    return {
        "generated_at": utc_now(),
        "system": system_snapshot(),
        "packages": packages,
        "installation_summary": {
            "phase": "Phase 1",
            "steps_completed": ["1.1", "1.2", "1.3", "1.4", "1.5"],
            "components": [
                "Environment setup",
                "Yas Marina track pipeline",
                "GPU optimization toolkit",
                "Repository structure and CI",
                "Bootstrap finalization",
            ],
        },
        "test_summary": tests.get("summary", {}),
        "project_stats": code_stats(root),
        "git": git_info(root),
        "reports": {
            "initial_commit": commit,
            "remote": remote,
            "health": health,
        },
        "phase1_checklist": {
            "environment_setup": True,
            "track_installation": True,
            "gpu_configuration": True,
            "repository_initialized": bool(git_info(root).get("head") != "unknown"),
            "tests_available": bool(tests),
        },
        "next_steps": [
            "Phase 2: Implement vehicle dynamics model",
            "Add Pacejka tire model",
            "Add aerodynamics + powertrain simulation",
            "Integrate dynamics model into training reward and state",
        ],
        "known_issues": health.get("issues", []),
        "acknowledgments": [
            "Assetto Corsa community",
            "Gymnasium project",
            "PyTorch ecosystem",
            "pre-commit framework",
        ],
    }


def write_outputs(root: Path, payload: dict[str, Any], output_dir: Path, include_pdf: bool) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "phase1_completion_report.json"
    md_path = output_dir / "phase1_completion_report.md"
    html_path = output_dir / "phase1_completion_report.html"

    write_json(json_path, payload)

    md_lines = [
        "# Phase 1 Completion Report",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Platform: {payload['system']['platform']}",
        f"- Python: {payload['system']['python'].split()[0]}",
        "",
        "## Test Summary",
        f"- {payload['test_summary']}",
        "",
        "## Git",
        f"- HEAD: {payload['git']['head']}",
        f"- Commits: {payload['git']['commit_count']}",
        "",
        "## Phase 1 Checklist",
    ]
    for key, value in payload["phase1_checklist"].items():
        mark = "x" if value else " "
        md_lines.append(f"- [{mark}] {key}")

    md_lines.extend(["", "## Next Steps"])
    for step in payload["next_steps"]:
        md_lines.append(f"- {step}")

    write_text(md_path, "\n".join(md_lines) + "\n")

    html = [
        "<html><head><title>Phase 1 Completion Report</title></head><body>",
        "<h1>Phase 1 Completion Report</h1>",
        f"<p>Generated: {payload['generated_at']}</p>",
        "<h2>Test Summary</h2>",
        f"<pre>{json.dumps(payload['test_summary'], indent=2)}</pre>",
        "<h2>Next Steps</h2><ul>",
    ]
    html.extend([f"<li>{step}</li>" for step in payload["next_steps"]])
    html.append("</ul></body></html>")
    write_text(html_path, "\n".join(html))

    outputs = {
        "json": str(json_path),
        "markdown": str(md_path),
        "html": str(html_path),
    }

    if include_pdf:
        pdf_path = output_dir / "phase1_completion_report.pdf"
        try:
            from reportlab.lib.pagesizes import A4  # type: ignore
            from reportlab.pdfgen import canvas  # type: ignore

            c = canvas.Canvas(str(pdf_path), pagesize=A4)
            text = c.beginText(40, 800)
            text.textLine("Phase 1 Completion Report")
            text.textLine(f"Generated: {payload['generated_at']}")
            text.textLine(f"HEAD: {payload['git']['head']}")
            for step in payload["next_steps"]:
                text.textLine(f"- {step}")
            c.drawText(text)
            c.save()
            outputs["pdf"] = str(pdf_path)
        except Exception:
            outputs["pdf"] = "not-generated (reportlab missing)"

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 1 completion report")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/reports"))
    parser.add_argument("--include-pdf", action="store_true")
    args = parser.parse_args()

    logger = setup_logger("generate_phase1_report")
    root = repo_root()

    payload = build_payload(root)
    outputs = write_outputs(root, payload, args.output_dir, args.include_pdf)
    logger.info("Generated Phase 1 report outputs: %s", outputs)
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
