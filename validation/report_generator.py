#!/usr/bin/env python3
"""Validation report generation utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ValidationReportGenerator:
    """Generate markdown, json, and html validation reports."""

    def __init__(self, validation_results: dict[str, Any]) -> None:
        self.results = validation_results

    def generate_markdown_report(self, output_path: str | Path) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        summary = self.results["summary"]
        systems = self.results["results"]

        lines: list[str] = []
        lines.append("# Vehicle Dynamics Validation Report\n")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        lines.append("## Summary\n")
        lines.append(
            f"Overall status: **{summary['subsystems_passed']}/{summary['subsystems_total']} subsystems passed**\n\n"
        )

        lines.append("### Confidence Scores\n")
        for name, score in summary["confidence"].items():
            lines.append(f"- {name}: {score:.2f}\n")
        lines.append("\n")

        lines.append("### Sensitivity Highlights\n")
        for item in summary["sensitivity"]["most_sensitive"]:
            grad = summary["sensitivity"]["local_gradients"][item]
            lines.append(f"- {item}: {grad:.3f}\n")
        lines.append("\n")

        for subsystem, payload in systems.items():
            status = "PASS" if payload["all_passed"] else "FAIL"
            icon = "✅" if payload["all_passed"] else "❌"
            lines.append(f"## {icon} {subsystem}: {status}\n")
            for test in payload["individual_tests"]:
                t_icon = "✅" if test["passed"] else "❌"
                lines.append(f"### {t_icon} {test['test_name']}\n")

                details = test.get("details", [])
                if details:
                    for detail in details:
                        lines.append(f"- {detail}\n")
                else:
                    lines.append("- No issues found.\n")

                metrics = test.get("metrics", {})
                if metrics:
                    lines.append("- Metrics:\n")
                    for mk, mv in metrics.items():
                        lines.append(f"  - {mk}: {mv}\n")
                lines.append("\n")

        out.write_text("".join(lines), encoding="utf-8")

    def generate_json_report(self, output_path: str | Path) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.results, indent=2), encoding="utf-8")

    def generate_html_report(self, output_path: str | Path) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        summary = self.results["summary"]
        systems = self.results["results"]

        rows = []
        for name, payload in systems.items():
            status = "PASS" if payload["all_passed"] else "FAIL"
            color = "#1f8b4c" if payload["all_passed"] else "#b42318"
            score = summary["confidence"].get(name, 0.0)
            rows.append(
                f"<tr><td>{name}</td><td style='color:{color};font-weight:700'>{status}</td><td>{score:.2f}</td></tr>"
            )

        html = f"""
<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>Validation Report</title>
  <style>
    body {{ font-family: Segoe UI, sans-serif; margin: 24px; color: #202124; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 900px; }}
    th, td {{ border: 1px solid #dadce0; padding: 8px 10px; text-align: left; }}
    th {{ background: #f1f3f4; }}
  </style>
</head>
<body>
  <h1>Vehicle Dynamics Validation Report</h1>
  <p>Subsystems passed: {summary['subsystems_passed']} / {summary['subsystems_total']}</p>
  <table>
    <thead><tr><th>Subsystem</th><th>Status</th><th>Confidence</th></tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
        out.write_text(html, encoding="utf-8")
