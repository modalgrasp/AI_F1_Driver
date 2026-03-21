#!/usr/bin/env python3
"""Update bootstrap-related documentation after repository setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bootstrap_common import repo_root, setup_logger, write_text


BADGE_BLOCK_START = "<!-- BOOTSTRAP_BADGES_START -->"
BADGE_BLOCK_END = "<!-- BOOTSTRAP_BADGES_END -->"


def update_readme(readme: Path, repo_url: str | None) -> None:
    text = readme.read_text(encoding="utf-8")

    badges = [
        BADGE_BLOCK_START,
        "![Build](https://img.shields.io/badge/build-bootstrap_ready-brightgreen)",
        "![Coverage](https://img.shields.io/badge/coverage-pending-lightgrey)",
        "![Docs](https://img.shields.io/badge/docs-updated-blue)",
    ]
    if repo_url:
        badges.append(f"![Repository](https://img.shields.io/badge/repo-configured-success)")
    badges.append(BADGE_BLOCK_END)
    badge_text = "\n".join(badges)

    if BADGE_BLOCK_START in text and BADGE_BLOCK_END in text:
        start = text.index(BADGE_BLOCK_START)
        end = text.index(BADGE_BLOCK_END) + len(BADGE_BLOCK_END)
        text = text[:start] + badge_text + text[end:]
    else:
        text += "\n\n" + badge_text + "\n"

    workflow_section = """
## Git Workflow
- Create feature branches from `main`
- Open pull requests for all changes
- Ensure pre-commit and CI checks pass before merge
- Use semantic commit messages: `type(scope): summary`

## Bootstrap Commands
```bash
python scripts/bootstrap_repository.py --dry-run
python scripts/bootstrap_repository.py --user-name YOUR_NAME --user-email YOUR_EMAIL --remote-url YOUR_REMOTE
```
"""

    if "## Git Workflow" not in text:
        text += "\n" + workflow_section

    write_text(readme, text)


def write_quickstart(root: Path, repo_url: str | None) -> Path:
    path = root / "docs" / "reports" / "developer_quick_start.md"
    lines = [
        "# Developer Quick Start",
        "",
        "## 1. Clone Repository",
    ]
    if repo_url:
        lines.append(f"```bash\ngit clone {repo_url}\ncd F1\n```")
    else:
        lines.append("```bash\n# use your repository URL\ngit clone <repo-url>\ncd F1\n```")

    lines.extend(
        [
            "",
            "## 2. Environment Setup",
            "```bash\npython -m venv f1_racing_env\n# activate venv\npip install -r requirements-dev.txt\n```",
            "",
            "## 3. Quality Checks",
            "```bash\npre-commit install --install-hooks\npre-commit run --all-files\npython scripts/run_phase1_tests.py\n```",
            "",
            "## 4. Contribution Flow",
            "- Create branch: `git checkout -b feat/your-change`",
            "- Commit with semantic message",
            "- Push and open pull request",
        ]
    )
    write_text(path, "\n".join(lines) + "\n")
    return path


def write_workflow_diagram(root: Path) -> Path:
    path = root / "docs" / "reports" / "contribution_workflow.md"
    lines = [
        "# Contribution Workflow Diagram",
        "",
        "```mermaid",
        "flowchart LR",
        "A[Clone Repository] --> B[Create Feature Branch]",
        "B --> C[Implement Changes]",
        "C --> D[Run pre-commit + tests]",
        "D --> E[Commit + Push]",
        "E --> F[Open Pull Request]",
        "F --> G[CI Checks + Review]",
        "G --> H[Merge to main]",
        "```",
    ]
    write_text(path, "\n".join(lines) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Update documentation after repository bootstrap")
    parser.add_argument("--repo-url", default=None)
    args = parser.parse_args()

    logger = setup_logger("update_bootstrap_docs")
    root = repo_root()

    update_readme(root / "README.md", args.repo_url)
    quick = write_quickstart(root, args.repo_url)
    diagram = write_workflow_diagram(root)

    report = {
        "updated": ["README.md", str(quick), str(diagram)],
    }
    write_text(root / "docs" / "reports" / "bootstrap_docs_update.json", json.dumps(report, indent=2))
    logger.info("Bootstrap docs updated.")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
