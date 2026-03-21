# Contributing to F1 Autonomous Racing AI

## Reporting Bugs
1. Use the bug issue template.
2. Include reproducible steps, expected vs actual behavior, and logs.
3. Include OS, Python version, GPU info, and commit hash.

## Suggesting Features
1. Use feature request template.
2. Explain motivation, expected impact, and alternatives.

## Development Workflow
1. Fork and clone repository.
2. Create branch:
   - feature/*
   - bugfix/*
   - hotfix/*
3. Install dependencies:
   - pip install -r requirements-dev.txt
4. Run checks:
   - pre-commit run --all-files
   - pytest -q
5. Open pull request with linked issue.

## Code Style Guidelines
1. PEP 8 compliant.
2. Type hints required for public functions/classes.
3. Google-style docstrings required.
4. Max line length is 88 (Black default).

## Testing Requirements
1. Add tests for all new features.
2. Add regression tests for bug fixes.
3. Maintain coverage above 80%.

## Commit Convention
Format:
<type>(<scope>): <subject>

Types:
- feat
- fix
- docs
- style
- refactor
- test
- chore

Example:
feat(environment): add tire temperature tracking

## Pull Request Process
1. Link relevant issue.
2. Describe technical changes and impact.
3. Include tests and docs updates.
4. Ensure CI passes.

## Code Review
1. At least one reviewer approval required.
2. Address all requested changes.
3. Prefer squash merge to keep history clean.
