# Step 1.5 Bootstrap Usage

## One-command bootstrap

Windows:

```bat
bootstrap_repository.bat --user-name "YOUR_NAME" --user-email "YOUR_EMAIL" --remote-url "YOUR_REMOTE_URL"
```

Linux/macOS:

```bash
./bootstrap_repository.sh --user-name "YOUR_NAME" --user-email "YOUR_EMAIL" --remote-url "YOUR_REMOTE_URL"
```

Python direct:

```bash
python scripts/bootstrap_repository.py --user-name "YOUR_NAME" --user-email "YOUR_EMAIL" --remote-url "YOUR_REMOTE_URL"
```

## Safe preview

```bash
python scripts/bootstrap_repository.py --dry-run --interactive
```

## Component scripts

- scripts/prepare_initial_commit.py
- scripts/create_initial_commit.py
- scripts/setup_remote_repository.py
- scripts/install_precommit_hooks.py
- scripts/run_phase1_tests.py
- scripts/generate_phase1_report.py
- scripts/check_repository_health.py
- scripts/backup_repository.py
- scripts/validate_dev_environment.py
- scripts/update_bootstrap_docs.py
- scripts/post_bootstrap_validation.py

## Troubleshooting

1. If commit fails, run:
   - `python scripts/create_initial_commit.py --dry-run`
2. If remote push fails, re-run:
   - `python scripts/setup_remote_repository.py --interactive`
3. If hooks fail, run:
   - `python scripts/install_precommit_hooks.py --interactive --auto-stage`
4. If tests fail, inspect:
   - `logs/bootstrap/phase1_test_report.md`
