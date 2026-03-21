#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "F1 Racing AI - Repository Bootstrap"
echo "Phase 1 Finalization"
echo "=========================================="
echo ""

python scripts/bootstrap_repository.py "$@"

echo ""
echo "=========================================="
echo "Bootstrap Finished"
echo "=========================================="
echo "Review logs/bootstrap/bootstrap_summary.json for detailed status."
