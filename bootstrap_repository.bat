@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo F1 Racing AI - Repository Bootstrap
echo Phase 1 Finalization
echo ==========================================
echo.

python scripts/bootstrap_repository.py %*
if %errorlevel% neq 0 (
    echo Bootstrap failed. See logs\bootstrap\bootstrap_summary.json
    exit /b 1
)

echo.
echo ==========================================
echo Bootstrap Finished
echo ==========================================
echo Review logs\bootstrap\bootstrap_summary.json for detailed status.
exit /b 0
