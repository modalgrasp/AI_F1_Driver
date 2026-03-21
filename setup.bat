@echo off
setlocal enabledelayedexpansion

REM -----------------------------------------------------------------
REM F1 Autonomous Racing AI - Windows setup script
REM -----------------------------------------------------------------

set "ENV_NAME=f1_racing_env"
set "REQUIREMENTS=requirements.txt"

call :info "Starting setup for F1 Autonomous Racing AI"

if /I not "%OS%"=="Windows_NT" (
    call :error "This setup.bat script is intended for Windows."
    exit /b 1
)

where py >nul 2>nul
if %errorlevel% neq 0 (
    where python >nul 2>nul
    if %errorlevel% neq 0 (
        call :error "Python launcher and python executable not found. Install Python 3.10+ and retry."
        exit /b 1
    ) else (
        set "PY_CMD=python"
    )
) else (
    set "PY_CMD=py -3"
)

for /f "tokens=1,2 delims=." %%a in ('%PY_CMD% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do (
    set "PY_MAJ=%%a"
    set "PY_MIN=%%b"
)

if "%PY_MAJ%"=="" (
    call :error "Could not determine Python version."
    exit /b 1
)

if %PY_MAJ% lss 3 (
    call :error "Python 3.10+ required."
    exit /b 1
)
if %PY_MAJ% equ 3 if %PY_MIN% lss 10 (
    call :error "Python 3.10+ required."
    exit /b 1
)
call :ok "Python version check passed: %PY_MAJ%.%PY_MIN%"

if exist "%ENV_NAME%" (
    call :warn "Virtual environment '%ENV_NAME%' already exists."
) else (
    call :info "Creating virtual environment '%ENV_NAME%'..."
    %PY_CMD% -m venv %ENV_NAME%
    if %errorlevel% neq 0 (
        call :error "Failed to create virtual environment."
        exit /b 1
    )
)

call "%ENV_NAME%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    call :error "Failed to activate virtual environment."
    exit /b 1
)
call :ok "Virtual environment activated."

python -m pip install --upgrade pip setuptools wheel
if %errorlevel% neq 0 (
    call :error "Failed to upgrade pip tooling."
    exit /b 1
)

call :info "Installing latest PyTorch + CUDA 13.0 wheels"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if %errorlevel% neq 0 (
    call :error "PyTorch CUDA installation failed."
    call :info "Diagnostics: check Python version support and NVIDIA driver compatibility."
    exit /b 1
)

if not exist "%REQUIREMENTS%" (
    call :error "requirements.txt not found in current directory."
    exit /b 1
)

call :info "Installing dependencies from requirements.txt"
pip install -r "%REQUIREMENTS%"
if %errorlevel% neq 0 (
    call :error "Dependency installation failed."
    call :info "Diagnostics: try 'pip install --verbose -r requirements.txt'"
    exit /b 1
)

where nvidia-smi >nul 2>nul
if %errorlevel% neq 0 (
    call :warn "nvidia-smi not detected. CUDA driver may be unavailable."
) else (
    for /f "tokens=*" %%x in ('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2^>nul') do (
        call :info "GPU: %%x"
    )
)

call :info "Validating PyTorch CUDA setup"
python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_version=', torch.version.cuda); print('cudnn_available=', torch.backends.cudnn.is_available()); import sys; sys.exit(0 if torch.cuda.is_available() else 2)"
set "TORCH_EXIT=%errorlevel%"
if %TORCH_EXIT% neq 0 (
    call :warn "PyTorch imported, but CUDA is not available to torch."
    call :warn "Check NVIDIA driver, CUDA toolkit compatibility, and torch CUDA wheel version."
) else (
    call :ok "PyTorch CUDA validation passed."
)

call :ok "Setup completed. Next: python validate_setup.py"
exit /b 0

:info
echo [INFO] %~1
exit /b 0

:ok
echo [ OK ] %~1
exit /b 0

:warn
echo [WARN] %~1
exit /b 0

:error
echo [FAIL] %~1
exit /b 0
