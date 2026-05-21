@echo off
setlocal

rem Installs Python packages required by AutoYoloTrainer.
set "APP_DIR=%~dp0"
set "REQ_FILE=%APP_DIR%Code\requirements.txt"
set "NO_PAUSE="
if /I "%~1"=="/nopause" set "NO_PAUSE=1"

echo [AutoYoloTrainer] Dependency installation started.
echo.

rem Find Python launcher first, then fallback to python.exe.
where py > nul 2> nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
) else (
    where python > nul 2> nul
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
    ) else (
        echo [ERROR] Python was not found.
        echo Install Python 3.10 or newer, then run this file again.
        call :wait_before_exit
        exit /b 1
    )
)

if not exist "%REQ_FILE%" (
    echo [ERROR] requirements.txt was not found.
    echo Path: "%REQ_FILE%"
    call :wait_before_exit
    exit /b 1
)

echo [1/4] Checking Python version
%PYTHON_CMD% --version
if errorlevel 1 (
    echo [ERROR] Failed to run Python.
    call :wait_before_exit
    exit /b 1
)

echo.
echo [2/4] Upgrading pip
%PYTHON_CMD% -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    call :wait_before_exit
    exit /b 1
)

echo.
echo [3/4] Installing AutoYoloTrainer packages
%PYTHON_CMD% -m pip install -r "%REQ_FILE%"
if errorlevel 1 (
    echo [ERROR] Failed to install packages.
    call :wait_before_exit
    exit /b 1
)

echo.
echo [4/4] Verifying installation
%PYTHON_CMD% -c "import PyQt6, psutil, pynvml, ultralytics, cv2, torch; print('PyQt6 OK'); print('Ultralytics OK'); print('Torch CUDA:', torch.cuda.is_available())"
if errorlevel 1 (
    echo [ERROR] Verification failed.
    call :wait_before_exit
    exit /b 1
)

echo.
echo [DONE] Dependency installation finished.
echo If GPU training is required, check that Torch CUDA is True.
call :wait_before_exit
exit /b 0

:wait_before_exit
if not defined NO_PAUSE pause
exit /b 0
