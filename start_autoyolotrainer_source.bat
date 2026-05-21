@echo off
setlocal EnableExtensions

set "APP_DIR=%~dp0"
set "CONDA_ENV_NAME=yolo_stable"
set "CONDA_ENV_DIR="
set "MAIN_FILE=%APP_DIR%Code\main.py"

if not "%~1"=="" (
    set "CONDA_ENV_DIR=%~1"
) else if exist "%USERPROFILE%\miniconda3\envs\%CONDA_ENV_NAME%\python.exe" (
    set "CONDA_ENV_DIR=%USERPROFILE%\miniconda3\envs\%CONDA_ENV_NAME%"
) else if exist "%USERPROFILE%\anaconda3\envs\%CONDA_ENV_NAME%\python.exe" (
    set "CONDA_ENV_DIR=%USERPROFILE%\anaconda3\envs\%CONDA_ENV_NAME%"
) else if exist "C:\Users\innom\miniconda3\envs\%CONDA_ENV_NAME%\python.exe" (
    set "CONDA_ENV_DIR=C:\Users\innom\miniconda3\envs\%CONDA_ENV_NAME%"
)

echo [AutoYoloTrainer] Starting from source.
echo.

if not exist "%MAIN_FILE%" (
    echo [ERROR] Source entry file was not found.
    echo Path "%MAIN_FILE%"
    echo.
    echo This launcher requires the Code folder.
    pause
    exit /b 1
)

if defined CONDA_ENV_DIR (
    if exist "%CONDA_ENV_DIR%\python.exe" (
        set "PYTHON_EXE=%CONDA_ENV_DIR%\python.exe"
        set "AUTOYOLO_PYTHON_EXE=%CONDA_ENV_DIR%\python.exe"
        set "AUTOYOLO_CONDA_ENV_DIR=%CONDA_ENV_DIR%"
    )
)

if not defined PYTHON_EXE (
    where python > nul 2> nul
    if not errorlevel 1 (
        set "PYTHON_EXE=python"
    ) else (
        where py > nul 2> nul
        if not errorlevel 1 (
            set "PYTHON_EXE=py -3"
        ) else (
            echo [ERROR] Python was not found.
            pause
            exit /b 1
        )
    )
)

echo [AutoYoloTrainer] Python:
echo %PYTHON_EXE%
echo.

cd /d "%APP_DIR%Code"
%PYTHON_EXE% "%MAIN_FILE%"

if errorlevel 1 (
    echo.
    echo [ERROR] AutoYoloTrainer exited with error code %errorlevel%.
    pause
    exit /b %errorlevel%
)

endlocal
