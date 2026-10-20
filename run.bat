@echo off
chcp 65001 >nul
title Meeting Companion
cd /d "%~dp0"

echo.
echo   Meeting Companion — локальный ИИ-секретарь встреч
echo   ================================================
echo.

set "PORTABLE_PY=%~dp0python-portable\python.exe"
set "INSTALLER=%~dp0tools\python-3.13.15-amd64.exe"

rem -- MSVC-рантаймы: на чистом Windows нет MSVCP140/VCRUNTIME140_THREADS,
rem    из-за чего torch падает с WinError 1114. Копируем их в портативный
rem    Python и в torch\lib (идемпотентно, только при наличии файлов).
if exist "%~dp0tools\vc-runtime\" (
    if exist "%PORTABLE_PY%" (
        copy /y "%~dp0tools\vc-runtime\*.dll" "%~dp0python-portable\" >nul 2>&1
        copy /y "%~dp0tools\vc-runtime\*.dll" "%~dp0python-portable\Lib\site-packages\torch\lib\" >nul 2>&1
    )
)

rem -- Способ 1: портативный Python внутри проекта (не требует установки) ----
if exist "%PORTABLE_PY%" (
    echo   Python: портативный (python-portable\)
    "%PORTABLE_PY%" main.py
    goto :after_run
)

rem -- Способ 2: системный Python в PATH -------------------------------------
where python >nul 2>&1
if not errorlevel 1 (
    echo   Python: системный
    python main.py
    goto :after_run
)

rem -- Способ 3: тихая установка Python из bundled-установщика ---------------
echo   Python не найден. Устанавливаю из tools\ (один раз, без интернета)...
if not exist "%INSTALLER%" (
    echo   [ОШИБКА] Установщик не найден: tools\python-3.13.15-amd64.exe
    echo   Скачайте его с python.org и положите в папку tools\.
    pause
    exit /b 1
)
"%INSTALLER%" /quiet InstallAllUsers=0 PrependPath=0 Include_test=0 Include_launcher=0 Include_doc=0
if errorlevel 1 (
    echo   [ОШИБКА] Установка Python не удалась.
    pause
    exit /b 1
)
echo   Установка завершена. Перезапустите run.bat.
pause
exit /b 0

:after_run
if errorlevel 1 (
    echo.
    echo   [ОШИБКА] Приложение завершилось с кодом %errorlevel%.
    pause
)