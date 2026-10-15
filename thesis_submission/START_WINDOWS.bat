@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%~dp0install_and_run.py" %*
    goto finished
)
where python3 >nul 2>nul
if not errorlevel 1 (
    python3 "%~dp0install_and_run.py" %*
    goto finished
)
where python >nul 2>nul
if not errorlevel 1 (
    python "%~dp0install_and_run.py" %*
    goto finished
)
echo Python was not found. Install 64-bit Python 3.12 or newer and try again.
echo When installing Python, select "Add Python to PATH".
pause
exit /b 1
:finished
set "THESIS_EXIT_CODE=%errorlevel%"
echo.
if not "%THESIS_EXIT_CODE%"=="0" echo The analysis did not finish. See the message above and logs in the logs folder.
pause
exit /b %THESIS_EXIT_CODE%
