@echo off
REM ============================================
REM JEANIROTRABOT - Build Standalone .exe
REM ============================================

echo [1/3] Installing dependencies...
pip install -r requirements.txt pyinstaller
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo [2/3] Building JEANIROTRABOT.exe...
pyinstaller ^
    --onefile ^
    --windowed ^
    --name=JEANIROTRABOT ^
    --hidden-import=ttkbootstrap ^
    --hidden-import=MetaTrader5 ^
    --hidden-import=openai ^
    --hidden-import=mplfinance ^
    --hidden-import=pandas._libs.tslibs.base ^
    --hidden-import=numpy ^
    main.py

if errorlevel 1 (
    echo ERROR: Build failed.
    pause
    exit /b 1
)

echo [3/3] Done!
echo.
echo Output: dist\JEANIROTRABOT.exe
echo.
echo Cara pakai:
echo   1. Copy JEANIROTRABOT.exe ke folder mana saja
echo   2. Pastikan MetaTrader 5 sudah terinstal dan running
echo   3. Double-click JEANIROTRABOT.exe
echo   4. Isi credentials MT5 di GUI, klik Connect
echo   5. (Opsional) Isi OpenAI API Key untuk AI Agent
echo   6. Klik Start Robot - selesai!
echo.
pause
