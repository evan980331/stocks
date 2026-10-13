@echo off
REM ===================================================================
REM  台股本地量化系統 - Windows 一鍵啟動選單
REM  所有服務僅綁定 127.0.0.1，不開放外網，不使用任何雲端部署。
REM ===================================================================
setlocal
cd /d "%~dp0"

set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
  echo [!] 找不到虛擬環境 .venv，請先執行：
  echo     python -m venv .venv
  echo     .venv\Scripts\activate
  echo     pip install -r requirements.txt
  pause
  exit /b 1
)

:menu
cls
echo ============================================
echo   台股本地量化系統
echo ============================================
echo   [1] 抓取資料        ingest
echo   [2] 特徵與訊號      run
echo   [3] 回測            backtest
echo   [4] 啟動儀表板      start dashboard
echo   [5] 啟動監控 Worker start worker
echo   [6] 系統健康狀態    health
echo   [7] 服務狀態        status
echo   [8] 停止所有服務    stop all
echo   [0] 離開
echo ============================================
set /p choice=請選擇：

if "%choice%"=="1" "%PY%" -m twstock.cli ingest
if "%choice%"=="2" "%PY%" -m twstock.cli run
if "%choice%"=="3" "%PY%" -m twstock.cli backtest
if "%choice%"=="4" "%PY%" -m twstock.cli start dashboard & echo 儀表板：http://127.0.0.1:8000
if "%choice%"=="5" "%PY%" -m twstock.cli start worker
if "%choice%"=="6" "%PY%" -m twstock.cli health
if "%choice%"=="7" "%PY%" -m twstock.cli status
if "%choice%"=="8" "%PY%" -m twstock.cli stop all
if "%choice%"=="0" exit /b 0

echo.
pause
goto menu
