@echo off
setlocal

REM Bu .bat dosyasi kampus_app klasorunun icine konacak.
REM Dosyanin bulundugu klasore gec:
cd /d "%~dp0"

echo ==========================================
echo Kampus App kurulumu ve baslatma
echo ==========================================
echo.

REM Python komutunu kontrol et
where python >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=python"
) else (
    where py >nul 2>nul
    if %errorlevel%==0 (
        set "PYTHON_CMD=py"
    ) else (
        echo HATA: Python bulunamadi.
        echo Python kurulu degilse once Python kur.
        pause
        exit /b 1
    )
)

echo [1/3] Bagimliliklar kuruluyor...
%PYTHON_CMD% -m pip install flask ultralytics opencv-python-headless pandas openpyxl reportlab Pillow

if errorlevel 1 (
    echo.
    echo HATA: Bagimlilik kurulumu basarisiz oldu.
    pause
    exit /b 1
)

echo.
echo [2/3] Uygulama klasoru hazir: %cd%
echo.

echo [3/3] Uygulama baslatiliyor...
echo Tarayici birazdan acilacak: http://localhost:5000
echo Kapatmak icin bu pencerede CTRL + C yapabilirsin.
echo.

REM Flask acilana kadar kisa bekleyip tarayiciyi ayri pencereden ac
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5000"

%PYTHON_CMD% start.py

echo.
echo Uygulama kapandi.
pause
