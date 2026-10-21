#!/usr/bin/env python3
"""
Kampüs Altyapı Kusur Tespit Sistemi - Başlatma Scripti
Çalıştır: python start.py
Tarayıcıda aç: http://localhost:5000
"""
import subprocess, sys, os

def check_deps():
    required = ['flask', 'ultralytics', 'cv2', 'pandas', 'openpyxl', 'reportlab', 'PIL']
    missing = []
    for pkg in required:
        try: __import__(pkg)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"[!] Eksik paketler: {missing}")
        print("[*] Yükleniyor...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
            'flask', 'ultralytics', 'opencv-python-headless',
            'pandas', 'openpyxl', 'reportlab', 'Pillow'])

if __name__ == '__main__':
    check_deps()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("\n" + "="*50)
    print("  Kampüs Altyapı Kusur Tespit Sistemi")
    print("="*50)
    print("  Tarayıcıda aç: http://localhost:5000")
    print("  Durdurmak için: Ctrl+C")
    print("="*50 + "\n")
    from app import app
    app.run(debug=False, host='0.0.0.0', port=5000)
