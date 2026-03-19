@echo off
REM Lancement rapide de sciRview (sans réinstallation)
call venv\Scripts\activate
rmdir /s /q app\__pycache__ 2>nul
rmdir /s /q ui\__pycache__ 2>nul
set PYTHONDONTWRITEBYTECODE=1
streamlit run ui/app.py
