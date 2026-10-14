@echo off
setlocal
cd /d "%~dp0"
title VQ2 Quantum Studio - Qwen3.8-35B-A3B
echo ========================================================
echo   INICIANDO VQ2 QUANTUM STUDIO (OFFLINE)
echo   Modelo: Qwen3.8-35B-A3B VQ2 RDO (9.38 GB)
echo   GPU: NVIDIA GeForce RTX 3060 12GB
echo   Interface: http://127.0.0.1:8080
echo ========================================================
start http://127.0.0.1:8080
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -u web_app\server.py %*
) else (
    python -u web_app\server.py %*
)
pause
