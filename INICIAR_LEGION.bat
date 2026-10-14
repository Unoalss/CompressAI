@echo off
setlocal
cd /d "%~dp0"
title LEGION AI - A Legiao dos 10.240 Especialistas
echo ========================================================
echo   INICIANDO LEGION AI (CHATBOT QUANTICO OFFLINE)
echo   Modelo: Qwen3.8-35B-A3B MoE Paginado (10.240 Experts)
echo   Interface: http://127.0.0.1:7777
echo ========================================================
start http://127.0.0.1:7777
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -u legion\server.py %*
) else (
    python -u legion\server.py %*
)
pause
