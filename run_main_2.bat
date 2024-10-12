@echo off
setlocal enabledelayedexpansion

set epoch=40
set lr=0.001

for /l %%a in (3,-1,3) do (
    set /a alpha=%%a
    set alpha=0.!alpha!
    
    for /l %%b in (5,-1,5) do (
        set /a beta=%%b
        set beta=0.!beta!
        
        echo Running with alpha=!alpha!, beta=!beta!, epoch=%epoch%, lr=%lr%
        c:/Users/USER2024/project_hepatic_AI_model/.venv/Scripts/python.exe main_2.py !alpha! !beta! %epoch% %lr%
        
        echo.
    )
)

echo All runs completed.
pause