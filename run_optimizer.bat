@echo off
setlocal enabledelayedexpansion

set alpha_list=0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1
set beta_list=0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1
set epoch_list=1
set lr_list=0.001
set log_name=temp.log9


:: date and time as the csv file name
for /f "tokens=1-6 delims=/-.: " %%a in ("%date% %time%") do (
    set month=%%b
    set day=%%c
    set hour=%%e
    set minute=%%f
)
set datetime_param=%month%%day%_%hour%%minute%

echo training start > %log_name%
for %%a in (%alpha_list%) do (
	for %%b in (%beta_list%) do (
		for %%e in (%epoch_list%) do (
			for %%l in (%lr_list%) do (
					echo Running with alpha=%%a, beta=%%b, epoch=%%e, lr=%%l, datatime=%datetime_param%
					c:/Users/USER2024/project_hepatic_AI_model/.venv/Scripts/python.exe optimizer.py %%a %%b %%e %%l %datetime_param% >> %log_name%
					echo.
			)
		)
	)
)

echo All runs completed.
endlocal
pause