@echo off
echo Creating Windows Scheduled Tasks for Weather Data Collection and Monitoring

:: Get the current directory
set "SCRIPT_DIR=%~dp0"

:: Create the data collection task
echo Creating data collection task...
schtasks /create /tn "Airfield Weather Data Collection" ^
    /tr "\"%SCRIPT_DIR%start_data_collection.bat\"" ^
    /sc minute ^
    /mo 30 ^
    /st 00:00 ^
    /ru "%USERNAME%" ^
    /f

if %ERRORLEVEL% NEQ 0 (
    echo Failed to create data collection task. Please run as administrator.
    goto :error
)

:: Create the monitoring task
echo Creating monitoring task...
schtasks /create /tn "Airfield Weather Data Monitoring" ^
    /tr "python \"%SCRIPT_DIR%monitor_collection.py\"" ^
    /sc hourly ^
    /mo 1 ^
    /st 00:30 ^
    /ru "%USERNAME%" ^
    /f

if %ERRORLEVEL% NEQ 0 (
    echo Failed to create monitoring task. Please run as administrator.
    goto :error
)

echo All tasks created successfully!
echo Data collection will run every 30 minutes
echo Monitoring will run every hour
echo You can view both tasks in Task Scheduler
goto :end

:error
echo Task creation failed. Please ensure you're running as administrator.

:end
pause
