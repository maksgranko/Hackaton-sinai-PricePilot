@echo off
REM Скрипт для сборки и запуска PricePilot в Docker (Windows)

echo 🐳 Сборка PricePilot Docker контейнера...

REM Проверяем наличие необходимых файлов
if not exist "requirements.txt" (
    echo ❌ Файл requirements.txt не найден!
    pause
    exit /b 1
)

if not exist "app\main.py" (
    echo ❌ Файл app\main.py не найден!
    pause
    exit /b 1
)

REM Собираем образ
echo 📦 Сборка Docker образа...
docker build -t pricepilot:latest .

if %errorlevel% neq 0 (
    echo ❌ Ошибка при сборке образа!
    pause
    exit /b 1
)

echo ✅ Сборка завершена!

REM Останавливаем и удаляем старый контейнер (если есть)
echo 🧹 Очистка старых контейнеров...
docker stop pricepilot 2>nul
docker rm pricepilot 2>nul

REM Запускаем контейнер
echo 🚀 Запуск контейнера...
docker run -d ^
    --name pricepilot ^
    -p 8000:8000 ^
    -e SECRET_KEY="your-secret-key-here" ^
    -e TEST_USER_EMAIL="demo@example.com" ^
    -e TEST_USER_PASSWORD="demo" ^
    -e WEBUI_USERNAME="demo@example.com" ^
    -e WEBUI_PASSWORD="demo" ^
    -e REBUILD_CACHE=0 ^
    -v "%cd%\simple-train.csv:/app/simple-train.csv:ro" ^
    -v "%cd%\simple-train shorted.csv:/app/simple-train shorted.csv:ro" ^
    -v "%cd%\model_enhanced.joblib:/app/model_enhanced.joblib:ro" ^
    -v "%cd%\feature_names.joblib:/app/feature_names.joblib:ro" ^
    pricepilot:latest

if %errorlevel% neq 0 (
    echo ❌ Ошибка при запуске контейнера!
    pause
    exit /b 1
)

echo 🎉 PricePilot запущен!
echo 🌐 Откройте http://localhost:8000 в браузере
echo 📊 API документация: http://localhost:8000/docs
echo.
echo 📋 Полезные команды:
echo   docker logs pricepilot          - просмотр логов
echo   docker stop pricepilot          - остановка
echo   docker rm pricepilot            - удаление контейнера
echo   docker exec -it pricepilot bash - вход в контейнер
echo.
echo 🐳 Или используйте docker-compose:
echo   docker-compose up -d            - запуск
echo   docker-compose down             - остановка
echo   docker-compose logs -f          - просмотр логов
echo.
pause
