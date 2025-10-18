#!/bin/bash

# Скрипт для сборки и запуска PricePilot в Docker

set -e

echo "🐳 Сборка PricePilot Docker контейнера..."

# Проверяем наличие необходимых файлов
if [ ! -f "requirements.txt" ]; then
    echo "❌ Файл requirements.txt не найден!"
    exit 1
fi

if [ ! -f "app/main.py" ]; then
    echo "❌ Файл app/main.py не найден!"
    exit 1
fi

# Собираем образ
echo "📦 Сборка Docker образа..."
docker build -t pricepilot:latest .

echo "✅ Сборка завершена!"

# Запускаем контейнер
echo "🚀 Запуск контейнера..."
docker run -d \
    --name pricepilot \
    -p 8000:8000 \
    -e SECRET_KEY="your-secret-key-here" \
    -e TEST_USER_EMAIL="demo@example.com" \
    -e TEST_USER_PASSWORD="demo" \
    -e WEBUI_USERNAME="demo@example.com" \
    -e WEBUI_PASSWORD="demo" \
    -e REBUILD_CACHE=0 \
    -v "$(pwd)/simple-train.csv:/app/simple-train.csv:ro" \
    -v "$(pwd)/simple-train shorted.csv:/app/simple-train shorted.csv:ro" \
    -v "$(pwd)/model_enhanced.joblib:/app/model_enhanced.joblib:ro" \
    -v "$(pwd)/feature_names.joblib:/app/feature_names.joblib:ro" \
    pricepilot:latest

echo "🎉 PricePilot запущен!"
echo "🌐 Откройте http://localhost:8000 в браузере"
echo "📊 API документация: http://localhost:8000/docs"
echo ""
echo "📋 Полезные команды:"
echo "  docker logs pricepilot          - просмотр логов"
echo "  docker stop pricepilot          - остановка"
echo "  docker rm pricepilot            - удаление контейнера"
echo "  docker exec -it pricepilot bash - вход в контейнер"
echo ""
echo "🐳 Или используйте docker-compose:"
echo "  docker-compose up -d            - запуск"
echo "  docker-compose down             - остановка"
echo "  docker-compose logs -f          - просмотр логов"
