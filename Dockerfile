# Используем официальный образ Python 3.12
FROM python:3.12-slim

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем все файлы проекта в контейнер
COPY . .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Создаем пустой __init__.py в src для корректной работы импортов
RUN touch src/__init__.py

# Устанавливаем PYTHONPATH
ENV PYTHONPATH=/app

# Запускаем скрипт при старте контейнера
CMD ["python", "main.py"]
