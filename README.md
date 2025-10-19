# 🚕 PricePilot - Интеллектуальная система рекомендации цен для такси

## 🎯 Краткий тизер

**PricePilot** — интеллектуальная ML-система динамического ценообразования для такси-агрегаторов, которая анализирует исторические данные о заказах, поведение пользователей и водителей, используя машинное обучение (XGBoost + калибровка вероятностей) для предсказания оптимальной цены, максимизирующей вероятность принятия заказа. Система решает задачу бинарной классификации, строя кривую вероятности по всему диапазону цен и находя точку максимальной ожидаемой выгоды с учетом экономики топлива, временных паттернов и персонализации для всех участников рынка такси.

---

## 🚀 Быстрый старт

### Требования

- Python 3.10+
- 4 GB RAM минимум
- CSV файл с историческими данными (`simple-train.csv`) (если не приложены файлы ML)

### Установка

```bash
# 1. Загрузите корректную копию репозитория
https://github.com/maksgranko/Hackaton-sinai-PricePilot/archive/3435de4380d91a267afc86c13b0e876e666136e0.zip

# 2. Создайте виртуальное окружение
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
# ! Можно и без виртуального окружения, но в таком случае, в системе могут быть конфликты, ошибки !
# 3. Установите зависимости
pip install -r requirements.txt
```

---

## 📝 Пошаговая инструкция по запуску

### Шаг 1: Подготовка данных

Убедитесь, что в **корне проекта** находится файл `simple-train.csv`. Это может быть `train.csv`, просто переименованный — это файл с историческими данными для обучения модели.

```bash
# Если у вас train.csv, переименуйте его
mv train.csv simple-train.csv  # Linux/Mac
# или
ren train.csv simple-train.csv  # Windows
```

### Шаг 2: Обучение модели и создание кэша

**Важно:** Все команды выполняются **из корня проекта**!

```bash
# Построение кэша истории пользователей и водителей
# Создает: user_history.joblib, driver_history.joblib (опционально)
python ./src/build_history_cache.py

# Обучение ML-модели
# Создает: model_enhanced.joblib, feature_names.joblib
python ./main.py
```

После выполнения должны появиться файлы:

- `driver_history.joblib` (опционально)
- `feature_names.joblib`
- `model_enhanced.joblib`
- `user_history.joblib` (опционально)

> **Примечание:** Количество `.joblib` файлов может варьироваться в зависимости от конфигурации. Кэш истории (`user_history.joblib`, `driver_history.joblib`) опционален и может отсутствовать.

### Шаг 3: Запуск веб-интерфейса

```bash
# Запуск FastAPI сервера с автоперезагрузкой
uvicorn app.main:app --reload
```

Откройте браузер и перейдите на `http://127.0.0.1:8000`

### Шаг 4: Тестирование через интерфейс

В веб-интерфейсе доступно **левое бургер-меню** (иконка ☰) для тестирования API.

**Пример тестовых данных:**

```json
{
  "order_timestamp": 1718558240,
  "distance_in_meters": 3404,
  "duration_in_seconds": 486,
  "pickup_in_meters": 790,
  "pickup_in_seconds": 169,
  "driver_rating": 5.0,
  "platform": "android",
  "price_start_local": 180.0,
  "carname": "LADA",
  "carmodel": "GRANTA",
  "driver_reg_date": "2020-01-15",
  "user_id": 12345,
  "driver_id": 67890
}
```

**Опциональные поля:**

- `user_id`, `driver_id` - для персонализации (используется кэш)
- `carname`, `carmodel` - для определения класса такси
- По умолчанию в форме уже присутствуют базовые значения

---

## 📁 Структура проекта

```
Hackaton-sinai-PricePilot/
├── src/                     # ML-магия (обучение, предсказания)
│   ├── train_model.py       # обучение модели
│   ├── recommend_price.py   # рекомендация цен
│   └── build_history_cache.py  # построение кэша
├── app/                     # Web API (FastAPI)
│   ├── main.py              # главный эндпоинт
│   ├── services.py          # бизнес-логика
│   ├── schemas.py           # Pydantic схемы
│   ├── auth.py              # JWT аутентификация
│   └── config.py            # конфигурация
├── webui/                   # Веб-интерфейс
│   ├── templates/           # HTML шаблоны
│   └── static/              # CSS, JS, изображения
├── scripts/                 # Утилиты
│   └── mock_frontend.py     # тестовый клиент
├── main.py                  # ML-обучение (корень)
├── test_price_recommendation.py  # deprecated тесты
└── simple-train.csv         # данные для обучения
```

---

## 🐳 Docker развертывание

```bash
# Сборка и запуск
docker-compose up --build

# В фоновом режиме
docker-compose up -d
```

API будет доступен на `http://localhost:8000`

---

## 🧠 Архитектура ML-модели

### Ключевые возможности

✅ **Персонализация**: Учитывает историю заказов пользователя и ставок водителя  
✅ **Экономика топлива**: Расчет рентабельности с учетом расхода топлива  
✅ **Зоны ценообразования**: Разделение цен на 4 зоны (красная, желтая, зеленая)  
✅ **Временные паттерны**: Учет времени суток, дня недели, пиковых часов  
✅ **Калибровка вероятностей**: Точные предсказания с помощью CalibratedClassifierCV  

### Алгоритм

**XGBoost Classifier** (200 деревьев) + **Calibrated Classifier** (Sigmoid калибровка)

- **99 признаков** из 16 категорий
- **ROC-AUC**: ~0.75-0.85
- **Время предсказания**: ~4 секунды на запрос

### Основные категории признаков

- Ценовые признаки (7)
- Временные признаки (15)
- Признаки маршрута (12)
- Признаки подачи (7)
- Экономика топлива (12)
- История пользователя (6)
- История водителя (6)
- И другие...

---

## 📡 API документация

### Аутентификация

**Получение токена:**

```bash
curl -X POST "http://127.0.0.1:8000/auth/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=demo@example.com&password=demo"
```

**Демо-учетка:**

- Email: `demo@example.com`
- Пароль: `demo`

### Основные эндпоинты

- **GET** `/` - веб-интерфейс для водителей
- **POST** `/auth/token` - JWT аутентификация
- **POST** `/api/v1/orders/price-recommendation` - рекомендация цены
- **GET** `/health` - проверка статуса
- **GET** `/docs` - Swagger UI документация

---

## 🔗 Рекомендуемая версия

Рабочий коммит с продемонстрированным интерфейсом:

```
https://github.com/maksgranko/Hackaton-sinai-PricePilot/commit/e66f957ef220403f4f87ed40660313fc38daaf98
```

---

## 🛠️ Технологический стек

**Backend & API:**  
FastAPI, Uvicorn, PyJWT, httpx

**Machine Learning:**  
XGBoost, scikit-learn, pandas, numpy, scipy, joblib

**Frontend:**  
Vanilla JS (ES6+), CSS3, HTML5

**DevOps:**  
Docker, docker-compose

---

## 🤝 Авторы

- Разработано для **Hackathon**.
- Роль: Менеджер/Аналитик — Иван Лунин.
- Роль: Дизайнер (UX/UI) — Кирилл Опенченко.
- Роль: Data Scientist (ML-моделирование) — Максим Гранько.
- Роль: Backend-разработчик (API/Интеграция) — Виктор Волошко. 
- Роль: Программист (Data Engineering) — Олег Половинко.

---
Основные ссылки на Docker и GitHub:
Docker Hub: https://hub.docker.com/r/maksgranko/pricepilot
GitHub: https://github.com/maksgranko/Hackaton-sinai-PricePilot

**🚀 PricePilot** - Умное ценообразование для современного такси!
