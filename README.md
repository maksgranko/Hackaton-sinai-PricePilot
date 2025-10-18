# 🚕 PricePilot - Интеллектуальная система рекомендации цен для такси

**PricePilot** — это комплексная ML-система для динамического ценообразования в такси-агрегаторах. Система анализирует исторические данные о заказах, поведение пользователей и водителей, и предсказывает оптимальную цену для максимизации вероятности принятия заказа.

## 📋 Содержание

- [Обзор системы](#обзор-системы)
- [Архитектура ML-модели](#архитектура-ml-модели)
- [Признаки модели](#признаки-модели)
- [Установка и запуск](#установка-и-запуск)
- [API документация](#api-документация)
- [Веб-интерфейс](#веб-интерфейс)
- [Обучение модели](#обучение-модели)
- [Docker развертывание](#docker-развертывание)
- [Производительность](#производительность)

---

## 🎯 Обзор системы

**PricePilot** решает задачу **бинарной классификации**: предсказывает вероятность того, что водитель примет заказ (`is_done = done`) при заданной цене. Система строит **кривую вероятности** по всему диапазону цен и находит оптимальную точку, максимизирующую ожидаемую выгоду (Expected Value).

### Ключевые возможности

✅ **Персонализация**: Учитывает историю заказов пользователя и ставок водителя  
✅ **Экономика топлива**: Расчет рентабельности с учетом расхода топлива  
✅ **Зоны ценообразования**: Разделение цен на 4 зоны (красная, желтая, зеленая)  
✅ **Временные паттерны**: Учет времени суток, дня недели, пиковых часов  
✅ **Калибровка вероятностей**: Точные предсказания с помощью CalibratedClassifierCV  
✅ **Fast API**: RESTful API с JWT-аутентификацией и Swagger документацией  
✅ **Веб-интерфейс**: Интерактивная панель для водителей с визуализацией

---

## 🧠 Архитектура ML-модели

### Алгоритм: XGBoost + Calibrated Classifier

Модель состоит из двух этапов:

1. **XGBoost Classifier** (200 деревьев)
   - `learning_rate`: 0.05
   - `max_depth`: 6
   - `scale_pos_weight`: автоматический баланс классов
   - `tree_method`: hist (быстрый метод для больших данных)

2. **Calibrated Classifier** (Sigmoid калибровка)
   - Корректирует вероятности для точного предсказания
   - Использует метод Platt Scaling

### Метрики качества

- **ROC-AUC**: ~0.75-0.85 (зависит от данных)
- **PR-AUC**: ~0.70-0.80
- **Калибровка**: вероятности соответствуют реальным частотам

---

## 📊 Признаки модели

### Общее количество признаков: **99**

Модель использует 99 признаков, сгруппированных в 16 категорий:

#### 1. **Ценовые признаки (7)**
- `price_bid_local` - предложенная цена
- `price_start_local` - стартовая цена заказа
- `price_increase_abs` - абсолютная наценка
- `price_increase_pct` - процент наценки
- `is_price_increased` - флаг наценки
- `price_per_km` - цена за километр
- `price_per_minute` - цена за минуту

#### 2. **Временные признаки (15)**
- `hour_sin`, `hour_cos` - циклическое время суток
- `day_sin`, `day_cos` - циклический день недели
- `day_of_week`, `day_of_month` - календарные признаки
- `is_weekend` - выходной день
- `is_morning_peak` - утренний час пик (7-9)
- `is_evening_peak` - вечерний час пик (17-20)
- `is_peak_hour` - любой час пик
- `is_night` - ночное время (22-6)
- `is_lunch_time` - обеденное время (12-14)
- `is_month_start` - начало месяца (зарплата, дни 1-5)
- `is_month_end` - конец месяца (дни 25+)
- `hour_quartile` - квартиль часа (0-3)

#### 3. **Признаки маршрута (12)**
- `distance_in_meters`, `distance_km` - расстояние
- `duration_in_seconds`, `duration_min` - время в пути
- `avg_speed_kmh` - средняя скорость
- `is_traffic_jam` - пробка (<15 км/ч)
- `is_highway` - трасса (>50 км/ч)
- `is_short_trip` - короткая поездка (<2 км)
- `is_medium_trip` - средняя поездка (2-10 км)
- `is_long_trip` - длинная поездка (>10 км)
- `is_very_short` - очень короткая (<1 км)
- `is_very_long` - очень длинная (>20 км)

#### 4. **Признаки подачи (7)**
- `pickup_in_meters`, `pickup_km` - расстояние подачи
- `pickup_in_seconds` - время подачи
- `pickup_speed_kmh` - скорость подачи
- `pickup_to_trip_ratio` - соотношение подачи к поездке
- `pickup_time_ratio` - соотношение времени подачи
- `pickup_burden` - нагрузка подачи

#### 5. **Общие расстояния (2)**
- `total_distance` - общее расстояние (подача + поездка)
- `total_time` - общее время

#### 6. **Признаки водителя (7)**
- `driver_rating` - рейтинг водителя (1-5)
- `driver_experience_days` - опыт в днях
- `driver_experience_years` - опыт в годах
- `is_new_driver` - новичок (<30 дней)
- `is_experienced_driver` - опытный (>365 дней)
- `has_perfect_rating` - идеальный рейтинг (5.0)
- `rating_deviation` - отклонение от 5.0

#### 7. **Признаки отклика (4)**
- `response_time_seconds` - время отклика
- `response_time_log` - логарифм времени отклика
- `is_fast_response` - быстрый отклик (<10 сек)
- `is_slow_response` - медленный отклик (>60 сек)

#### 8. **Тип такси (3)**
- `taxi_type_economy` - эконом класс
- `taxi_type_comfort` - комфорт класс
- `taxi_type_business` - бизнес класс

#### 9. **Платформа (2)**
- `platform_android` - Android приложение
- `platform_ios` - iOS приложение

#### 10. **Взаимодействия базовых признаков (9)**
- `price_inc_x_distance` - наценка × расстояние
- `price_inc_x_night` - наценка × ночь
- `price_inc_x_peak` - наценка × час пик
- `price_inc_x_weekend` - наценка × выходные
- `distance_x_night` - расстояние × ночь
- `distance_x_weekend` - расстояние × выходные
- `distance_x_peak` - расстояние × час пик
- `speed_x_peak` - скорость × час пик
- `rating_x_price_inc` - рейтинг × наценка
- `experience_x_price_inc` - опыт × наценка

#### 11. **⛽ Экономика топлива (12)**
- `fuel_cost_rub` - стоимость топлива в рублях
- `fuel_liters` - расход топлива в литрах
- `price_to_fuel_ratio` - отношение цены к топливу
- `min_profitable_price` - минимальная рентабельная цена
- `price_above_min_profitable` - превышение минимума
- `price_above_min_profitable_pct` - превышение в процентах
- `is_highly_profitable` - высокая рентабельность (>2× топливо)
- `is_profitable` - рентабельно (>1.3× топливо)
- `is_unprofitable` - нерентабельно
- `net_profit` - чистая прибыль (цена - топливо)
- `net_profit_per_km` - прибыль на км
- `net_profit_per_minute` - прибыль на минуту

#### 12. **Взаимодействия топлива (3)**
- `fuel_ratio_x_distance` - топливный коэф. × расстояние
- `fuel_ratio_x_peak` - топливный коэф. × час пик
- `net_profit_x_rating` - прибыль × рейтинг

#### 13. **👤 История пользователя (6)**
- `user_order_count` - количество заказов
- `user_acceptance_rate` - процент принятых заказов
- `user_avg_price_ratio` - средний коэффициент цены
- `user_is_new` - новый пользователь (≤5 заказов)
- `user_is_vip` - VIP пользователь (≥20 заказов)
- `user_is_price_sensitive` - чувствителен к цене

#### 14. **🚗 История водителя (6)**
- `driver_bid_count` - количество ставок
- `driver_acceptance_rate` - процент принятых ставок
- `driver_avg_bid_ratio` - средний коэффициент ставки
- `driver_is_active` - активный водитель (≥20 ставок)
- `driver_is_aggressive` - агрессивные ставки (>1.2×)
- `driver_is_flexible` - гибкие ставки (<1.1×)

#### 15. **🔗 Взаимодействия истории (3)**
- `user_driver_match_score` - совпадение пользователя и водителя
- `price_vs_user_avg` - цена относительно средней пользователя
- `price_vs_driver_avg` - цена относительно средней водителя

#### 16. **🗺️ Эффективность маршрута (1)**
- `route_efficiency` - эффективность маршрута (км/мин)

---

## 🚀 Установка и запуск

### Требования

- Python 3.10+
- 4 GB RAM минимум (8 GB рекомендуется для обучения)
- CSV файл с историческими данными (`simple-train.csv`)

### Быстрый старт

```bash
# 1. Клонируйте репозиторий
git clone <repository-url>
cd Hackaton-sinai-PricePilot

# 2. Создайте виртуальное окружение
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Установите зависимости
pip install -r requirements.txt

# 4. (Опционально) Обучите модель
python src/train_model.py

# 5. Постройте кэш истории
python src/build_history_cache.py

# 6. Запустите API
uvicorn app.main:app --reload
```

API будет доступен по адресу `http://127.0.0.1:8000`  
Swagger документация: `http://127.0.0.1:8000/docs`

---

## 📡 API документация

### Аутентификация

Все эндпоинты защищены JWT токенами.

**Получение токена:**
```bash
curl -X POST "http://127.0.0.1:8000/auth/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=demo@example.com&password=demo"
```

**Ответ:**
```json
{
  "access_token": "eyJhbGc...",
  "token_type": "bearer"
}
```

**Демо-учетка:**
- Email: `demo@example.com`
- Пароль: `demo`

### Эндпоинт рекомендации цен

**POST** `/api/v1/orders/price-recommendation`

**Headers:**
```
Authorization: Bearer <access_token>
Content-Type: application/json
```

**Запрос:**
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
- `carname`, `carmodel` - для определения класса такси
- `driver_reg_date` - для расчета опыта водителя
- `user_id`, `driver_id` - для персонализации (используется кэш истории)

**Ответ:**
```json
{
  "zones": [
    {
      "zone_id": 1,
      "zone_name": "zone_1_red_low",
      "price_range": {"min": 54.0, "max": 225.14},
      "metrics": {
        "avg_probability_percent": 41.15,
        "avg_normalized_probability_percent": 73.72,
        "avg_expected_value": 55.96
      }
    }
  ],
  "optimal_price": {
    "price": 378.36,
    "probability_percent": 45.3,
    "normalized_probability_percent": 81.16,
    "expected_value": 171.4,
    "zone_id": 3,
    "net_profit": 126.8
  },
  "zone_thresholds": {
    "green_zone": "≥70% вероятность принятия",
    "yellow_low_zone": "50-70% вероятность принятия",
    "yellow_high_zone": "30-50% вероятность принятия",
    "red_zone": "<30% вероятность принятия"
  },
  "fuel_economics": {
    "fuel_cost": 16.83,
    "fuel_liters": 0.31,
    "distance_km": 3.4,
    "fuel_price_per_liter": 55.0,
    "consumption_per_100km": 9.0,
    "min_profitable_price": 21.88,
    "net_profit_from_optimal": 154.52
  },
  "analysis": {
    "start_price": 180.0,
    "max_probability_percent": 55.82,
    "max_probability_price": 54.0,
    "scan_range": {"min": 180.0, "max": 450.0},
    "timestamp": "2025-10-18 16:19:26"
  }
}
```

### Прочие эндпоинты

- **GET** `/health` - проверка здоровья сервиса
- **GET** `/` - веб-интерфейс для водителей
- **GET** `/docs` - Swagger UI
- **GET** `/redoc` - ReDoc документация

---

## 🖥️ Веб-интерфейс

**PricePilot** включает интерактивный веб-интерфейс для водителей:

- 📍 Визуальная карта с пунктами A и B
- 🎨 Цветовые зоны ценообразования
- ⚡ График вероятности принятия заказа
- ⛽ Расчет расхода топлива и прибыли
- 🎯 Рекомендованная цена с обоснованием

**Доступ:** `http://127.0.0.1:8000/`

### Настройка через переменные окружения

```bash
export WEBUI_BACKEND_BASE=""              # базовый URL API
export WEBUI_TOKEN_PATH="/auth/token"    # путь для токена
export WEBUI_PRICING_PATH="/api/v1/orders/price-recommendation"
export WEBUI_USERNAME="demo@example.com" # демо-учетка
export WEBUI_PASSWORD="demo"
```

---

## 🎓 Обучение модели

### Подготовка данных

Требуется CSV файл с историческими данными заказов:

**Обязательные поля:**
- `order_id`, `user_id`, `driver_id`
- `order_timestamp`, `tender_timestamp`
- `distance_in_meters`, `duration_in_seconds`
- `pickup_in_meters`, `pickup_in_seconds`
- `price_start_local`, `price_bid_local`
- `driver_rating`, `driver_reg_date`
- `carname`, `carmodel`, `platform`
- `is_done` (целевая переменная: `"done"` или `"cancel"`)

### Запуск обучения

```bash
python src/train_model.py
```

**Процесс обучения:**

1. ✅ Загрузка данных из `simple-train.csv`
2. 🧹 Очистка и валидация (удаление аномалий)
3. 📊 Расчет истории пользователей и водителей
4. 🔧 Создание 99 признаков
5. 📈 Разделение на train/test (80/20)
6. 🤖 Обучение XGBoost (200 деревьев)
7. 🎲 Калибровка вероятностей
8. 💾 Сохранение модели (`model_enhanced.joblib`)

**Результат:**
- `model_enhanced.joblib` - обученная модель
- `feature_names.joblib` - список признаков
- Метрики: ROC-AUC, PR-AUC, важность признаков

### Построение кэша истории

Для персонализации предсказаний:

```bash
python src/build_history_cache.py
```

**Создаются файлы:**
- `user_history.joblib` - история пользователей
- `driver_history.joblib` - история водителей

---

## 🐳 Docker развертывание

### Быстрый старт с Docker Compose

```bash
# Сборка и запуск
docker-compose up --build

# В фоновом режиме
docker-compose up -d
```

API будет доступен на `http://localhost:8000`

### Ручная сборка Docker

```bash
# Сборка образа
docker build -t pricepilot .

# Запуск контейнера
docker run -p 8000:8000 \
  -v $(pwd)/simple-train.csv:/app/simple-train.csv:ro \
  -v $(pwd)/model_enhanced.joblib:/app/model_enhanced.joblib:ro \
  pricepilot
```

### Конфигурация через переменные окружения

```yaml
environment:
  - SECRET_KEY=your-secret-key-here
  - ACCESS_TOKEN_EXPIRE_MINUTES=60
  - TEST_USER_EMAIL=demo@example.com
  - TEST_USER_PASSWORD=demo
  - PRICING_ML_MODULE=src.recommend_price
  - PRICING_ML_CALLABLE=recommend_price
  - REBUILD_CACHE=0  # 1 для принудительного пересоздания кэша
```

---

## ⚡ Производительность

### Время предсказания

⏱️ **~4 секунды** на один запрос

**Breakdown:**
- Загрузка модели (при первом запросе): ~500 мс
- Построение признаков: ~100 мс
- Batch prediction (500 точек): ~3000 мс
- Анализ и формирование ответа: ~400 мс

**Оптимизации:**
- ✅ Batch prediction вместо loop
- ✅ Кэширование модели с `@lru_cache`
- ✅ Предрасчет истории пользователей/водителей
- ✅ Векторизация операций с NumPy

### Требования к ресурсам

- **RAM**: 512 MB (min) / 1 GB (рекомендуется)
- **CPU**: 0.25 cores (min) / 0.5 cores (рекомендуется)
- **Disk**: 100 MB для модели + кэшей

### Масштабирование

Для высокой нагрузки рекомендуется:
- Запуск нескольких worker'ов Uvicorn
- Использование Nginx для балансировки
- Кэширование результатов в Redis
- Асинхронная обработка с Celery

---

## 🧪 Тестирование

### Запуск тестов

```bash
# Тест ML-модели
python test_price_recommendation.py

# Тест API через mock frontend
python scripts/mock_frontend.py
```

### Пример теста

```python
from src.recommend_price import recommend_price

order = {
    "order_timestamp": 1588341000,
    "distance_in_meters": 3404,
    "duration_in_seconds": 486,
    "pickup_in_meters": 790,
    "pickup_in_seconds": 169,
    "driver_rating": 5.0,
    "platform": "android",
    "price_start_local": 180.0
}

result = recommend_price(order, output_json=False)
print(f"Оптимальная цена: {result['optimal_price']['price']}₽")
```

---

## 🔧 Настройка ML-модуля

### Переменные окружения

```bash
export PRICING_ML_MODULE=src.recommend_price      # модуль с ML
export PRICING_ML_CALLABLE=recommend_price        # функция предсказания
export PRICING_MODEL_PATH=model_enhanced.joblib   # путь к модели
export PRICING_ML_ALLOW_STUB_FALLBACK=false       # fallback на заглушку
export PRICING_SCAN_POINTS=500                    # точек для сканирования
```

---

## 📚 Структура проекта

```
Hackaton-sinai-PricePilot/
├── app/                      # FastAPI приложение
│   ├── main.py              # главный файл API
│   ├── services.py          # бизнес-логика
│   ├── schemas.py           # Pydantic схемы
│   ├── auth.py              # JWT аутентификация
│   └── config.py            # конфигурация
├── src/                      # ML модуль
│   ├── train_model.py       # обучение модели
│   ├── recommend_price.py   # предсказание цен
│   └── build_history_cache.py  # построение кэша
├── webui/                    # веб-интерфейс
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/style.css
│       └── js/app.js
├── scripts/
│   └── mock_frontend.py     # тестовый клиент
├── Dockerfile               # Docker образ
├── docker-compose.yml       # Docker Compose конфиг
├── requirements.txt         # Python зависимости
├── model_enhanced.joblib    # обученная модель
├── user_history.joblib      # кэш истории пользователей
├── driver_history.joblib    # кэш истории водителей
└── README.md                # документация
```

---

## 🤝 Вклад и поддержка

Разработано для **Hackathon Sinai**.

### Авторы
- ML Pipeline & Feature Engineering
- FastAPI Backend
- Web UI & Visualization

### Технологии
- **Backend**: FastAPI, Uvicorn, Pydantic
- **ML**: XGBoost, scikit-learn, pandas, numpy
- **Frontend**: Vanilla JS, CSS3
- **Deploy**: Docker, Docker Compose
- **Auth**: JWT (PyJWT)

---

## 📄 Лицензия

Этот проект разработан для образовательных целей в рамках хакатона.

---

## 🎯 Roadmap

- [ ] Поддержка GPU для ускорения предсказаний
- [ ] A/B тестирование стратегий ценообразования
- [ ] Интеграция с реал-тайм потоками данных
- [ ] Reinforcement Learning для динамической оптимизации
- [ ] Мобильное приложение для водителей
- [ ] Дашборд аналитики для операторов

---

**🚀 PricePilot** - Умное ценообразование для современного такси!
