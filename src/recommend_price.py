"""
Модуль рекомендации оптимальной цены бида для водителей Drivee
Совместим с обновлённой train_model.py (использует те же признаки)
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime


def detect_taxi_type(carname, carmodel):
    """Определяет тип такси по марке и модели"""
    carname = str(carname).strip()
    carmodel = str(carmodel).strip()
    
    economy_brands = ['Daewoo', 'Lifan', 'FAW', 'Great Wall', 'Geely', 'ЗАЗ', 'Chery']
    economy_models = [
        'Logan', 'Symbol', 'Sandero', 'Lacetti', 'Aveo', 'Nexia', 'Rio', 'Spectra',
        'Granta', 'Гранта', 'Kalina', 'Калина', 'Priora', 'Приора', 
        '2110', '2112', '2115', '2107', '2114', 'Самара', 'S18'
    ]
    
    business_brands = ['Toyota', 'Honda', 'Mitsubishi', 'Subaru']
    business_models = [
        'Camry', 'Corolla', 'RAV4', 'Avensis', 'Civic', 'Accord', 
        'Qashqai', 'X-Trail', 'Tiguan', 'Passat CC', 'Passat',
        'CX-5', 'Outlander', 'Kyron', 'Legacy'
    ]
    
    lada_comfort_models = ['Vesta', 'Веста', 'X-Ray', 'Largus', 'Ларгус', 'GFK110']
    
    if carname in economy_brands or carmodel in economy_models:
        return "economy"
    
    if carname in business_brands or carmodel in business_models:
        return "business"
    
    if carname in ['LADA', 'Лада', 'ВАЗ (LADA)'] and carmodel in lada_comfort_models:
        return "comfort"
    
    return "comfort"


def build_features_for_price(order_data, price_bid):
    """
    Строит признаки для конкретной цены бида
    ВАЖНО: Должен использовать ТЕ ЖЕ признаки что и в train_model.py!
    
    Args:
        order_data: dict с информацией о заказе
        price_bid: float - тестируемая цена бида
        
    Returns:
        pd.DataFrame с одной строкой признаков
    """
    
    # Создаём временный DataFrame для единообразия
    temp_df = pd.DataFrame([{
        'order_timestamp': order_data['order_timestamp'],
        'tender_timestamp': order_data.get('tender_timestamp', order_data['order_timestamp']),
        'driver_reg_date': order_data.get('driver_reg_date', '2020-01-01'),
        'distance_in_meters': order_data['distance_in_meters'],
        'duration_in_seconds': order_data['duration_in_seconds'],
        'pickup_in_meters': order_data['pickup_in_meters'],
        'pickup_in_seconds': order_data['pickup_in_seconds'],
        'driver_rating': order_data.get('driver_rating', 5.0),
        'carname': order_data.get('carname', 'Renault'),
        'carmodel': order_data.get('carmodel', 'Logan'),
        'platform': order_data.get('platform', 'android'),
        'price_start_local': order_data['price_start_local'],
        'price_bid_local': price_bid  # КРИТИЧНО: варьируемая цена!
    }])
    
    # Конвертация дат
    ts = pd.to_datetime(temp_df["order_timestamp"], errors="coerce")
    hour = ts.dt.hour.fillna(0)
    wday = ts.dt.weekday.fillna(0)
    
    features = {}
    
    # ========================================================================
    # ЦЕНОВЫЕ ПРИЗНАКИ (КРИТИЧНО!)
    # ========================================================================
    features['price_bid_local'] = price_bid
    features['price_start_local'] = order_data['price_start_local']
    features['price_increase_abs'] = price_bid - order_data['price_start_local']
    features['price_increase_pct'] = ((price_bid - order_data['price_start_local']) / 
                                      order_data['price_start_local'] * 100)
    features['is_price_increased'] = float(features['price_increase_pct'] > 0)
    
    # Нормализованные цены
    features['price_per_km'] = price_bid / (order_data['distance_in_meters'] / 1000 + 0.1)
    features['price_per_minute'] = price_bid / (order_data['duration_in_seconds'] / 60 + 0.1)
    
    # ========================================================================
    # ВРЕМЕННЫЕ ПРИЗНАКИ
    # ========================================================================
    h = hour.iloc[0]
    w = wday.iloc[0]
    
    features['hour_sin'] = np.sin(2 * np.pi * h / 24)
    features['hour_cos'] = np.cos(2 * np.pi * h / 24)
    features['day_of_week'] = w
    features['day_sin'] = np.sin(2 * np.pi * w / 7)
    features['day_cos'] = np.cos(2 * np.pi * w / 7)
    features['is_weekend'] = float(w >= 5)
    
    # Часы пик
    is_morning = 1 if 7 <= h <= 9 else 0
    is_evening = 1 if 17 <= h <= 20 else 0
    features['is_morning_peak'] = float(is_morning)
    features['is_evening_peak'] = float(is_evening)
    features['is_peak_hour'] = float(is_morning or is_evening)
    features['is_night'] = float(h < 6 or h >= 22)
    features['is_lunch_time'] = float(12 <= h <= 14)
    
    # ========================================================================
    # ХАРАКТЕРИСТИКИ ПОЕЗДКИ
    # ========================================================================
    dist_m = order_data['distance_in_meters']
    dur_s = order_data['duration_in_seconds']
    
    features['distance_in_meters'] = dist_m
    features['duration_in_seconds'] = dur_s
    features['distance_km'] = dist_m / 1000
    features['duration_min'] = dur_s / 60
    
    # Скорость
    avg_speed = (dist_m / dur_s * 3.6) if dur_s > 0 else 25.0
    features['avg_speed_kmh'] = avg_speed
    features['is_traffic_jam'] = float(avg_speed < 15)
    features['is_highway'] = float(avg_speed > 50)
    
    # Категории дистанции
    dist_km = features['distance_km']
    features['is_short_trip'] = float(dist_km < 2)
    features['is_medium_trip'] = float(2 <= dist_km < 10)
    features['is_long_trip'] = float(dist_km >= 10)
    
    # ========================================================================
    # ПОДАЧА ВОДИТЕЛЯ
    # ========================================================================
    pickup_m = order_data['pickup_in_meters']
    pickup_s = order_data['pickup_in_seconds']
    
    features['pickup_in_meters'] = pickup_m
    features['pickup_in_seconds'] = pickup_s
    features['pickup_km'] = pickup_m / 1000
    
    pickup_speed = (pickup_m / pickup_s * 3.6) if pickup_s > 0 else 20.0
    features['pickup_speed_kmh'] = pickup_speed
    
    # Соотношения
    features['pickup_to_trip_ratio'] = pickup_m / (dist_m + 1)
    features['pickup_time_ratio'] = pickup_s / (dur_s + 1)
    features['total_distance'] = pickup_m + dist_m
    features['total_time'] = pickup_s + dur_s
    
    # ========================================================================
    # ВОДИТЕЛЬ
    # ========================================================================
    features['driver_rating'] = order_data.get('driver_rating', 5.0)
    
    # Стаж (если не указан driver_reg_date, предполагаем 1 год стажа)
    try:
        driver_reg = pd.to_datetime(order_data.get('driver_reg_date', '2020-01-01'))
        order_ts = ts.iloc[0]
        exp_days = (order_ts - driver_reg).days
        if exp_days < 0:
            exp_days = 365  # fallback
    except:
        exp_days = 365
    
    features['driver_experience_days'] = exp_days
    features['driver_experience_years'] = exp_days / 365.25
    features['is_new_driver'] = float(exp_days < 30)
    features['is_experienced_driver'] = float(exp_days > 365)
    features['has_perfect_rating'] = float(features['driver_rating'] == 5.0)
    features['rating_deviation'] = 5.0 - features['driver_rating']
    
    # Время ответа (предполагаем среднее 30 сек)
    response_time = 30.0
    features['response_time_seconds'] = response_time
    features['response_time_log'] = np.log1p(response_time)
    features['is_fast_response'] = float(response_time < 10)
    features['is_slow_response'] = float(response_time > 60)
    
    # ========================================================================
    # АВТОМОБИЛЬ
    # ========================================================================
    taxi_type = detect_taxi_type(
        order_data.get('carname', 'Renault'),
        order_data.get('carmodel', 'Logan')
    )
    
    features['taxi_type_economy'] = float(taxi_type == 'economy')
    features['taxi_type_comfort'] = float(taxi_type == 'comfort')
    features['taxi_type_business'] = float(taxi_type == 'business')
    
    # Платформа
    platform = order_data.get('platform', 'android')
    features['platform_android'] = float(platform == 'android')
    features['platform_ios'] = float(platform == 'ios')
    
    # ========================================================================
    # ВЗАИМОДЕЙСТВИЯ ПРИЗНАКОВ
    # ========================================================================
    features['price_inc_x_distance'] = features['price_increase_pct'] * features['distance_km']
    features['price_inc_x_night'] = features['price_increase_pct'] * features['is_night']
    features['price_inc_x_peak'] = features['price_increase_pct'] * features['is_peak_hour']
    features['price_inc_x_weekend'] = features['price_increase_pct'] * features['is_weekend']
    
    features['distance_x_night'] = features['distance_km'] * features['is_night']
    features['distance_x_weekend'] = features['distance_km'] * features['is_weekend']
    features['distance_x_peak'] = features['distance_km'] * features['is_peak_hour']
    
    features['speed_x_peak'] = features['avg_speed_kmh'] * features['is_peak_hour']
    features['rating_x_price_inc'] = features['driver_rating'] * features['price_increase_pct']
    features['experience_x_price_inc'] = features['driver_experience_years'] * features['price_increase_pct']
    
    # Преобразуем в DataFrame
    result = pd.DataFrame([features])
    
    return result


def find_optimal_price(order_data, model, num_points=500):
    """
    Находит оптимальную цену максимизирующую Expected Revenue
    
    Args:
        order_data: dict с информацией о заказе
        model: обученная модель
        num_points: количество точек для сканирования
        
    Returns:
        dict с результатами оптимизации
    """
    
    start_price = order_data['price_start_local']
    
    # Диапазон сканирования: от 70% до 160% начальной цены
    min_price = max(start_price * 0.7, 100)  # минимум 100₽
    max_price = start_price * 1.6
    
    # Генерируем цены для тестирования
    prices = np.linspace(min_price, max_price, num_points)
    
    probabilities = []
    expected_revenues = []
    
    # Сканируем каждую цену
    for price in prices:
        # Строим признаки для данной цены
        features = build_features_for_price(order_data, price)
        
        # Предсказываем вероятность
        prob = model.predict_proba(features)[0, 1]
        
        # Ожидаемый доход = цена × вероятность
        expected_revenue = price * prob
        
        probabilities.append(prob)
        expected_revenues.append(expected_revenue)
    
    # Находим оптимум
    probabilities = np.array(probabilities)
    expected_revenues = np.array(expected_revenues)
    
    best_idx = np.argmax(expected_revenues)
    optimal_price = prices[best_idx]
    optimal_prob = probabilities[best_idx]
    optimal_revenue = expected_revenues[best_idx]
    
    # Создаём 5 ценовых зон
    zones = []
    zone_colors = ['green', 'yellow-green', 'yellow', 'orange', 'red']
    zone_names = ['Высокая вероятность', 'Хорошая цена', 'Средняя', 'Рискованная', 'Низкая вероятность']
    
    # Разбиваем на 5 зон по вероятности
    prob_quantiles = np.percentile(probabilities, [20, 40, 60, 80])
    
    for i, (color, name) in enumerate(zip(zone_colors, zone_names)):
        if i == 0:
            mask = probabilities >= prob_quantiles[3]
        elif i == 1:
            mask = (probabilities >= prob_quantiles[2]) & (probabilities < prob_quantiles[3])
        elif i == 2:
            mask = (probabilities >= prob_quantiles[1]) & (probabilities < prob_quantiles[2])
        elif i == 3:
            mask = (probabilities >= prob_quantiles[0]) & (probabilities < prob_quantiles[1])
        else:
            mask = probabilities < prob_quantiles[0]
        
        zone_prices = prices[mask]
        zone_probs = probabilities[mask]
        
        if len(zone_prices) > 0:
            zones.append({
                'color': color,
                'name': name,
                'price_from': int(zone_prices.min()),
                'price_to': int(zone_prices.max()),
                'avg_probability': float(zone_probs.mean()),
                'avg_revenue': float((zone_prices * zone_probs).mean())
            })
    
    result = {
        'optimal_price': int(optimal_price),
        'optimal_probability': float(optimal_prob),
        'expected_revenue': float(optimal_revenue),
        'start_price': int(start_price),
        'price_increase_pct': float((optimal_price - start_price) / start_price * 100),
        'zones': zones,
        'scan_stats': {
            'prices_tested': int(num_points),
            'min_price': float(min_price),
            'max_price': float(max_price),
            'avg_probability': float(probabilities.mean()),
            'max_probability': float(probabilities.max())
        }
    }
    
    return result


def recommend_price(order_data, output_json=True, model_path="model_enhanced.joblib"):
    """
    Главная функция рекомендации цены
    """
    
    # Загрузка модели
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"⚠️ Модель не найдена: {model_path}")
    
    model = joblib.load(model_path)
    
    # ОТЛАДКА: Проверяем что загрузили
    print(f"  Тип загруженного объекта: {type(model)}")
    if hasattr(model, 'predict_proba'):
        print(f"  ✓ Модель имеет метод predict_proba")
    else:
        print(f"  ✗ ОШИБКА: Объект не имеет predict_proba!")
        print(f"  Содержимое: {model}")
        raise TypeError(f"Загружен неправильный объект: {type(model)}")
    
    # Валидация входных данных
    required_fields = [
        'order_timestamp', 'distance_in_meters', 'duration_in_seconds',
        'pickup_in_meters', 'pickup_in_seconds', 'price_start_local'
    ]
    
    for field in required_fields:
        if field not in order_data:
            raise ValueError(f"⚠️ Отсутствует обязательное поле: {field}")
    
    # Поиск оптимальной цены
    result = find_optimal_price(order_data, model, num_points=500)
    
    if output_json:
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    return result


if __name__ == "__main__":
    # Пример использования
    
    print("\n" + "="*70)
    print("ТЕСТ СИСТЕМЫ РЕКОМЕНДАЦИИ ЦЕН")
    print("="*70)
    
    # Проверяем наличие модели
    if not os.path.exists("model_enhanced.joblib"):
        print("\n⚠️ Модель не найдена! Сначала запустите train_model.py")
        exit(1)
    
    # Тестовый заказ
    test_order = {
        "order_timestamp": int(datetime.now().timestamp()),
        "distance_in_meters": 5000,
        "duration_in_seconds": 900,
        "pickup_in_meters": 1500,
        "pickup_in_seconds": 200,
        "driver_rating": 4.9,
        "platform": "android",
        "price_start_local": 250,
        "carname": "Toyota",
        "carmodel": "Camry",
        "driver_reg_date": "2020-01-15"
    }
    
    print("\nТестовый заказ:")
    print(f"  Дистанция: {test_order['distance_in_meters']/1000:.1f} км")
    print(f"  Время в пути: {test_order['duration_in_seconds']/60:.1f} мин")
    print(f"  Начальная цена: {test_order['price_start_local']}₽")
    print(f"  Автомобиль: {test_order['carname']} {test_order['carmodel']}")
    
    print("\nПоиск оптимальной цены...")
    result = recommend_price(test_order, output_json=False)
    
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ")
    print("="*70)
    print(f"\n✅ Рекомендованная цена: {result['optimal_price']}₽")
    print(f"   Вероятность принятия: {result['optimal_probability']*100:.1f}%")
    print(f"   Ожидаемый доход: {result['expected_revenue']:.2f}₽")
    print(f"   Наценка к начальной цене: {result['price_increase_pct']:+.1f}%")
    
    print("\nЦеновые зоны:")
    for zone in result['zones']:
        print(f"  {zone['name']:25s} [{zone['color']:12s}] "
              f"{zone['price_from']:4d}-{zone['price_to']:4d}₽ "
              f"(вероятность: {zone['avg_probability']*100:4.1f}%)")
    
    print("\n" + "="*70)
