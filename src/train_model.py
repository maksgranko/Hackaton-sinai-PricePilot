import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def calculate_fuel_cost(distance_in_meters, fuel_consumption_per_100km=9.0, fuel_price_per_liter=55.0):
    """
    Рассчитывает стоимость топлива для поездки.
    
    Args:
        distance_in_meters: расстояние в метрах
        fuel_consumption_per_100km: расход топлива на 100 км (по умолчанию 9 л)
        fuel_price_per_liter: цена за литр топлива в рублях (по умолчанию 55 ₽)
    
    Returns:
        dict с информацией о расходе топлива
    """
    distance_km = distance_in_meters / 1000.0
    fuel_liters = (distance_km * fuel_consumption_per_100km) / 100.0
    fuel_cost = fuel_liters * fuel_price_per_liter
    
    return {
        'fuel_liters': fuel_liters,
        'fuel_cost_rub': fuel_cost,
        'distance_km': distance_km
    }

def calculate_user_history_features(df):
    """
    Рассчитывает признаки истории пользователя (user_id).
    
    Args:
        df: DataFrame с историческими данными
    
    Returns:
        DataFrame с признаками истории для каждого user_id
    """
    user_stats = df.groupby('user_id').agg({
        'is_done': ['count', lambda x: (x == 'done').sum()],  # Всего заказов и принятых
        'price_bid_local': 'mean',  # Средняя цена ставки
        'price_start_local': 'mean',  # Средняя стартовая цена
    }).reset_index()
    
    user_stats.columns = ['user_id', 'user_order_count', 'user_done_count', 
                          'user_avg_bid', 'user_avg_start_price']
    
    # Процент принятых заказов
    user_stats['user_acceptance_rate'] = user_stats['user_done_count'] / user_stats['user_order_count']
    
    # Средний коэффициент наценки
    user_stats['user_avg_price_ratio'] = user_stats['user_avg_bid'] / (user_stats['user_avg_start_price'] + 0.1)
    
    # Категориальные признаки
    user_stats['user_is_new'] = (user_stats['user_order_count'] <= 5).astype(float)
    user_stats['user_is_vip'] = (user_stats['user_order_count'] >= 20).astype(float)
    user_stats['user_is_price_sensitive'] = (user_stats['user_avg_price_ratio'] < 1.1).astype(float)
    
    return user_stats

def calculate_driver_history_features(df):
    """
    Рассчитывает признаки истории водителя (driver_id).
    
    Args:
        df: DataFrame с историческими данными
    
    Returns:
        DataFrame с признаками истории для каждого driver_id
    """
    driver_stats = df.groupby('driver_id').agg({
        'is_done': ['count', lambda x: (x == 'done').sum()],  # Всего ставок и принятых
        'price_bid_local': 'mean',  # Средняя ставка
        'price_start_local': 'mean',  # Средняя стартовая цена
    }).reset_index()
    
    driver_stats.columns = ['driver_id', 'driver_bid_count', 'driver_done_count',
                            'driver_avg_bid', 'driver_avg_start_price']
    
    # Процент принятых ставок
    driver_stats['driver_acceptance_rate'] = driver_stats['driver_done_count'] / driver_stats['driver_bid_count']
    
    # Средний коэффициент наценки водителя
    driver_stats['driver_avg_bid_ratio'] = driver_stats['driver_avg_bid'] / (driver_stats['driver_avg_start_price'] + 0.1)
    
    # Категориальные признаки
    driver_stats['driver_is_active'] = (driver_stats['driver_bid_count'] >= 20).astype(float)
    driver_stats['driver_is_aggressive'] = (driver_stats['driver_avg_bid_ratio'] > 1.2).astype(float)
    driver_stats['driver_is_flexible'] = (driver_stats['driver_avg_bid_ratio'] < 1.1).astype(float)
    
    return driver_stats

def clean_and_validate_data(df, verbose=True, keep_only_done=False):
    initial_count = len(df)
    
    if verbose:
        print("\n" + "="*70)
        print("ОЧИСТКА И ВАЛИДАЦИЯ ДАННЫХ")
        print("="*70)
        print(f"Исходных записей: {initial_count}")
        if keep_only_done:
            print("Режим: Только принятые биды (done)")
        else:
            print("Режим: Все биды (done + cancel) для обучения на конкуренции")
        print()
    
    df['order_timestamp'] = pd.to_datetime(df['order_timestamp'], errors='coerce')
    df['tender_timestamp'] = pd.to_datetime(df['tender_timestamp'], errors='coerce')
    df['driver_reg_date'] = pd.to_datetime(df['driver_reg_date'], errors='coerce')
    
    df['response_time_seconds'] = (df['tender_timestamp'] - df['order_timestamp']).dt.total_seconds()
    df['avg_speed_kmh'] = (df['distance_in_meters'] / df['duration_in_seconds'] * 3.6)
    df['avg_speed_kmh'] = df['avg_speed_kmh'].replace([np.inf, -np.inf], np.nan)
    df['pickup_speed_kmh'] = (df['pickup_in_meters'] / df['pickup_in_seconds'] * 3.6)
    df['pickup_speed_kmh'] = df['pickup_speed_kmh'].replace([np.inf, -np.inf], np.nan)
    df['pickup_ratio'] = df['pickup_in_meters'] / df['distance_in_meters']
    df['pickup_ratio'] = df['pickup_ratio'].replace([np.inf, -np.inf], np.nan)
    df['price_increase_pct'] = ((df['price_bid_local'] - df['price_start_local']) / df['price_start_local'] * 100)
    
    problems = {}
    
    problems['future_driver'] = (df['driver_reg_date'] > df['order_timestamp'])
    problems['future_bid'] = (df['tender_timestamp'] < df['order_timestamp'])
    problems['slow_response'] = (df['response_time_seconds'] > 300)
    
    problems['zero_distance'] = (df['distance_in_meters'] <= 0)
    problems['zero_duration'] = (df['duration_in_seconds'] <= 0)
    problems['zero_price'] = (df['price_bid_local'] <= 0)
    
    problems['too_short_trip'] = (df['distance_in_meters'] < 500)
    problems['too_quick_trip'] = (df['duration_in_seconds'] < 60)
    
    problems['extreme_distance'] = (df['distance_in_meters'] > 100000)
    problems['extreme_duration'] = (df['duration_in_seconds'] > 7200)
    
    problems['too_fast_city'] = (df['avg_speed_kmh'].notna()) & (df['avg_speed_kmh'] > 80)
    min_possible_duration = df['distance_in_meters'] / (120 / 3.6)
    problems['physically_impossible'] = (df['duration_in_seconds'] < min_possible_duration)
    problems['too_slow'] = (df['avg_speed_kmh'].notna()) & (df['avg_speed_kmh'] < 8)
    
    problems['extreme_pickup_ratio'] = (df['pickup_ratio'].notna()) & (df['pickup_ratio'] > 5)
    problems['extreme_pickup_speed'] = (df['pickup_speed_kmh'].notna()) & (df['pickup_speed_kmh'] > 100)
    
    problems['extreme_markup'] = (df['price_increase_pct'] > 100)
    problems['extreme_price'] = (df['price_bid_local'] > 5000)
    
    duplicate_mask = df.duplicated(subset=[
        'order_id', 'driver_id', 'price_bid_local', 
        'pickup_in_meters', 'tender_timestamp'
    ], keep='first')
    problems['exact_duplicate'] = duplicate_mask
    
    if keep_only_done:
        problems['not_accepted'] = (df['is_done'] != 'done')
    
    if verbose:
        descriptions = {
            'future_driver': 'Водитель зарегистрирован после заказа',
            'future_bid': 'Бид отправлен до создания заказа',
            'slow_response': 'Медленный ответ водителя (>5 мин)',
            'zero_distance': 'Нулевое/отрицательное расстояние',
            'zero_duration': 'Нулевая/отрицательная длительность',
            'zero_price': 'Нулевая/отрицательная цена',
            'too_short_trip': 'Слишком короткая поездка (<500м)',
            'too_quick_trip': 'Слишком быстрая поездка (<60сек)',
            'extreme_distance': 'Слишком длинная поездка (>100 км)',
            'extreme_duration': 'Слишком долгая поездка (>2 ч)',
            'too_fast_city': 'Слишком высокая скорость (>80 км/ч)',
            'physically_impossible': 'Физически невозможная скорость (>120 км/ч)',
            'too_slow': 'Слишком низкая скорость (<8 км/ч)',
            'extreme_pickup_ratio': 'Подача >5x длиннее поездки',
            'extreme_pickup_speed': 'Скорость подачи >100 км/ч',
            'extreme_markup': 'Наценка >100%',
            'extreme_price': 'Цена >5000₽',
            'exact_duplicate': 'Точный дубликат (все поля совпадают)',
            'not_accepted': 'Отклонённый бид (is_done=cancel)'
        }
        
        print(f"{'Проблема':<50s} {'Записей':>10s}")
        print("-"*62)
        
        for name, mask in problems.items():
            count = mask.sum()
            if count > 0:
                desc = descriptions.get(name, name)
                print(f"{desc:<50s} {count:>10d}")
    
    delete_mask = pd.Series(False, index=df.index)
    for mask in problems.values():
        delete_mask |= mask
    
    df_clean = df[~delete_mask].copy()
    
    if verbose:
        print("-"*62)
        print(f"{'ИТОГО удалено:':<50s} {delete_mask.sum():>10d} ({delete_mask.sum()/initial_count*100:.1f}%)")
        print(f"{'Осталось записей:':<50s} {len(df_clean):>10d} ({len(df_clean)/initial_count*100:.1f}%)")
        
        unique_orders = df_clean['order_id'].nunique()
        avg_bids_per_order = len(df_clean) / unique_orders if unique_orders > 0 else 0
        print(f"\n{'Уникальных заказов:':<50s} {unique_orders:>10d}")
        print(f"{'Среднее бидов на заказ:':<50s} {avg_bids_per_order:>10.2f}")
        
        df_clean['is_done_binary'] = (df_clean['is_done'] == 'done').astype(int)
        done_count = df_clean['is_done_binary'].sum()
        cancel_count = len(df_clean) - done_count
        done_pct = done_count / len(df_clean) * 100 if len(df_clean) > 0 else 0
        
        print(f"\n{'Баланс классов:':<50s}")
        print(f"{'  Done':<50s} {done_count:>10d} ({done_pct:5.1f}%)")
        print(f"{'  Cancel':<50s} {cancel_count:>10d} ({100-done_pct:5.1f}%)")
        
        if done_count > 0:
            ratio = cancel_count / done_count
            print(f"\n{'scale_pos_weight для XGBoost:':<50s} {ratio:>10.2f}")
        print("="*70)
    
    return df_clean

def detect_taxi_type(carname, carmodel):
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

def build_enhanced_features(frame):
    """
    Создает признаки для ML-модели.
    
    Args:
        frame: DataFrame с данными заказов
    
    Returns:
        DataFrame с признаками
    """
    # 📊 НОВЫЕ ПРИЗНАКИ: История пользователей и водителей
    user_history = calculate_user_history_features(frame)
    driver_history = calculate_driver_history_features(frame)
    
    # Объединяем с основными данными
    frame = frame.merge(user_history, on='user_id', how='left')
    frame = frame.merge(driver_history, on='driver_id', how='left')
    
    # Заполняем пропуски для новых пользователей/водителей
    frame['user_order_count'] = frame['user_order_count'].fillna(1)
    frame['user_done_count'] = frame['user_done_count'].fillna(0)
    frame['user_acceptance_rate'] = frame['user_acceptance_rate'].fillna(0.5)
    frame['user_avg_bid'] = frame['user_avg_bid'].fillna(frame['price_bid_local'])
    frame['user_avg_start_price'] = frame['user_avg_start_price'].fillna(frame['price_start_local'])
    frame['user_avg_price_ratio'] = frame['user_avg_price_ratio'].fillna(1.0)
    frame['user_is_new'] = frame['user_is_new'].fillna(1.0)
    frame['user_is_vip'] = frame['user_is_vip'].fillna(0.0)
    frame['user_is_price_sensitive'] = frame['user_is_price_sensitive'].fillna(0.5)
    
    frame['driver_bid_count'] = frame['driver_bid_count'].fillna(1)
    frame['driver_done_count'] = frame['driver_done_count'].fillna(0)
    frame['driver_acceptance_rate'] = frame['driver_acceptance_rate'].fillna(0.5)
    frame['driver_avg_bid'] = frame['driver_avg_bid'].fillna(frame['price_bid_local'])
    frame['driver_avg_start_price'] = frame['driver_avg_start_price'].fillna(frame['price_start_local'])
    frame['driver_avg_bid_ratio'] = frame['driver_avg_bid_ratio'].fillna(1.0)
    frame['driver_is_active'] = frame['driver_is_active'].fillna(0.5)
    frame['driver_is_aggressive'] = frame['driver_is_aggressive'].fillna(0.0)
    frame['driver_is_flexible'] = frame['driver_is_flexible'].fillna(0.5)
    
    ts = pd.to_datetime(frame["order_timestamp"], errors="coerce")
    hour = ts.dt.hour.fillna(0)
    wday = ts.dt.weekday.fillna(0)
    
    features = {}
    
    features['price_bid_local'] = frame['price_bid_local'].values
    features['price_start_local'] = frame['price_start_local'].values
    features['price_increase_abs'] = (frame['price_bid_local'] - frame['price_start_local']).values
    features['price_increase_pct'] = ((frame['price_bid_local'] - frame['price_start_local']) / 
                                      frame['price_start_local'] * 100).values
    features['is_price_increased'] = (features['price_increase_pct'] > 0).astype(float)
    
    features['price_per_km'] = frame['price_bid_local'] / (frame['distance_in_meters'] / 1000 + 0.1)
    features['price_per_minute'] = frame['price_bid_local'] / (frame['duration_in_seconds'] / 60 + 0.1)
    
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    features['day_of_week'] = wday.values
    features['day_sin'] = np.sin(2 * np.pi * wday / 7)
    features['day_cos'] = np.cos(2 * np.pi * wday / 7)
    features['is_weekend'] = (wday >= 5).astype(float).values
    
    is_morning_rush = ((hour >= 7) & (hour <= 9)).astype(int)
    is_evening_rush = ((hour >= 17) & (hour <= 20)).astype(int)
    features['is_morning_peak'] = is_morning_rush.values
    features['is_evening_peak'] = is_evening_rush.values
    features['is_peak_hour'] = ((is_morning_rush + is_evening_rush) > 0).astype(float).values
    features['is_night'] = ((hour < 6) | (hour >= 22)).astype(float).values
    features['is_lunch_time'] = ((hour >= 12) & (hour <= 14)).astype(float).values
    
    features['distance_in_meters'] = frame['distance_in_meters'].values
    features['duration_in_seconds'] = frame['duration_in_seconds'].values
    features['distance_km'] = (frame['distance_in_meters'] / 1000).values
    features['duration_min'] = (frame['duration_in_seconds'] / 60).values
    
    speed = frame['distance_in_meters'] / (frame['duration_in_seconds'] + 0.1) * 3.6
    features['avg_speed_kmh'] = np.clip(speed.values, 0, 150)
    features['is_traffic_jam'] = (features['avg_speed_kmh'] < 15).astype(float)
    features['is_highway'] = (features['avg_speed_kmh'] > 50).astype(float)
    
    dist = features['distance_km']
    features['is_short_trip'] = (dist < 2).astype(float)
    features['is_medium_trip'] = ((dist >= 2) & (dist < 10)).astype(float)
    features['is_long_trip'] = (dist >= 10).astype(float)
    
    features['pickup_in_meters'] = frame['pickup_in_meters'].values
    features['pickup_in_seconds'] = frame['pickup_in_seconds'].values
    features['pickup_km'] = (frame['pickup_in_meters'] / 1000).values
    
    pickup_speed = frame['pickup_in_meters'] / (frame['pickup_in_seconds'] + 0.1) * 3.6
    features['pickup_speed_kmh'] = np.clip(pickup_speed.values, 0, 150)
    
    features['pickup_to_trip_ratio'] = np.clip(
        frame['pickup_in_meters'] / (frame['distance_in_meters'] + 1).values,
        0, 10
    )
    features['pickup_time_ratio'] = np.clip(
        frame['pickup_in_seconds'] / (frame['duration_in_seconds'] + 1).values,
        0, 10
    )
    features['total_distance'] = (frame['pickup_in_meters'] + frame['distance_in_meters']).values
    features['total_time'] = (frame['pickup_in_seconds'] + frame['duration_in_seconds']).values
    
    features['driver_rating'] = frame['driver_rating'].values
    
    driver_reg = pd.to_datetime(frame['driver_reg_date'], errors='coerce')
    experience_days = (ts - driver_reg).dt.days.fillna(365)
    experience_days = np.clip(experience_days.values, 0, 3650)
    features['driver_experience_days'] = experience_days
    features['driver_experience_years'] = experience_days / 365.25
    features['is_new_driver'] = (experience_days < 30).astype(float)
    features['is_experienced_driver'] = (experience_days > 365).astype(float)
    features['has_perfect_rating'] = (frame['driver_rating'] == 5.0).astype(float).values
    features['rating_deviation'] = (5.0 - frame['driver_rating']).values
    
    tender_ts = pd.to_datetime(frame['tender_timestamp'], errors='coerce')
    response_time = (tender_ts - ts).dt.total_seconds().fillna(30)
    response_time = np.clip(response_time.values, 0, 600)
    features['response_time_seconds'] = response_time
    features['response_time_log'] = np.log1p(response_time)
    features['is_fast_response'] = (response_time < 10).astype(float)
    features['is_slow_response'] = (response_time > 60).astype(float)
    
    taxi_types = frame.apply(lambda row: detect_taxi_type(row['carname'], row['carmodel']), axis=1)
    features['taxi_type_economy'] = (taxi_types == 'economy').astype(float).values
    features['taxi_type_comfort'] = (taxi_types == 'comfort').astype(float).values
    features['taxi_type_business'] = (taxi_types == 'business').astype(float).values
    
    features['platform_android'] = (frame['platform'] == 'android').astype(float).values
    features['platform_ios'] = (frame['platform'] == 'ios').astype(float).values
    
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
    
    # ⛽ НОВЫЕ ПРИЗНАКИ: Экономика топлива
    # Расчет стоимости топлива для каждой поездки (векторизованно)
    distance_km = frame['distance_in_meters'].values / 1000.0
    fuel_liters = (distance_km * 9.0) / 100.0  # 9 л на 100 км
    fuel_cost = fuel_liters * 55.0  # 55 ₽ за литр
    
    features['fuel_cost_rub'] = fuel_cost
    features['fuel_liters'] = fuel_liters
    
    # Отношение цены к стоимости топлива - ключевой показатель рентабельности
    features['price_to_fuel_ratio'] = frame['price_bid_local'].values / (fuel_cost + 0.1)
    
    # Минимальная рентабельная цена (топливо + 30%)
    min_profitable = fuel_cost * 1.3
    features['min_profitable_price'] = min_profitable
    
    # Насколько текущая цена выше/ниже минимальной рентабельной
    features['price_above_min_profitable'] = frame['price_bid_local'].values - min_profitable
    features['price_above_min_profitable_pct'] = ((frame['price_bid_local'].values - min_profitable) / 
                                                   (min_profitable + 0.1) * 100)
    
    # Флаги для категорий рентабельности
    features['is_highly_profitable'] = (frame['price_bid_local'].values >= min_profitable * 2).astype(float)
    features['is_profitable'] = (frame['price_bid_local'].values >= min_profitable).astype(float)
    features['is_unprofitable'] = (frame['price_bid_local'].values < min_profitable).astype(float)
    
    # Чистая прибыль от ставки (цена - топливо)
    net_profit = frame['price_bid_local'].values - fuel_cost
    features['net_profit'] = net_profit
    features['net_profit_per_km'] = net_profit / (distance_km + 0.1)
    features['net_profit_per_minute'] = net_profit / (features['duration_min'] + 0.1)
    
    # Взаимодействия топлива с другими признаками
    features['fuel_ratio_x_distance'] = features['price_to_fuel_ratio'] * features['distance_km']
    features['fuel_ratio_x_peak'] = features['price_to_fuel_ratio'] * features['is_peak_hour']
    features['net_profit_x_rating'] = net_profit * features['driver_rating']
    
    # 👤 НОВЫЕ ПРИЗНАКИ: История пользователя
    features['user_order_count'] = frame['user_order_count'].values
    features['user_acceptance_rate'] = frame['user_acceptance_rate'].values
    features['user_avg_price_ratio'] = frame['user_avg_price_ratio'].values
    features['user_is_new'] = frame['user_is_new'].values
    features['user_is_vip'] = frame['user_is_vip'].values
    features['user_is_price_sensitive'] = frame['user_is_price_sensitive'].values
    
    # 🚗 НОВЫЕ ПРИЗНАКИ: История водителя
    features['driver_bid_count'] = frame['driver_bid_count'].values
    features['driver_acceptance_rate'] = frame['driver_acceptance_rate'].values
    features['driver_avg_bid_ratio'] = frame['driver_avg_bid_ratio'].values
    features['driver_is_active'] = frame['driver_is_active'].values
    features['driver_is_aggressive'] = frame['driver_is_aggressive'].values
    features['driver_is_flexible'] = frame['driver_is_flexible'].values
    
    # 🔗 НОВЫЕ ПРИЗНАКИ: Взаимодействия истории
    features['user_driver_match_score'] = features['user_acceptance_rate'] * features['driver_acceptance_rate']
    features['price_vs_user_avg'] = frame['price_bid_local'].values / (frame['user_avg_bid'].values + 0.1)
    features['price_vs_driver_avg'] = frame['price_bid_local'].values / (frame['driver_avg_bid'].values + 0.1)
    
    # 🗺️ НОВЫЕ ПРИЗНАКИ: Улучшенные признаки маршрута
    features['route_efficiency'] = features['distance_km'] / (features['duration_min'] + 0.1)  # км/мин
    features['is_very_short'] = (features['distance_km'] < 1).astype(float)
    features['is_very_long'] = (features['distance_km'] > 20).astype(float)
    features['pickup_burden'] = features['pickup_km'] / (features['distance_km'] + 0.1)  # Насколько подача нагружает
    
    # ⏰ НОВЫЕ ПРИЗНАКИ: Временные паттерны
    day_of_month = ts.dt.day.fillna(15)
    features['day_of_month'] = day_of_month.values
    features['is_month_start'] = (day_of_month <= 5).astype(float).values  # Начало месяца (зарплата)
    features['is_month_end'] = (day_of_month >= 25).astype(float).values  # Конец месяца (деньги кончаются)
    features['hour_quartile'] = (hour // 6).astype(float).values  # 0: 0-6, 1: 6-12, 2: 12-18, 3: 18-24
    
    result = pd.DataFrame(features)
    
    result = result.replace([np.inf, -np.inf], np.nan)
    
    for col in result.columns:
        if result[col].isna().any():
            median_val = result[col].median()
            if pd.isna(median_val) or np.isinf(median_val):
                result[col] = result[col].fillna(0)
            else:
                result[col] = result[col].fillna(median_val)
    
    result = result.fillna(0)
    result = result.replace([np.inf], 1e10)
    result = result.replace([-np.inf], -1e10)
    
    for col in result.columns:
        if result[col].std() > 0:
            mean = result[col].mean()
            std = result[col].std()
            result[col] = np.clip(result[col], mean - 10*std, mean + 10*std)
    
    return result

def train_model(train_path="simple-train.csv", use_gpu=False, test_size=0.2, random_state=42):
    print("\n" + "="*70)
    print("ОБУЧЕНИЕ ML-МОДЕЛИ DRIVEE")
    print("="*70)
    
    print(f"\n📁 Загрузка данных из {train_path}...")
    df = pd.read_csv(train_path)
    print(f"   Загружено: {len(df)} записей")
    
    df = clean_and_validate_data(
        df, 
        verbose=True,
        keep_only_done=False
    )
    
    if len(df) < 100:
        raise ValueError("⚠️ Слишком мало данных после очистки! Проверьте исходный датасет.")
    
    print("\n🎯 Подготовка целевой переменной...")
    y = (df['is_done'] == 'done').astype(int)
    print(f"   Done: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   Cancel: {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")
    
    print("\n🔧 Создание признаков...")
    X = build_enhanced_features(df)
    print(f"   Создано признаков: {X.shape[1]}")
    print(f"   Размер данных: {X.shape[0]} записей × {X.shape[1]} признаков")
    
    print(f"\n📊 Разделение данных (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"   Train: {len(X_train)} записей")
    print(f"   Test:  {len(X_test)} записей")
    
    print("\n🤖 Обучение XGBoost...")
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    params = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 10,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'gamma': 0.2,
        'reg_alpha': 0.3,
        'reg_lambda': 2.0,
        'scale_pos_weight': scale_pos_weight,
        'tree_method': 'hist',
        'random_state': random_state,
        'eval_metric': 'logloss'
    }
    
    print(f"   Параметры:")
    print(f"     • n_estimators: {params['n_estimators']}")
    print(f"     • learning_rate: {params['learning_rate']}")
    print(f"     • max_depth: {params['max_depth']}")
    print(f"     • scale_pos_weight: {scale_pos_weight:.2f}")
    print(f"     • tree_method: {params['tree_method']}")
    
    model = xgb.XGBClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    print("\n🎲 Калибровка вероятностей...")
    calibrated_model = CalibratedClassifierCV(
        model, 
        method='sigmoid',
        cv='prefit'
    )
    calibrated_model.fit(X_train, y_train)
    
    print("\n📈 Оценка качества модели:")
    
    y_train_pred = calibrated_model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_pred)
    
    y_test_pred = calibrated_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred)
    
    precision, recall, _ = precision_recall_curve(y_test, y_test_pred)
    pr_auc = auc(recall, precision)
    
    print(f"   ROC-AUC (train): {train_auc:.4f}")
    print(f"   ROC-AUC (test):  {test_auc:.4f}")
    print(f"   PR-AUC (test):   {pr_auc:.4f}")
    
    print("\n🔝 Топ-10 важных признаков:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:30s} {row['importance']:.4f}")
    
    print("\n💾 Сохранение модели...")
    joblib.dump(calibrated_model, "model_enhanced.joblib")
    print("   ✓ Модель сохранена: model_enhanced.joblib")
    
    joblib.dump(X.columns.tolist(), "feature_names.joblib")
    print("   ✓ Признаки сохранены: feature_names.joblib")
    
    print("\n" + "="*70)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print("="*70)
    
    return calibrated_model, feature_importance

if __name__ == "__main__":
    try:
        model, importance = train_model(
            train_path="simple-train.csv",
            use_gpu=False,
            test_size=0.2,
            random_state=42
        )
        print("\n🎉 Модель готова к использованию!")
        
    except Exception as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
