"""
Модуль обучения ML-модели для оптимизации цен бидов водителей Drivee
Включает очистку данных, feature engineering и обучение XGBoost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')


def clean_and_validate_data(df, verbose=True):
    """
    Очищает датасет от физически невозможных и некачественных записей
    
    Args:
        df: Исходный DataFrame
        verbose: Выводить ли статистику
        
    Returns:
        pd.DataFrame: Очищенный датасет
    """
    
    initial_count = len(df)
    
    if verbose:
        print("\n" + "="*70)
        print("ОЧИСТКА И ВАЛИДАЦИЯ ДАННЫХ")
        print("="*70)
        print(f"Исходных записей: {initial_count}\n")
    
    # Конвертация временных меток
    df['order_timestamp'] = pd.to_datetime(df['order_timestamp'], errors='coerce')
    df['tender_timestamp'] = pd.to_datetime(df['tender_timestamp'], errors='coerce')
    df['driver_reg_date'] = pd.to_datetime(df['driver_reg_date'], errors='coerce')
    
    # Расчёт вспомогательных метрик для проверки
    df['response_time_seconds'] = (df['tender_timestamp'] - df['order_timestamp']).dt.total_seconds()
    df['avg_speed_kmh'] = (df['distance_in_meters'] / df['duration_in_seconds'] * 3.6)
    df['avg_speed_kmh'] = df['avg_speed_kmh'].replace([np.inf, -np.inf], np.nan)
    df['pickup_speed_kmh'] = (df['pickup_in_meters'] / df['pickup_in_seconds'] * 3.6)
    df['pickup_speed_kmh'] = df['pickup_speed_kmh'].replace([np.inf, -np.inf], np.nan)
    df['pickup_ratio'] = df['pickup_in_meters'] / df['distance_in_meters']
    df['pickup_ratio'] = df['pickup_ratio'].replace([np.inf, -np.inf], np.nan)
    df['price_increase_pct'] = ((df['price_bid_local'] - df['price_start_local']) / df['price_start_local'] * 100)
    
    # Создаём маски для каждой проблемы
    problems = {}
    
    # 1. Временные аномалии
    problems['future_driver'] = (df['driver_reg_date'] > df['order_timestamp'])
    problems['future_bid'] = (df['tender_timestamp'] < df['order_timestamp'])
    problems['slow_response'] = (df['response_time_seconds'] > 300)  # >5 минут
    
    # 2. Нулевые/отрицательные значения
    problems['zero_distance'] = (df['distance_in_meters'] <= 0)
    problems['zero_duration'] = (df['duration_in_seconds'] <= 0)
    problems['zero_price'] = (df['price_bid_local'] <= 0)
    
    # 3. Экстремальные значения
    problems['extreme_distance'] = (df['distance_in_meters'] > 100000)  # >100 км
    problems['extreme_duration'] = (df['duration_in_seconds'] > 7200)   # >2 часа
    
    # 4. Проверка физической возможности скорости
    problems['too_fast_city'] = (df['avg_speed_kmh'].notna()) & (df['avg_speed_kmh'] > 80)
    min_possible_duration = df['distance_in_meters'] / (120 / 3.6)
    problems['physically_impossible'] = (df['duration_in_seconds'] < min_possible_duration)
    problems['too_slow'] = (df['avg_speed_kmh'].notna()) & (df['avg_speed_kmh'] < 8)
    
    # 5. Аномалии в подаче
    problems['extreme_pickup_ratio'] = (df['pickup_ratio'].notna()) & (df['pickup_ratio'] > 5)
    problems['extreme_pickup_speed'] = (df['pickup_speed_kmh'].notna()) & (df['pickup_speed_kmh'] > 100)
    
    # 6. Ценовые аномалии
    problems['extreme_markup'] = (df['price_increase_pct'] > 100)
    problems['extreme_price'] = (df['price_bid_local'] > 5000)
    
    # Вывод статистики
    if verbose:
        descriptions = {
            'future_driver': 'Водитель зарегистрирован после заказа',
            'future_bid': 'Бид отправлен до создания заказа',
            'slow_response': 'Медленный ответ водителя (>5 мин)',
            'zero_distance': 'Нулевое/отрицательное расстояние',
            'zero_duration': 'Нулевая/отрицательная длительность',
            'zero_price': 'Нулевая/отрицательная цена',
            'extreme_distance': 'Слишком длинная поездка (>100 км)',
            'extreme_duration': 'Слишком долгая поездка (>2 ч)',
            'too_fast_city': 'Слишком высокая скорость (>80 км/ч)',
            'physically_impossible': 'Физически невозможная скорость (>120 км/ч)',
            'too_slow': 'Слишком низкая скорость (<8 км/ч)',
            'extreme_pickup_ratio': 'Подача >5x длиннее поездки',
            'extreme_pickup_speed': 'Скорость подачи >100 км/ч',
            'extreme_markup': 'Наценка >100%',
            'extreme_price': 'Цена >5000₽'
        }
        
        print(f"{'Проблема':<50s} {'Записей':>10s}")
        print("-"*62)
        
        for name, mask in problems.items():
            count = mask.sum()
            if count > 0:
                desc = descriptions.get(name, name)
                print(f"{desc:<50s} {count:>10d}")
    
    # Комбинированная маска удаления
    delete_mask = pd.Series(False, index=df.index)
    for mask in problems.values():
        delete_mask |= mask
    
    # Применяем фильтр
    df_clean = df[~delete_mask].copy()
    
    if verbose:
        print("-"*62)
        print(f"{'ИТОГО удалено:':<50s} {delete_mask.sum():>10d} ({delete_mask.sum()/initial_count*100:.1f}%)")
        print(f"{'Осталось записей:':<50s} {len(df_clean):>10d} ({len(df_clean)/initial_count*100:.1f}%)")
        
        # Баланс классов
        df_clean['is_done_binary'] = (df_clean['is_done'] == 'done').astype(int)
        done_count = df_clean['is_done_binary'].sum()
        cancel_count = len(df_clean) - done_count
        done_pct = done_count / len(df_clean) * 100
        
        print(f"\n{'Баланс классов:':<50s}")
        print(f"{'  Done':<50s} {done_count:>10d} ({done_pct:5.1f}%)")
        print(f"{'  Cancel':<50s} {cancel_count:>10d} ({100-done_pct:5.1f}%)")
        
        ratio = cancel_count / done_count
        print(f"\n{'scale_pos_weight для XGBoost:':<50s} {ratio:>10.2f}")
        print("="*70)
    
    return df_clean


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


def build_enhanced_features(frame):
    """
    Строит расширенный набор признаков с акцентом на ценовые факторы
    Включает price_bid_local и производные для правильного обучения модели!
    """
    
    ts = pd.to_datetime(frame["order_timestamp"], errors="coerce")
    hour = ts.dt.hour.fillna(0)
    wday = ts.dt.weekday.fillna(0)
    
    features = {}
    
    # ========================================================================
    # КРИТИЧЕСКИ ВАЖНО: ЦЕНОВЫЕ ПРИЗНАКИ
    # ========================================================================
    features['price_bid_local'] = frame['price_bid_local'].values
    features['price_start_local'] = frame['price_start_local'].values
    features['price_increase_abs'] = (frame['price_bid_local'] - frame['price_start_local']).values
    features['price_increase_pct'] = ((frame['price_bid_local'] - frame['price_start_local']) / 
                                      frame['price_start_local'] * 100).values
    features['is_price_increased'] = (features['price_increase_pct'] > 0).astype(float)
    
    # Нормализованные цены (с защитой от деления на 0)
    features['price_per_km'] = frame['price_bid_local'] / (frame['distance_in_meters'] / 1000 + 0.1)
    features['price_per_minute'] = frame['price_bid_local'] / (frame['duration_in_seconds'] / 60 + 0.1)
    
    # ========================================================================
    # ВРЕМЕННЫЕ ПРИЗНАКИ
    # ========================================================================
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    features['day_of_week'] = wday.values
    features['day_sin'] = np.sin(2 * np.pi * wday / 7)
    features['day_cos'] = np.cos(2 * np.pi * wday / 7)
    features['is_weekend'] = (wday >= 5).astype(float).values
    
    # Часы пик
    is_morning_rush = ((hour >= 7) & (hour <= 9)).astype(int)
    is_evening_rush = ((hour >= 17) & (hour <= 20)).astype(int)
    features['is_morning_peak'] = is_morning_rush.values
    features['is_evening_peak'] = is_evening_rush.values
    features['is_peak_hour'] = ((is_morning_rush + is_evening_rush) > 0).astype(float).values
    features['is_night'] = ((hour < 6) | (hour >= 22)).astype(float).values
    features['is_lunch_time'] = ((hour >= 12) & (hour <= 14)).astype(float).values
    
    # ========================================================================
    # ХАРАКТЕРИСТИКИ ПОЕЗДКИ
    # ========================================================================
    features['distance_in_meters'] = frame['distance_in_meters'].values
    features['duration_in_seconds'] = frame['duration_in_seconds'].values
    features['distance_km'] = (frame['distance_in_meters'] / 1000).values
    features['duration_min'] = (frame['duration_in_seconds'] / 60).values
    
    # Скорость (с защитой от деления на 0)
    speed = frame['distance_in_meters'] / (frame['duration_in_seconds'] + 0.1) * 3.6
    features['avg_speed_kmh'] = np.clip(speed.values, 0, 150)  # Клипнем в разумные пределы
    features['is_traffic_jam'] = (features['avg_speed_kmh'] < 15).astype(float)
    features['is_highway'] = (features['avg_speed_kmh'] > 50).astype(float)
    
    # Категории дистанции
    dist = features['distance_km']
    features['is_short_trip'] = (dist < 2).astype(float)
    features['is_medium_trip'] = ((dist >= 2) & (dist < 10)).astype(float)
    features['is_long_trip'] = (dist >= 10).astype(float)
    
    # ========================================================================
    # ПОДАЧА ВОДИТЕЛЯ
    # ========================================================================
    features['pickup_in_meters'] = frame['pickup_in_meters'].values
    features['pickup_in_seconds'] = frame['pickup_in_seconds'].values
    features['pickup_km'] = (frame['pickup_in_meters'] / 1000).values
    
    pickup_speed = frame['pickup_in_meters'] / (frame['pickup_in_seconds'] + 0.1) * 3.6
    features['pickup_speed_kmh'] = np.clip(pickup_speed.values, 0, 150)
    
    # Соотношения (с защитой от деления на 0)
    features['pickup_to_trip_ratio'] = np.clip(
        frame['pickup_in_meters'] / (frame['distance_in_meters'] + 1).values,
        0, 10  # Максимум 10x
    )
    features['pickup_time_ratio'] = np.clip(
        frame['pickup_in_seconds'] / (frame['duration_in_seconds'] + 1).values,
        0, 10
    )
    features['total_distance'] = (frame['pickup_in_meters'] + frame['distance_in_meters']).values
    features['total_time'] = (frame['pickup_in_seconds'] + frame['duration_in_seconds']).values
    
    # ========================================================================
    # ВОДИТЕЛЬ
    # ========================================================================
    features['driver_rating'] = frame['driver_rating'].values
    
    # Стаж водителя
    driver_reg = pd.to_datetime(frame['driver_reg_date'], errors='coerce')
    experience_days = (ts - driver_reg).dt.days.fillna(365)
    experience_days = np.clip(experience_days.values, 0, 3650)  # Макс 10 лет
    features['driver_experience_days'] = experience_days
    features['driver_experience_years'] = experience_days / 365.25
    features['is_new_driver'] = (experience_days < 30).astype(float)
    features['is_experienced_driver'] = (experience_days > 365).astype(float)
    features['has_perfect_rating'] = (frame['driver_rating'] == 5.0).astype(float).values
    features['rating_deviation'] = (5.0 - frame['driver_rating']).values
    
    # Время ответа
    tender_ts = pd.to_datetime(frame['tender_timestamp'], errors='coerce')
    response_time = (tender_ts - ts).dt.total_seconds().fillna(30)
    response_time = np.clip(response_time.values, 0, 600)  # Макс 10 минут
    features['response_time_seconds'] = response_time
    features['response_time_log'] = np.log1p(response_time)
    features['is_fast_response'] = (response_time < 10).astype(float)
    features['is_slow_response'] = (response_time > 60).astype(float)
    
    # ========================================================================
    # АВТОМОБИЛЬ
    # ========================================================================
    taxi_types = frame.apply(lambda row: detect_taxi_type(row['carname'], row['carmodel']), axis=1)
    features['taxi_type_economy'] = (taxi_types == 'economy').astype(float).values
    features['taxi_type_comfort'] = (taxi_types == 'comfort').astype(float).values
    features['taxi_type_business'] = (taxi_types == 'business').astype(float).values
    
    # Платформа
    features['platform_android'] = (frame['platform'] == 'android').astype(float).values
    features['platform_ios'] = (frame['platform'] == 'ios').astype(float).values
    
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
    result = pd.DataFrame(features)
    
    # ========================================================================
    # УСИЛЕННАЯ ОБРАБОТКА NaN и Inf
    # ========================================================================
    
    # 1. Заменяем Inf на NaN
    result = result.replace([np.inf, -np.inf], np.nan)
    
    # 2. Заполняем NaN медианой ИЛИ нулём если медиана не определена
    for col in result.columns:
        if result[col].isna().any():
            median_val = result[col].median()
            if pd.isna(median_val) or np.isinf(median_val):
                result[col] = result[col].fillna(0)
            else:
                result[col] = result[col].fillna(median_val)
    
    # 3. Финальная проверка и очистка
    result = result.fillna(0)  # На всякий случай
    result = result.replace([np.inf], 1e10)
    result = result.replace([-np.inf], -1e10)
    
    # 4. Клипнем все значения в разумные пределы
    for col in result.columns:
        if result[col].std() > 0:  # Только для изменяющихся признаков
            mean = result[col].mean()
            std = result[col].std()
            result[col] = np.clip(result[col], mean - 10*std, mean + 10*std)
    
    return result


def train_model(train_path="simple-train.csv", use_gpu=False, test_size=0.2, random_state=42):
    """
    Обучает модель XGBoost с калибровкой вероятностей
    """
    
    print("\n" + "="*70)
    print("ОБУЧЕНИЕ ML-МОДЕЛИ DRIVEE")
    print("="*70)
    
    # 1. Загрузка данных
    print(f"\n📁 Загрузка данных из {train_path}...")
    df = pd.read_csv(train_path)
    print(f"   Загружено: {len(df)} записей")
    
    # 2. КРИТИЧНО: Очистка данных
    df = clean_and_validate_data(df, verbose=True)
    
    if len(df) < 100:
        raise ValueError("⚠️ Слишком мало данных после очистки! Проверьте исходный датасет.")
    
    # 3. Целевая переменная
    print("\n🎯 Подготовка целевой переменной...")
    y = (df['is_done'] == 'done').astype(int)
    print(f"   Done: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   Cancel: {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")
    
    # 4. Feature Engineering
    print("\n🔧 Создание признаков...")
    X = build_enhanced_features(df)
    print(f"   Создано признаков: {X.shape[1]}")
    print(f"   Размер данных: {X.shape[0]} записей × {X.shape[1]} признаков")
    
    # 5. Разделение на train/test
    print(f"\n📊 Разделение данных (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"   Train: {len(X_train)} записей")
    print(f"   Test:  {len(X_test)} записей")
    
    # 6. Обучение XGBoost
    print("\n🤖 Обучение XGBoost...")
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 8,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': scale_pos_weight,
        'tree_method': 'gpu_hist' if use_gpu else 'hist',
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
    
    # 7. Калибровка вероятностей
    print("\n🎲 Калибровка вероятностей...")
    calibrated_model = CalibratedClassifierCV(
        model, 
        method='sigmoid',
        cv='prefit'
    )
    calibrated_model.fit(X_train, y_train)
    
    # 8. Оценка качества
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
    
    # Feature importance
    print("\n🔝 Топ-10 важных признаков:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:30s} {row['importance']:.4f}")
    
    # 9. Сохранение модели
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
