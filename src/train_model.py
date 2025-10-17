import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

def build_enhanced_features(frame):
    """Строит расширенный набор признаков (независимо от года/месяца)"""
    ts = pd.to_datetime(frame["order_timestamp"], errors="coerce")
    hour = ts.dt.hour.fillna(0)
    wday = ts.dt.weekday.fillna(0)
    
    # Временные признаки (независимы от года!)
    is_morning_rush = ((hour >= 7) & (hour <= 9)).astype(int)
    is_evening_rush = ((hour >= 15) & (hour <= 17)).astype(int)
    is_night_rush = ((hour >= 19) & (hour <= 21)).astype(int)
    is_peak_hour = ((is_morning_rush + is_evening_rush + is_night_rush) > 0).astype(float)
    is_weekend = (wday >= 5).astype(float)
    is_night = ((hour < 6) | (hour >= 22)).astype(float)
    
    # Базовые характеристики поездки
    dist_km = frame["distance_in_meters"] / 1000.0
    dur_min = frame["duration_in_seconds"] / 60.0
    pickup_km = frame["pickup_in_meters"] / 1000.0
    pickup_min = frame["pickup_in_seconds"] / 60.0
    log_start = np.log1p(frame["price_start_local"])
    price_per_km = frame["price_start_local"] / (dist_km + 0.1)
    
    # Стаж водителя
    driver_reg = pd.to_datetime(frame["driver_reg_date"], errors="coerce")
    days_since_reg = (ts - driver_reg).dt.days.fillna(0)
    driver_experience_months = days_since_reg / 30.0
    is_new_driver = (days_since_reg < 30).astype(float)
    
    # Марка автомобиля
    premium_brands = ['Toyota', 'Volkswagen', 'Hyundai', 'Nissan', 'Skoda']
    is_premium_car = frame.get("carname", pd.Series("", index=frame.index)).isin(premium_brands).astype(float)
    
    # Частота клиента (агрегация по user_id)
    if 'user_id' in frame.columns:
        user_counts = frame.groupby('user_id').size()
        frame['user_order_count'] = frame['user_id'].map(user_counts).fillna(1)
        is_frequent_user = (frame['user_order_count'] > 5).astype(float)
    else:
        is_frequent_user = pd.Series(0, index=frame.index)
    
    # Время отклика водителя
    if 'tender_timestamp' in frame.columns:
        tender_time = pd.to_datetime(frame["tender_timestamp"], errors="coerce")
        response_time_seconds = (tender_time - ts).dt.total_seconds().fillna(60)
        response_time_minutes = response_time_seconds / 60.0
        # Логарифм времени отклика (сглаживает выбросы)
        log_response_time = np.log1p(response_time_minutes)
    else:
        response_time_minutes = pd.Series(0, index=frame.index)
        log_response_time = pd.Series(0, index=frame.index)
    
    X = pd.DataFrame({
        # Базовые признаки
        "dist_km": dist_km,
        "dur_min": dur_min,
        "pickup_km": pickup_km,
        "pickup_min": pickup_min,
        "rating": frame.get("driver_rating", pd.Series(0, index=frame.index)),
        "log_start": log_start,
        
        # Временные признаки (циклические - НЕ зависят от года!)
        "hour_sin": np.sin(2*np.pi*hour/24.0),
        "hour_cos": np.cos(2*np.pi*hour/24.0),
        "wday_sin": np.sin(2*np.pi*wday/7.0),
        "wday_cos": np.cos(2*np.pi*wday/7.0),
        
        # Часы пик
        "is_peak_hour": is_peak_hour,
        "is_morning_rush": is_morning_rush.astype(float),
        "is_evening_rush": is_evening_rush.astype(float),
        "is_night_rush": is_night_rush.astype(float),
        "is_weekend": is_weekend,
        "is_night": is_night,
        
        # Производные признаки
        "speed_kmh": dist_km / ((dur_min + 1) / 60.0),
        "pickup_speed": pickup_km / ((pickup_min + 1) / 60.0),
        "total_time_min": dur_min + pickup_min,
        "price_per_km": price_per_km,
        "price_per_min": frame["price_start_local"] / (dur_min + 0.1),
        
        # Информация о водителе
        "driver_experience_months": driver_experience_months,
        "is_new_driver": is_new_driver,
        "is_premium_car": is_premium_car,
        
        # Информация о клиенте
        "is_frequent_user": is_frequent_user,
        
        # Время отклика
        "response_time_minutes": response_time_minutes,
        "log_response_time": log_response_time,
        
        # Базовые взаимодействия
        "rush_x_rating": is_peak_hour * frame.get("driver_rating", pd.Series(0, index=frame.index)),
        "weekend_x_dist": is_weekend * dist_km,
        "peak_x_price_per_km": is_peak_hour * price_per_km,
        "hour_x_weekend": hour * is_weekend / 24.0,
        "morning_x_dist": is_morning_rush.astype(float) * dist_km,
        "evening_x_dist": is_evening_rush.astype(float) * dist_km,
        
        # Усиленные временные взаимодействия
        "night_x_price": is_night * log_start,
        "night_x_dist": is_night * dist_km,
        "night_x_price_per_km": is_night * price_per_km,
        "hour_x_price_per_km": hour * price_per_km / 24.0,
        "hour_x_dist": hour * dist_km / 24.0,
        "peak_x_weekend_x_dist": is_peak_hour * is_weekend * dist_km,
        "peak_x_weekend_x_price": is_peak_hour * is_weekend * log_start,
        
        # Новые взаимодействия
        "premium_x_price": is_premium_car * log_start,
        "experience_x_rating": driver_experience_months * frame.get("driver_rating", pd.Series(0, index=frame.index)),
        "new_driver_x_price": is_new_driver * log_start,
        "frequent_user_x_price": is_frequent_user * log_start,
    }).fillna(0.0)
    
    X = X.replace([np.inf, -np.inf], 0)
    return X

def train_model(train_path="simple-train.csv"):
    """Обучает улучшенную модель с расширенными признаками"""
    print("📚 Загрузка данных...")
    df = pd.read_csv(train_path)
    
    # Проверяем наличие всех необходимых столбцов
    required_cols = ['order_timestamp', 'price_start_local', 'is_done', 'driver_reg_date']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют столбцы: {missing}")
    
    df = df.dropna(subset=required_cols)
    print(f"📊 Всего записей: {len(df)}")
    
    X = build_enhanced_features(df)
    y = (df["is_done"].astype(str).str.lower() == "done").astype(int)
    print(f"✅ Выполненных заказов: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
    print(f"❌ Отменённых заказов: {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.2f}%)")
    
    print(f"\n🔧 Использовано признаков: {len(X.columns)}")
    print(f"   Новые: стаж водителя, марка машины, частота клиента, время отклика")
    print(f"   Модель НЕ зависит от года/месяца - только день недели и час суток")
    
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    base_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=50,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42
    )
    
    print("\n🚀 Обучение модели...")
    clf = CalibratedClassifierCV(base_model, cv=5, method="sigmoid")
    clf.fit(Xtr, ytr)
    
    train_score = clf.score(Xtr, ytr)
    test_score = clf.score(Xte, yte)
    print(f"📈 Точность на train: {train_score:.4f}")
    print(f"📈 Точность на test: {test_score:.4f}")
    
    joblib.dump({
        "model": clf,
        "feat_cols": X.columns.tolist()
    }, "model_enhanced.joblib")
    
    print("\n✅ Модель обучена и сохранена в model_enhanced.joblib")
    return clf

if __name__ == "__main__":
    train_model()
