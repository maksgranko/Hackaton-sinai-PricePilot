import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import joblib

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
    """Строит расширенный набор признаков"""
    ts = pd.to_datetime(frame["order_timestamp"], errors="coerce")
    hour = ts.dt.hour.fillna(0)
    wday = ts.dt.weekday.fillna(0)
    
    # Временные признаки
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
    
    # Тип такси
    if 'carname' in frame.columns and 'carmodel' in frame.columns:
        taxi_types = frame.apply(lambda row: detect_taxi_type(row['carname'], row['carmodel']), axis=1)
        is_economy = (taxi_types == 'economy').astype(float)
        is_comfort = (taxi_types == 'comfort').astype(float)
        is_business = (taxi_types == 'business').astype(float)
    else:
        is_economy = pd.Series(0.3, index=frame.index)
        is_comfort = pd.Series(0.5, index=frame.index)
        is_business = pd.Series(0.2, index=frame.index)
    
    # Марка автомобиля (премиум)
    premium_brands = ['Toyota', 'Volkswagen', 'Hyundai', 'Nissan', 'Skoda']
    is_premium_car = frame.get("carname", pd.Series("", index=frame.index)).isin(premium_brands).astype(float)
    
    # Частота клиента
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
        log_response_time = np.log1p(response_time_minutes)
    else:
        response_time_minutes = pd.Series(0, index=frame.index)
        log_response_time = pd.Series(0, index=frame.index)
    
    # Усиленные временные взаимодействия
    hour_normalized = hour / 24.0
    rating = frame.get("driver_rating", pd.Series(0, index=frame.index))
    
    hour_x_rating = hour_normalized * rating
    night_x_rating = is_night * rating
    peak_x_rating_strong = is_peak_hour * rating * 2
    
    hour_x_dist_strong = hour_normalized * dist_km * 10
    night_x_dist_strong = is_night * dist_km * 5
    weekend_x_hour = is_weekend * hour_normalized
    peak_x_dist_strong = is_peak_hour * dist_km * 3
    
    X = pd.DataFrame({
        # Базовые признаки
        "dist_km": dist_km,
        "dur_min": dur_min,
        "pickup_km": pickup_km,
        "pickup_min": pickup_min,
        "rating": rating,
        "log_start": log_start,
        
        # Временные признаки (циклические)
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
        
        # Тип такси (НОВОЕ!)
        "is_economy": is_economy,
        "is_comfort": is_comfort,
        "is_business": is_business,
        
        # Информация о клиенте
        "is_frequent_user": is_frequent_user,
        
        # Время отклика
        "response_time_minutes": response_time_minutes,
        "log_response_time": log_response_time,
        
        # Базовые взаимодействия
        "rush_x_rating": is_peak_hour * rating,
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
        "experience_x_rating": driver_experience_months * rating,
        "new_driver_x_price": is_new_driver * log_start,
        "frequent_user_x_price": is_frequent_user * log_start,
        
        # Усиленные временные взаимодействия
        "hour_x_rating": hour_x_rating,
        "night_x_rating": night_x_rating,
        "peak_x_rating_strong": peak_x_rating_strong,
        "hour_x_dist_strong": hour_x_dist_strong,
        "night_x_dist_strong": night_x_dist_strong,
        "weekend_x_hour": weekend_x_hour,
        "peak_x_dist_strong": peak_x_dist_strong,
        
        # Взаимодействия с типом такси
        "business_x_dist": is_business * dist_km,
        "business_x_price": is_business * log_start,
        "economy_x_price": is_economy * log_start,
    }).fillna(0.0)
    
    X = X.replace([np.inf, -np.inf], 0)
    return X

def train_model(train_path="simple-train.csv", use_gpu=False):
    """Обучает модель с XGBoost"""
    print("📚 Загрузка данных...")
    df = pd.read_csv(train_path)
    
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
    print(f"   + Тип такси: economy/comfort/business")
    print(f"   + Усиленные временные взаимодействия")
    
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if use_gpu:
        print("\n🚀 Обучение XGBoost с GPU (gpu_hist)...")
        tree_method = 'gpu_hist'
        gpu_id = 0
    else:
        print("\n🚀 Обучение XGBoost с CPU (hist, многопоточность)...")
        tree_method = 'hist'
        gpu_id = -1
    
    base_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=15,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method=tree_method,
        gpu_id=gpu_id,
        n_jobs=-1,
        random_state=42,
        eval_metric='logloss'
    )
    
    base_model.fit(Xtr, ytr)
    
    print("\n🔍 Топ-20 самых важных признаков:")
    feature_importance = base_model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False).head(20)
    
    for idx, row in importance_df.iterrows():
        print(f"   {row['feature']:<35} {row['importance']:.4f}")
    
    print("\n🔧 Калибровка вероятностей (CalibratedClassifierCV)...")
    clf = CalibratedClassifierCV(base_model, cv=3, method="sigmoid", n_jobs=-1)
    clf.fit(Xtr, ytr)
    
    train_score = clf.score(Xtr, ytr)
    test_score = clf.score(Xte, yte)
    print(f"\n📈 Точность на train: {train_score:.4f}")
    print(f"📈 Точность на test: {test_score:.4f}")
    
    joblib.dump({
        "model": clf,
        "feat_cols": X.columns.tolist()
    }, "model_enhanced.joblib")
    
    print("\n✅ Модель обучена и сохранена в model_enhanced.joblib")
    if use_gpu:
        print("🎮 Использован GPU для ускорения")
    else:
        print("💻 Использован CPU (многопоточность)")
    
    return clf

if __name__ == "__main__":
    try:
        train_model(use_gpu=True)
    except Exception as e:
        print(f"\n⚠️  GPU недоступен: {e}")
        print("🔄 Переключаемся на CPU...")
        train_model(use_gpu=False)
