import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import time

def build_features_with_order(order_dict, feat_cols):
    """Строит признаки для одного заказа"""
    frame = pd.DataFrame([order_dict])
    
    # Обрабатываем timestamp
    if isinstance(frame["order_timestamp"].iloc[0], (int, float)):
        frame["order_timestamp"] = pd.to_datetime(frame["order_timestamp"], unit='s', errors="coerce")
    else:
        frame["order_timestamp"] = pd.to_datetime(frame["order_timestamp"], errors="coerce")
    
    ts = frame["order_timestamp"]
    hour = ts.dt.hour.fillna(0)
    wday = ts.dt.weekday.fillna(0)
    
    is_morning_rush = ((hour >= 7) & (hour <= 9)).astype(int)
    is_evening_rush = ((hour >= 15) & (hour <= 17)).astype(int)
    is_night_rush = ((hour >= 19) & (hour <= 21)).astype(int)
    is_peak_hour = ((is_morning_rush + is_evening_rush + is_night_rush) > 0).astype(float)
    is_weekend = (wday >= 5).astype(float)
    is_night = ((hour < 6) | (hour >= 22)).astype(float)
    
    dist_km = frame["distance_in_meters"]/1000.0
    dur_min = frame["duration_in_seconds"]/60.0
    pickup_km = frame["pickup_in_meters"]/1000.0
    pickup_min = frame["pickup_in_seconds"]/60.0
    log_start = np.log1p(frame["price_start_local"])
    price_per_km = frame["price_start_local"] / (dist_km + 0.1)
    
    # Стаж водителя (если указан)
    if "driver_reg_date" in frame.columns and pd.notna(frame["driver_reg_date"].iloc[0]):
        driver_reg = pd.to_datetime(frame["driver_reg_date"], errors="coerce")
        days_since_reg = (ts - driver_reg).dt.days.fillna(180)
        driver_experience_months = days_since_reg / 30.0
        is_new_driver = (days_since_reg < 30).astype(float)
    else:
        driver_experience_months = pd.Series(6.0, index=frame.index)
        is_new_driver = pd.Series(0.0, index=frame.index)
    
    # Марка машины
    premium_brands = ['Toyota', 'Volkswagen', 'Hyundai', 'Nissan', 'Skoda']
    if "carname" in frame.columns:
        is_premium_car = frame["carname"].isin(premium_brands).astype(float)
    else:
        is_premium_car = pd.Series(0.0, index=frame.index)
    
    # Частота клиента (по умолчанию средний пользователь)
    is_frequent_user = pd.Series(0.5, index=frame.index)
    
    # Время отклика (по умолчанию 1 минута)
    response_time_minutes = pd.Series(1.0, index=frame.index)
    log_response_time = pd.Series(0.0, index=frame.index)
    
    base = pd.DataFrame({
        "dist_km": dist_km,
        "dur_min": dur_min,
        "pickup_km": pickup_km,
        "pickup_min": pickup_min,
        "rating": frame.get("driver_rating", 4.5),
        "log_start": log_start,
        "hour_sin": np.sin(2*np.pi*hour/24.0),
        "hour_cos": np.cos(2*np.pi*hour/24.0),
        "wday_sin": np.sin(2*np.pi*wday/7.0),
        "wday_cos": np.cos(2*np.pi*wday/7.0),
        "is_peak_hour": is_peak_hour,
        "is_morning_rush": is_morning_rush.astype(float),
        "is_evening_rush": is_evening_rush.astype(float),
        "is_night_rush": is_night_rush.astype(float),
        "is_weekend": is_weekend,
        "is_night": is_night,
        "speed_kmh": dist_km / ((dur_min + 1) / 60.0),
        "pickup_speed": pickup_km / ((pickup_min + 1) / 60.0),
        "total_time_min": dur_min + pickup_min,
        "price_per_km": price_per_km,
        "price_per_min": frame["price_start_local"] / (dur_min + 0.1),
        "driver_experience_months": driver_experience_months,
        "is_new_driver": is_new_driver,
        "is_premium_car": is_premium_car,
        "is_frequent_user": is_frequent_user,
        "response_time_minutes": response_time_minutes,
        "log_response_time": log_response_time,
        "rush_x_rating": is_peak_hour * frame.get("driver_rating", 4.5),
        "weekend_x_dist": is_weekend * dist_km,
        "peak_x_price_per_km": is_peak_hour * price_per_km,
        "hour_x_weekend": hour * is_weekend / 24.0,
        "morning_x_dist": is_morning_rush.astype(float) * dist_km,
        "evening_x_dist": is_evening_rush.astype(float) * dist_km,
        "night_x_price": is_night * log_start,
        "night_x_dist": is_night * dist_km,
        "night_x_price_per_km": is_night * price_per_km,
        "hour_x_price_per_km": hour * price_per_km / 24.0,
        "hour_x_dist": hour * dist_km / 24.0,
        "peak_x_weekend_x_dist": is_peak_hour * is_weekend * dist_km,
        "peak_x_weekend_x_price": is_peak_hour * is_weekend * log_start,
        "premium_x_price": is_premium_car * log_start,
        "experience_x_rating": driver_experience_months * frame.get("driver_rating", 4.5),
        "new_driver_x_price": is_new_driver * log_start,
        "frequent_user_x_price": is_frequent_user * log_start,
    }).fillna(0.0)
    
    base = base.replace([np.inf, -np.inf], 0)
    base = base.reindex(columns=feat_cols, fill_value=0.0)
    return base

def recommend_price(order, output_format="json"):
    """Рекомендует оптимальные зоны цен"""
    
    model_path = "model_enhanced.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель {model_path} не найдена.")
    
    model_data = joblib.load(model_path)
    classifier = model_data["model"]
    feature_columns = model_data["feat_cols"]
    
    order_copy = order.copy()
    
    # Обрабатываем timestamp
    if isinstance(order.get("order_timestamp"), (int, float)):
        current_timestamp = int(order["order_timestamp"])
        current_dt = datetime.fromtimestamp(current_timestamp)
    else:
        current_dt = datetime.strptime(order["order_timestamp"], "%Y-%m-%d %H:%M:%S")
        current_timestamp = int(current_dt.timestamp())
    
    start_price = float(order["price_start_local"])
    
    # Динамический диапазон (НЕ зависит от года, только от дня недели и часа)
    hour = current_dt.hour
    wday = current_dt.weekday()
    
    is_peak = (7 <= hour <= 9) or (15 <= hour <= 17) or (19 <= hour <= 21)
    is_night = (hour < 6) or (hour >= 22)
    is_weekend = wday >= 5
    
    min_scan = start_price
    
    # Максимум зависит только от времени суток и дня недели
    if is_night:
        max_coef = 2.00
    elif is_peak and not is_weekend:
        max_coef = 2.20
    elif is_weekend:
        max_coef = 1.80
    else:
        max_coef = 1.60
    
    max_scan = start_price * max_coef
    price_candidates = np.linspace(min_scan, max_scan, 200)
    
    price_results = []
    for candidate_price in price_candidates:
        order_copy["price_start_local"] = candidate_price
        X_candidate = build_features_with_order(order_copy, feature_columns)
        prob_done = classifier.predict_proba(X_candidate)[0, 1]
        expected_value = candidate_price * prob_done
        
        price_results.append({
            "price": float(candidate_price),
            "probability": float(prob_done),
            "expected_value": float(expected_value)
        })
    
    optimal_result = max(price_results, key=lambda x: x["expected_value"])
    optimal_price = optimal_result["price"]
    optimal_prob = optimal_result["probability"]
    optimal_ev = optimal_result["expected_value"]
    
    max_prob = max(r["probability"] for r in price_results)
    max_prob_result = max(price_results, key=lambda x: x["probability"])
    
    # Зоны (4 зоны)
    zone_config = [
        {"zone_id": 1, "zone_name": "zone_1_red_low", "price_range": (min_scan, optimal_price * 0.75)},
        {"zone_id": 2, "zone_name": "zone_2_yellow_low", "price_range": (optimal_price * 0.75, optimal_price * 0.90)},
        {"zone_id": 3, "zone_name": "zone_3_green", "price_range": (optimal_price * 0.90, optimal_price * 1.10)},
        {"zone_id": 4, "zone_name": "zone_4_yellow_high", "price_range": (optimal_price * 1.10, max_scan)}
    ]
    
    zones_data = []
    for zone in zone_config:
        min_price_zone, max_price_zone = zone["price_range"]
        zone_prices = [r for r in price_results 
                      if min_price_zone <= r["price"] < max_price_zone]
        
        if not zone_prices:
            continue
        
        avg_prob = np.mean([r["probability"] for r in zone_prices])
        avg_ev = np.mean([r["expected_value"] for r in zone_prices])
        actual_min_price = min(r["price"] for r in zone_prices)
        actual_max_price = max(r["price"] for r in zone_prices)
        
        zones_data.append({
            "zone_id": zone["zone_id"],
            "zone_name": zone["zone_name"],
            "price_range": {
                "min": round(actual_min_price, 2),
                "max": round(actual_max_price, 2)
            },
            "metrics": {
                "avg_probability_percent": round(avg_prob * 100, 2),
                "avg_normalized_probability_percent": round((avg_prob / max_prob) * 100, 2),
                "avg_expected_value": round(avg_ev, 2)
            }
        })
    
    optimal_zone_id = 3
    for zone in zones_data:
        if zone["price_range"]["min"] <= optimal_price <= zone["price_range"]["max"]:
            optimal_zone_id = zone["zone_id"]
            break
    
    result = {
        "zones": zones_data,
        "optimal_price": {
            "price": round(optimal_price, 2),
            "probability_percent": round(optimal_prob * 100, 2),
            "normalized_probability_percent": round((optimal_prob / max_prob) * 100, 2),
            "expected_value": round(optimal_ev, 2),
            "zone_id": optimal_zone_id
        },
        "analysis": {
            "start_price": round(start_price, 2),
            "max_probability_percent": round(max_prob * 100, 2),
            "max_probability_price": round(max_prob_result["price"], 2),
            "scan_range": {
                "min": round(min_scan, 2),
                "max": round(max_scan, 2)
            },
            "timestamp": current_timestamp
        }
    }
    
    if output_format == "json":
        return json.dumps(result, ensure_ascii=False, indent=2)
    else:
        return result
