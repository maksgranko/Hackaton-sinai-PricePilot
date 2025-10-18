import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

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

def build_features_for_price(order_data, price_bid, reference_price):
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
        'price_start_local': reference_price,
        'price_bid_local': price_bid
    }])
    ts = pd.to_datetime(temp_df["order_timestamp"], errors="coerce")
    hour = ts.dt.hour.fillna(0)
    wday = ts.dt.weekday.fillna(0)
    features = {}
    features['price_bid_local'] = price_bid
    features['price_start_local'] = reference_price
    features['price_increase_abs'] = price_bid - reference_price
    features['price_increase_pct'] = ((price_bid - reference_price) / reference_price * 100)
    features['is_price_increased'] = float(features['price_increase_pct'] > 0)
    features['price_per_km'] = price_bid / (order_data['distance_in_meters'] / 1000 + 0.1)
    features['price_per_minute'] = price_bid / (order_data['duration_in_seconds'] / 60 + 0.1)
    h = hour.iloc[0]
    w = wday.iloc[0]
    features['hour_sin'] = np.sin(2 * np.pi * h / 24)
    features['hour_cos'] = np.cos(2 * np.pi * h / 24)
    features['day_of_week'] = w
    features['day_sin'] = np.sin(2 * np.pi * w / 7)
    features['day_cos'] = np.cos(2 * np.pi * w / 7)
    features['is_weekend'] = float(w >= 5)
    is_morning = 1 if 7 <= h <= 9 else 0
    is_evening = 1 if 17 <= h <= 20 else 0
    features['is_morning_peak'] = float(is_morning)
    features['is_evening_peak'] = float(is_evening)
    features['is_peak_hour'] = float(is_morning or is_evening)
    features['is_night'] = float(h < 6 or h >= 22)
    features['is_lunch_time'] = float(12 <= h <= 14)
    dist_m = order_data['distance_in_meters']
    dur_s = order_data['duration_in_seconds']
    features['distance_in_meters'] = dist_m
    features['duration_in_seconds'] = dur_s
    features['distance_km'] = dist_m / 1000
    features['duration_min'] = dur_s / 60
    avg_speed = (dist_m / dur_s * 3.6) if dur_s > 0 else 25.0
    features['avg_speed_kmh'] = avg_speed
    features['is_traffic_jam'] = float(avg_speed < 15)
    features['is_highway'] = float(avg_speed > 50)
    dist_km = features['distance_km']
    features['is_short_trip'] = float(dist_km < 2)
    features['is_medium_trip'] = float(2 <= dist_km < 10)
    features['is_long_trip'] = float(dist_km >= 10)
    pickup_m = order_data['pickup_in_meters']
    pickup_s = order_data['pickup_in_seconds']
    features['pickup_in_meters'] = pickup_m
    features['pickup_in_seconds'] = pickup_s
    features['pickup_km'] = pickup_m / 1000
    pickup_speed = (pickup_m / pickup_s * 3.6) if pickup_s > 0 else 20.0
    features['pickup_speed_kmh'] = pickup_speed
    features['pickup_to_trip_ratio'] = pickup_m / (dist_m + 1)
    features['pickup_time_ratio'] = pickup_s / (dur_s + 1)
    features['total_distance'] = pickup_m + dist_m
    features['total_time'] = pickup_s + dur_s
    features['driver_rating'] = order_data.get('driver_rating', 5.0)
    try:
        driver_reg = pd.to_datetime(order_data.get('driver_reg_date', '2020-01-01'))
        order_ts = ts.iloc[0]
        exp_days = (order_ts - driver_reg).days
        if exp_days < 0:
            exp_days = 365
    except:
        exp_days = 365
    features['driver_experience_days'] = exp_days
    features['driver_experience_years'] = exp_days / 365.25
    features['is_new_driver'] = float(exp_days < 30)
    features['is_experienced_driver'] = float(exp_days > 365)
    features['has_perfect_rating'] = float(features['driver_rating'] == 5.0)
    features['rating_deviation'] = 5.0 - features['driver_rating']
    response_time = 30.0
    features['response_time_seconds'] = response_time
    features['response_time_log'] = np.log1p(response_time)
    features['is_fast_response'] = float(response_time < 10)
    features['is_slow_response'] = float(response_time > 60)
    taxi_type = detect_taxi_type(
        order_data.get('carname', 'Renault'),
        order_data.get('carmodel', 'Logan')
    )
    features['taxi_type_economy'] = float(taxi_type == 'economy')
    features['taxi_type_comfort'] = float(taxi_type == 'comfort')
    features['taxi_type_business'] = float(taxi_type == 'business')
    platform = order_data.get('platform', 'android')
    features['platform_android'] = float(platform == 'android')
    features['platform_ios'] = float(platform == 'ios')
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
    result = pd.DataFrame([features])
    return result

def estimate_reference_price(order_data):
    dist_km = order_data['distance_in_meters'] / 1000
    dur_min = order_data['duration_in_seconds'] / 60
    base_price = 100
    price_per_km = 15
    price_per_min = 5
    estimated = base_price + (dist_km * price_per_km) + (dur_min * price_per_min)
    return estimated

def find_optimal_price(order_data, model, num_points=500):
    user_min_price = order_data['price_start_local']
    reference_price = estimate_reference_price(order_data)
    
    search_min = min(user_min_price, reference_price * 0.5)
    search_max = reference_price * 3
    
    test_prices = np.linspace(search_min, search_max, 100)
    test_probs = []
    for price in test_prices:
        features = build_features_for_price(order_data, price, reference_price)
        prob = model.predict_proba(features)[0, 1]
        test_probs.append(prob)
    
    test_probs = np.array(test_probs)
    prob_threshold = 0.05
    valid_indices = test_probs >= prob_threshold
    if valid_indices.any():
        max_price = test_prices[valid_indices][-1]
    else:
        max_price = reference_price * 1.5
    
    prices = np.linspace(search_min, max_price, num_points)
    probabilities = []
    expected_values = []
    for price in prices:
        features = build_features_for_price(order_data, price, reference_price)
        prob = model.predict_proba(features)[0, 1]
        expected_value = price * prob
        probabilities.append(prob)
        expected_values.append(expected_value)
    
    probabilities = np.array(probabilities)
    expected_values = np.array(expected_values)
    
    valid_mask = prices >= user_min_price
    if not valid_mask.any():
        valid_mask = np.ones(len(prices), dtype=bool)
    
    valid_prices = prices[valid_mask]
    valid_probs = probabilities[valid_mask]
    valid_expected_values = expected_values[valid_mask]
    
    max_prob = probabilities.max()
    normalized_probs = probabilities / max_prob if max_prob > 0 else probabilities
    valid_normalized_probs = normalized_probs[valid_mask]
    
    best_idx = np.argmax(valid_expected_values)
    optimal_price = valid_prices[best_idx]
    optimal_prob = valid_probs[best_idx]
    optimal_normalized_prob = valid_normalized_probs[best_idx]
    optimal_expected_value = valid_expected_values[best_idx]
    
    max_prob_idx = np.argmax(probabilities)
    max_prob_price = prices[max_prob_idx]
    
    sorted_indices = np.argsort(valid_expected_values)[::-1]
    
    num_zones = 4
    zone_size = len(sorted_indices) // num_zones
    zones = []
    zone_configs = [
        {'id': 1, 'name': 'zone_1_red_low'},
        {'id': 2, 'name': 'zone_2_yellow_low'},
        {'id': 3, 'name': 'zone_3_green'},
        {'id': 4, 'name': 'zone_4_yellow_high'}
    ]
    
    for i in range(num_zones):
        start_idx = i * zone_size
        end_idx = (i + 1) * zone_size if i < num_zones - 1 else len(sorted_indices)
        zone_indices = sorted_indices[start_idx:end_idx]
        
        zone_prices = valid_prices[zone_indices]
        zone_probs = valid_probs[zone_indices]
        zone_normalized_probs = valid_normalized_probs[zone_indices]
        zone_expected_values = valid_expected_values[zone_indices]
        
        zones.append({
            'zone_id': zone_configs[i]['id'],
            'zone_name': zone_configs[i]['name'],
            'price_range': {
                'min': round(float(zone_prices.min()), 2),
                'max': round(float(zone_prices.max()), 2)
            },
            'metrics': {
                'avg_probability_percent': round(float(zone_probs.mean() * 100), 2),
                'avg_normalized_probability_percent': round(float(zone_normalized_probs.mean() * 100), 2),
                'avg_expected_value': round(float(zone_expected_values.mean()), 2)
            }
        })
    
    zones.sort(key=lambda x: x['price_range']['min'])
    
    for i in range(len(zones) - 1):
        zones[i]['price_range']['max'] = zones[i + 1]['price_range']['min']
    
    optimal_zone_id = None
    for zone in zones:
        if zone['price_range']['min'] <= optimal_price <= zone['price_range']['max']:
            optimal_zone_id = zone['zone_id']
            break
    if optimal_zone_id is None:
        optimal_zone_id = zones[0]['zone_id']
    
    result = {
        'zones': zones,
        'optimal_price': {
            'price': round(float(optimal_price), 2),
            'probability_percent': round(float(optimal_prob * 100), 2),
            'normalized_probability_percent': round(float(optimal_normalized_prob * 100), 2),
            'expected_value': round(float(optimal_expected_value), 2),
            'zone_id': optimal_zone_id
        },
        'analysis': {
            'start_price': float(user_min_price),
            'max_probability_percent': round(float(max_prob * 100), 2),
            'max_probability_price': round(float(max_prob_price), 2),
            'scan_range': {
                'min': round(float(user_min_price), 2),
                'max': round(float(max_price), 2)
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    return result

def recommend_price(order_data, output_json=True, model_path="model_enhanced.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"⚠️ Модель не найдена: {model_path}")
    model = joblib.load(model_path)
    if hasattr(model, 'predict_proba'):
        pass
    else:
        raise TypeError(f"Загружен неправильный объект: {type(model)}")
    required_fields = [
        'order_timestamp', 'distance_in_meters', 'duration_in_seconds',
        'pickup_in_meters', 'pickup_in_seconds', 'price_start_local'
    ]
    for field in required_fields:
        if field not in order_data:
            raise ValueError(f"⚠️ Отсутствует обязательное поле: {field}")
    result = find_optimal_price(order_data, model, num_points=500)
    if output_json:
        return json.dumps(result, ensure_ascii=False, indent=2)
    return result

if __name__ == "__main__":
    if not os.path.exists("model_enhanced.joblib"):
        print("\n⚠️ Модель не найдена! Сначала запустите train_model.py")
        exit(1)
    test_order = {
        "order_timestamp": int(datetime.now().timestamp()),
        "distance_in_meters": 3000,
        "duration_in_seconds": 200,
        "pickup_in_meters": 2000,
        "pickup_in_seconds": 120,
        "driver_rating": 4.7,
        "platform": "android",
        "price_start_local": 180,
        "carname": "LADA",
        "carmodel": "GRANTA",
        "driver_reg_date": "2020-01-15"
    }
    result = recommend_price(test_order, output_json=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
