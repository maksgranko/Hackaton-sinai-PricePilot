import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src import train_model, recommend_price

def main():
    print("="*70)
    print("🚕 ML-СИСТЕМА РЕКОМЕНДАЦИИ ЦЕН ДЛЯ ТАКСИ (v2.0)")
    print("="*70)
    
    model_path = "model_enhanced.joblib"
    if not os.path.exists(model_path):
        print("\n⚠️  Модель не найдена. Начинаем обучение...")
        print("="*70)
        train_model(train_path="simple-train.csv")
    else:
        print(f"\n✅ Модель {model_path} найдена. Пропускаем обучение.")
    
    order = {
        "order_timestamp": int(datetime.now().timestamp()),
        "distance_in_meters": 3404,
        "duration_in_seconds": 486,
        "pickup_in_meters": 790,
        "pickup_in_seconds": 169,
        "driver_rating": 5,
        "platform": "android",
        "price_start_local": 180,
    }
    
    result = recommend_price(order, output_format='json')
    print(result)

if __name__ == "__main__":
    main()
