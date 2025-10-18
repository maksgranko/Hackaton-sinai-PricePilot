import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train_model import train_model
from recommend_price import recommend_price

def main():
    model_path = "model_enhanced.joblib"
    
    if not os.path.exists(model_path):
        try:
            train_model(train_path="simple-train.csv", use_gpu=False)
        except Exception as e:
            print(f"⚠️ Ошибка при обучении: {e}")
            import traceback
            traceback.print_exc()
            return
    
    order = {
        "order_timestamp": int(datetime.now().timestamp()),
        "distance_in_meters": 12000,
        "duration_in_seconds": 1600,
        "pickup_in_meters": 2000,
        "pickup_in_seconds": 120,
        "driver_rating": 4.8,
        "platform": "android",
        "price_start_local": 180,
        "carname": "LADA",
        "carmodel": "GRANTA",
        "driver_reg_date": "2020-01-15"
    }
    
    try:
        result = recommend_price(order, output_json=True)
        print(result)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
