import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.train_model import train_model
from src.recommend_price import recommend_price


def main():
    """Главная функция: обучение модели и тестовый запрос"""
    
    model_path = "model_enhanced.joblib"
    if not os.path.exists(model_path):
        print("⚠️  Модель не найдена. Начинаем обучение...")
        try:
            train_model(train_path="simple-train.csv", use_gpu=True)
        except Exception as e:
            print(f"⚠️  GPU недоступен, используем CPU...")
            train_model(train_path="simple-train.csv", use_gpu=False)
    
    # Единственный тестовый заказ (параметры можно изменить)
    order = {
        "order_timestamp": int(datetime.now().timestamp()),
        "distance_in_meters": 3404,
        "duration_in_seconds": 486,
        "pickup_in_meters": 790,
        "pickup_in_seconds": 169,
        "driver_rating": 5,
        "platform": "android",
        "price_start_local": 180,
        "carname": "Toyota",
        "carmodel": "Camry"
    }
    
    result = recommend_price(order, output_format='dict')
    
    # Сохраняем результат
    output_file = "recommendation_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Результат сохранён в {output_file}")


if __name__ == "__main__":
    main()
