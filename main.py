import sys
import os
import json
from datetime import datetime

# Добавляем src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train_model import train_model
from recommend_price import recommend_price

def main():
    """Главная функция: обучение модели и тестовый запрос"""
    
    model_path = "model_enhanced.joblib"
    
    if not os.path.exists(model_path):
        print("⚠️ Модель не найдена. Начинаем обучение...")
        try:
            train_model(train_path="simple-train.csv", use_gpu=False)
        except Exception as e:
            print(f"⚠️ Ошибка при обучении: {e}")
            import traceback
            traceback.print_exc()
            print("\nПопробуйте запустить вручную: python src/train_model.py")
            return
    
    # Тестовый заказ
    order = {
        "order_timestamp": int(datetime.now().timestamp()),
        "distance_in_meters": 5000,
        "duration_in_seconds": 600,
        "pickup_in_meters": 1000,
        "pickup_in_seconds": 90,
        "driver_rating": 2.3,
        "platform": "android",
        "price_start_local": 300,
        "carname": "Toyota",
        "carmodel": "Camry",
        "driver_reg_date": "2020-01-15"
    }
    
    print("\n" + "="*70)
    print("ТЕСТОВЫЙ ЗАКАЗ")
    print("="*70)
    print(f"Дистанция: {order['distance_in_meters']/1000:.1f} км")
    print(f"Время: {order['duration_in_seconds']/60:.1f} мин")
    print(f"Начальная цена: {order['price_start_local']}₽")
    print(f"Автомобиль: {order['carname']} {order['carmodel']}")
    
    print("\nПоиск оптимальной цены...")
    try:
        result = recommend_price(order, output_json=False)
        
        print("\n" + "="*70)
        print("РЕКОМЕНДАЦИЯ")
        print("="*70)
        print(f"\n✅ Оптимальная цена: {result['optimal_price']}₽")
        print(f"   Вероятность принятия: {result['optimal_probability']*100:.1f}%")
        print(f"   Ожидаемый доход: {result['expected_revenue']:.2f}₽")
        print(f"   Наценка: {result['price_increase_pct']:+.1f}%")
        
        print("\nЦеновые зоны:")
        for zone in result['zones']:
            print(f"  {zone['name']:25s} [{zone['color']:12s}] "
                  f"{zone['price_from']:4d}-{zone['price_to']:4d}₽ "
                  f"(P={zone['avg_probability']*100:4.1f}%)")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
