import sys
import os
import json
from datetime import datetime

# Добавляем путь к модулям
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.train_model import train_model
from src.recommend_price import recommend_price


def main():
    """Главная функция: обучение модели и тестовый запрос"""
    
    print("="*80)
    print("🚕 ML-СИСТЕМА РЕКОМЕНДАЦИИ ЦЕН ДЛЯ ТАКСИ (v3.0)")
    print("="*80)
    print("✨ Новое:")
    print("   • XGBoost (быстрее в 3x)")
    print("   • Классификация типов такси (эконом/комфорт/бизнес)")
    print("   • 54 признака (включая временные взаимодействия)")
    print("   • Все 5 ценовых зон")
    print("="*80)
    
    # Проверяем наличие модели
    model_path = "model_enhanced.joblib"
    if not os.path.exists(model_path):
        print("\n⚠️  Модель не найдена. Начинаем обучение...")
        print("="*80)
        try:
            print("🎮 Попытка использовать GPU...")
            train_model(train_path="simple-train.csv", use_gpu=True)
        except Exception as e:
            print(f"\n⚠️  GPU недоступен: {str(e)[:100]}")
            print("💻 Переключаемся на CPU...")
            train_model(train_path="simple-train.csv", use_gpu=False)
        print("\n" + "="*80)
    else:
        print(f"\n✅ Модель найдена: {model_path}")
        print("   Пропускаем обучение.")
        print("="*80)
    
    # Тестовый запрос
    print("\n📋 ТЕСТОВЫЙ ЗАПРОС:")
    print("="*80)
    
    # Пример с автоопределением типа такси
    order = {
        "order_timestamp": int(datetime.now().timestamp()),
        "distance_in_meters": 404,
        "duration_in_seconds": 15,
        "pickup_in_meters": 790,
        "pickup_in_seconds": 169,
        "driver_rating": 5,
        "platform": "android",
        "price_start_local": 180,
        "carname": "Toyota",
        "carmodel": "Camry"
    }
    
    print("\n2️⃣  Заказ с автоопределением типа (Toyota Camry):")
    print(f"   Марка: {order['carname']}")
    print(f"   Модель: {order['carmodel']}")
    
    result = recommend_price(order, output_format='dict')
    print(f"\n   🎯 Тип такси: {result['taxi_type']} (автоопределён)")
    print(f"   💰 Оптимальная цена: {result['optimal_price']['price']} руб")
    print(f"   📊 Вероятность: {result['optimal_price']['probability_percent']}%")
    print(f"   📈 Ожидаемый доход: {result['optimal_price']['expected_value']} руб")
    
    # Сохраняем результат в JSON
    print("\n" + "="*80)
    print("💾 СОХРАНЕНИЕ РЕЗУЛЬТАТА")
    print("="*80)
    
    output_file = "recommendation_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Результат сохранён в {output_file}")
    print("\n" + "="*80)
    print("🎉 ГОТОВО!")
    print("="*80)


if __name__ == "__main__":
    main()
