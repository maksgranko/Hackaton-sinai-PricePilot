import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src import recommend_price

def main():
    print("="*70)
    print("🚕 ТЕСТИРОВАНИЕ ML-СИСТЕМЫ РЕКОМЕНДАЦИИ ЦЕН (v2.0)")
    print("="*70)
    
    order = {
        "order_timestamp": int(datetime(2020, 5, 1, 14, 30, 0).timestamp()),
        "distance_in_meters": 3404,
        "duration_in_seconds": 486,
        "pickup_in_meters": 790,
        "pickup_in_seconds": 169,
        "driver_rating": 5,
        "platform": "android",
        "price_start_local": 180,
    }
    
    result_json = recommend_price(order, output_format='json')
    result = json.loads(result_json)
    


    print("\n" + "="*70)
    print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("="*70)
    
print("\n🔍 Анализируем заказ:")
    print(f"   Расстояние: {order['distance_in_meters']/1000:.2f} км")
    print(f"   Время в пути: {order['duration_in_seconds']/60:.1f} мин")
    print(f"   Стартовая цена: {order['price_start_local']} руб.")
    
    print("\n💰 ЗОНЫ ЦЕН (от дешёвой к дорогой):")
    print("-"*70)
    
    zone_icons = {
        1: "🔴",
        2: "🟡",
        3: "🟢",
        4: "🟠",
        5: "🔴"
    }
    
    for zone in result["zones"]:
        icon = zone_icons.get(zone["zone_id"], "⚪")
        print(f"\n{icon} ZONE {zone['zone_id']}")
        print(f"   Диапазон: {zone['price_range']['min']:.2f} - {zone['price_range']['max']:.2f} руб.")
        print(f"   Вероятность: {zone['metrics']['avg_probability_percent']:.2f}% "
              f"(normalized: {zone['metrics']['avg_normalized_probability_percent']:.2f}%)")
        print(f"   Ожидаемый доход: {zone['metrics']['avg_expected_value']:.2f} руб.")
    
    opt = result["optimal_price"]
    print("\n" + "="*70)
    print("🎯 ОПТИМАЛЬНАЯ ЦЕНА (максимум Expected Value)")
    print("="*70)
    print(f"💵 Цена: {opt['price']:.2f} руб.")
    print(f"📈 Вероятность принятия: {opt['probability_percent']:.2f}%")
    print(f"💰 Ожидаемый доход (EV): {opt['expected_value']:.2f} руб.")
    print(f"🏷️  Находится в зоне: ZONE {opt['zone_id']}")
    
    print("\n" + "="*70)
    print("📏 ВИЗУАЛИЗАЦИЯ ЗОН")
    print("="*70)
    print("\n    ┌─────────┬─────────┬─────────┬─────────┬─────────┐")
    print("    │ ZONE 1  │ ZONE 2  │ ZONE 3  │ ZONE 4  │ ZONE 5  │")
    print("    │  🔴RED  │ 🟡YELLOW│ 🟢GREEN │ 🟠ORANGE│  🔴RED  │")
    print("    └─────────┴─────────┴─────────┴─────────┴─────────┘")
    print("    Слишком    Приемлемо  ОПТИМУМ  Приемлемо  Слишком")
    print("     дешево     дешево              дорого     дорого")
    print(f"\n                         ▲ {opt['price']:.2f}₽")
    print("                  (оптимальная цена)")
    
    print("\n" + "="*70)
    print("📖 ИНТЕРПРЕТАЦИЯ ЗОН")
    print("="*70)
    print("\n✨ Зоны располагаются относительно оптимальной цены (максимум EV)")
    print("🔴 Zone 1: Слишком дешево - невыгодно водителю")
    print("🟡 Zone 2: Дешевле оптимума - высокая вероятность, меньше доход")
    print("🟢 Zone 3: Оптимальная зона - лучший баланс")
    print("🟠 Zone 4: Дороже оптимума - ниже вероятность, выше доход при успехе")
    print("🔴 Zone 5: Слишком дорого - клиент откажется")

if __name__ == "__main__":
    main()
