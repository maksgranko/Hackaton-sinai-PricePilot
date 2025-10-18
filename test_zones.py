"""Тестирование новой логики зон на разных сценариях"""
import json
from datetime import datetime
from src.recommend_price import recommend_price

# Тест 1: Короткая поездка
print("="*70)
print("ТЕСТ 1: Короткая поездка (1.5 км, утро)")
print("="*70)
test1 = {
    "order_timestamp": int(datetime(2025, 10, 18, 8, 30).timestamp()),
    "distance_in_meters": 1500,
    "duration_in_seconds": 180,
    "pickup_in_meters": 800,
    "pickup_in_seconds": 90,
    "driver_rating": 4.9,
    "platform": "android",
    "price_start_local": 150,
    "carname": "LADA",
    "carmodel": "GRANTA",
    "driver_reg_date": "2020-01-15"
}
result1 = recommend_price(test1, output_json=False)
print(f"\n📍 Оптимальная цена: {result1['optimal_price']['price']}₽")
print(f"   Вероятность принятия: {result1['optimal_price']['probability_percent']}%")
print(f"   Зона: {result1['optimal_price']['zone_id']}")
print("\n📊 Зоны:")
for zone in result1['zones']:
    print(f"   {zone['zone_name']:25s} {zone['price_range']['min']:6.1f}-{zone['price_range']['max']:6.1f}₽  "
          f"(вероятность: {zone['metrics']['avg_probability_percent']:5.1f}%)")

# Тест 2: Длинная поездка
print("\n" + "="*70)
print("ТЕСТ 2: Длинная поездка (15 км, вечер)")
print("="*70)
test2 = {
    "order_timestamp": int(datetime(2025, 10, 18, 19, 0).timestamp()),
    "distance_in_meters": 15000,
    "duration_in_seconds": 1200,
    "pickup_in_meters": 2000,
    "pickup_in_seconds": 180,
    "driver_rating": 5.0,
    "platform": "ios",
    "price_start_local": 400,
    "carname": "Toyota",
    "carmodel": "Camry",
    "driver_reg_date": "2018-05-20"
}
result2 = recommend_price(test2, output_json=False)
print(f"\n📍 Оптимальная цена: {result2['optimal_price']['price']}₽")
print(f"   Вероятность принятия: {result2['optimal_price']['probability_percent']}%")
print(f"   Зона: {result2['optimal_price']['zone_id']}")
print("\n📊 Зоны:")
for zone in result2['zones']:
    print(f"   {zone['zone_name']:25s} {zone['price_range']['min']:6.1f}-{zone['price_range']['max']:6.1f}₽  "
          f"(вероятность: {zone['metrics']['avg_probability_percent']:5.1f}%)")

# Тест 3: Средняя поездка ночью
print("\n" + "="*70)
print("ТЕСТ 3: Средняя поездка (5 км, ночь)")
print("="*70)
test3 = {
    "order_timestamp": int(datetime(2025, 10, 18, 23, 30).timestamp()),
    "distance_in_meters": 5000,
    "duration_in_seconds": 600,
    "pickup_in_meters": 1500,
    "pickup_in_seconds": 150,
    "driver_rating": 4.7,
    "platform": "android",
    "price_start_local": 250,
    "carname": "Renault",
    "carmodel": "Logan",
    "driver_reg_date": "2021-03-10"
}
result3 = recommend_price(test3, output_json=False)
print(f"\n📍 Оптимальная цена: {result3['optimal_price']['price']}₽")
print(f"   Вероятность принятия: {result3['optimal_price']['probability_percent']}%")
print(f"   Зона: {result3['optimal_price']['zone_id']}")
print("\n📊 Зоны:")
for zone in result3['zones']:
    print(f"   {zone['zone_name']:25s} {zone['price_range']['min']:6.1f}-{zone['price_range']['max']:6.1f}₽  "
          f"(вероятность: {zone['metrics']['avg_probability_percent']:5.1f}%)")

print("\n" + "="*70)
print("✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
print("="*70)

