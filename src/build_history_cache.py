"""
Скрипт для предрасчета истории пользователей и водителей.
Создает кэш-файлы для быстрого доступа при inference.
"""

import pandas as pd
import joblib
import sys
from pathlib import Path

def calculate_user_history(df):
    """
    Рассчитывает статистику по каждому user_id.
    """
    print("[USER] Расчет истории пользователей...")
    
    user_stats = df.groupby('user_id').agg({
        'is_done': ['count', lambda x: (x == 'done').sum()],
        'price_bid_local': 'mean',
        'price_start_local': 'mean',
    }).reset_index()
    
    user_stats.columns = ['user_id', 'user_order_count', 'user_done_count', 
                          'user_avg_bid', 'user_avg_start_price']
    
    # Процент принятых заказов
    user_stats['user_acceptance_rate'] = user_stats['user_done_count'] / user_stats['user_order_count']
    
    # Средний коэффициент наценки
    user_stats['user_avg_price_ratio'] = user_stats['user_avg_bid'] / (user_stats['user_avg_start_price'] + 0.1)
    
    # Категориальные признаки
    user_stats['user_is_new'] = (user_stats['user_order_count'] <= 5).astype(float)
    user_stats['user_is_vip'] = (user_stats['user_order_count'] >= 20).astype(float)
    user_stats['user_is_price_sensitive'] = (user_stats['user_avg_price_ratio'] < 1.1).astype(float)
    
    print(f"   [OK] Обработано {len(user_stats)} пользователей")
    print(f"   [OK] Средний acceptance rate: {user_stats['user_acceptance_rate'].mean():.2%}")
    print(f"   [OK] VIP пользователей: {user_stats['user_is_vip'].sum():.0f}")
    
    return user_stats

def calculate_driver_history(df):
    """
    Рассчитывает статистику по каждому driver_id.
    """
    print("\n[DRIVER] Расчет истории водителей...")
    
    driver_stats = df.groupby('driver_id').agg({
        'is_done': ['count', lambda x: (x == 'done').sum()],
        'price_bid_local': 'mean',
        'price_start_local': 'mean',
    }).reset_index()
    
    driver_stats.columns = ['driver_id', 'driver_bid_count', 'driver_done_count',
                            'driver_avg_bid', 'driver_avg_start_price']
    
    # Процент принятых ставок
    driver_stats['driver_acceptance_rate'] = driver_stats['driver_done_count'] / driver_stats['driver_bid_count']
    
    # Средний коэффициент наценки водителя
    driver_stats['driver_avg_bid_ratio'] = driver_stats['driver_avg_bid'] / (driver_stats['driver_avg_start_price'] + 0.1)
    
    # Категориальные признаки
    driver_stats['driver_is_active'] = (driver_stats['driver_bid_count'] >= 20).astype(float)
    driver_stats['driver_is_aggressive'] = (driver_stats['driver_avg_bid_ratio'] > 1.2).astype(float)
    driver_stats['driver_is_flexible'] = (driver_stats['driver_avg_bid_ratio'] < 1.1).astype(float)
    
    print(f"   [OK] Обработано {len(driver_stats)} водителей")
    print(f"   [OK] Средний acceptance rate: {driver_stats['driver_acceptance_rate'].mean():.2%}")
    print(f"   [OK] Активных водителей: {driver_stats['driver_is_active'].sum():.0f}")
    
    return driver_stats

def main(csv_path='simple-train.csv'):
    """
    Основная функция для построения кэша истории.
    """
    print("="*70)
    print("ПОСТРОЕНИЕ КЭША ИСТОРИИ ПОЛЬЗОВАТЕЛЕЙ И ВОДИТЕЛЕЙ")
    print("="*70)
    
    # Загрузка данных
    print(f"\n[DATA] Загрузка данных из {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        print(f"   [OK] Загружено {len(df)} записей")
    except FileNotFoundError:
        print(f"   [WARN] Файл {csv_path} не найден, пробуем сокращенную версию...")
        csv_path = 'simple-train shorted.csv'
        df = pd.read_csv(csv_path)
        print(f"   [OK] Загружено {len(df)} записей из {csv_path}")
    
    # Расчет статистик
    user_history = calculate_user_history(df)
    driver_history = calculate_driver_history(df)
    
    # Сохранение
    print("\n[SAVE] Сохранение кэша...")
    user_path = 'user_history.joblib'
    driver_path = 'driver_history.joblib'
    
    joblib.dump(user_history, user_path)
    joblib.dump(driver_history, driver_path)
    
    print(f"   [OK] Сохранено: {user_path} ({user_history.memory_usage(deep=True).sum() / 1024:.1f} KB)")
    print(f"   [OK] Сохранено: {driver_path} ({driver_history.memory_usage(deep=True).sum() / 1024:.1f} KB)")
    
    # Статистика
    print("\n" + "="*70)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("="*70)
    print(f"Пользователей в кэше:  {len(user_history)}")
    print(f"Водителей в кэше:      {len(driver_history)}")
    print(f"\nСредние значения (fallback для новых):")
    print(f"  user_acceptance_rate:   {user_history['user_acceptance_rate'].mean():.3f}")
    print(f"  user_avg_price_ratio:   {user_history['user_avg_price_ratio'].mean():.3f}")
    print(f"  driver_acceptance_rate: {driver_history['driver_acceptance_rate'].mean():.3f}")
    print(f"  driver_avg_bid_ratio:   {driver_history['driver_avg_bid_ratio'].mean():.3f}")
    print("\n[SUCCESS] Кэш истории успешно построен!")
    print("="*70)

if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'simple-train.csv'
    main(csv_path)
else:
    # При импорте как модуль - загружаем из дефолтного пути
    pass

