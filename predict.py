"""
Скрипт для генерации предсказаний на тестовых данных.
Создаёт файл PREDICT.csv с колонкой is_done (done/cancel).
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train_model import build_enhanced_features, train_model

def predict_test_data(
    test_path="test.csv",
    model_path="model_enhanced.joblib",
    output_path="PREDICT.csv",
    threshold=0.5,
    train_if_missing=True
):
    """
    Создаёт предсказания для тестовых данных.
    
    Args:
        test_path: путь к тестовому файлу
        model_path: путь к обученной модели
        output_path: путь для сохранения предсказаний
        threshold: порог для классификации (done если proba >= threshold)
        train_if_missing: обучить модель, если её нет
    
    Returns:
        DataFrame с предсказаниями
    """
    print("\n" + "="*70)
    print("ГЕНЕРАЦИЯ ПРЕДСКАЗАНИЙ ДЛЯ ТЕСТОВЫХ ДАННЫХ")
    print("="*70)
    
    # ============================================================
    # 1. Проверка наличия модели
    # ============================================================
    if not os.path.exists(model_path):
        print(f"\n⚠️  Модель не найдена: {model_path}")
        
        if train_if_missing:
            print("📚 Обучаем модель на train данных...")
            try:
                model, _ = train_model(
                    train_path="simple-train.csv",
                    soft_cleaning=True,
                    test_size=0.2,
                    random_state=42
                )
                print(f"✅ Модель обучена и сохранена: {model_path}")
            except FileNotFoundError:
                print("❌ ОШИБКА: Файл simple-train.csv не найден!")
                print("   Сначала обучите модель: python src/train_model.py")
                return None
            except Exception as e:
                print(f"❌ ОШИБКА при обучении: {e}")
                return None
        else:
            print("❌ Сначала обучите модель: python src/train_model.py")
            return None
    
    # ============================================================
    # 2. Загрузка модели
    # ============================================================
    print(f"\n📦 Загрузка модели из {model_path}...")
    try:
        model = joblib.load(model_path)
        print("✅ Модель загружена успешно")
    except Exception as e:
        print(f"❌ ОШИБКА при загрузке модели: {e}")
        return None
    
    # ============================================================
    # 3. Загрузка тестовых данных
    # ============================================================
    print(f"\n📁 Загрузка тестовых данных из {test_path}...")
    try:
        df_test = pd.read_csv(test_path)
        print(f"✅ Загружено {len(df_test)} записей")
        print(f"   Колонок: {len(df_test.columns)}")
    except FileNotFoundError:
        print(f"❌ ОШИБКА: Файл {test_path} не найден!")
        return None
    except Exception as e:
        print(f"❌ ОШИБКА при загрузке: {e}")
        return None
    
    # Добавляем фейковую колонку is_done для тестовых данных (нужна для расчёта истории)
    # Она не влияет на предсказание, только на признаки истории
    if 'is_done' not in df_test.columns:
        df_test['is_done'] = 'cancel'  # Временное значение
        print("   ℹ️  Добавлена временная колонка 'is_done' для расчёта признаков")
    
    # Проверяем обязательные колонки
    required_columns = [
        'order_timestamp', 'tender_timestamp', 'driver_reg_date',
        'distance_in_meters', 'duration_in_seconds', 'price_bid_local',
        'price_start_local', 'pickup_in_meters', 'pickup_in_seconds',
        'driver_rating', 'carname', 'carmodel', 'platform',
        'user_id', 'driver_id'
    ]
    
    missing_cols = [col for col in required_columns if col not in df_test.columns]
    if missing_cols:
        print(f"⚠️  ВНИМАНИЕ: Отсутствуют колонки: {missing_cols}")
        print("   Предсказания могут быть неточными!")
    
    # ============================================================
    # 4. Создание признаков
    # ============================================================
    print("\n🔧 Создание признаков для тестовых данных...")
    try:
        X_test = build_enhanced_features(df_test)
        print(f"✅ Создано {X_test.shape[1]} признаков для {X_test.shape[0]} записей")
        
        # Статистика по признакам качества
        if 'data_quality_index' in X_test.columns:
            avg_quality = X_test['data_quality_index'].mean()
            high_quality = (X_test['is_high_quality'] == 1).sum()
            suspicious = (X_test['is_suspicious'] == 1).sum()
            low_quality = (X_test['is_low_quality'] == 1).sum()
            
            print(f"\n📊 Качество тестовых данных:")
            print(f"   Средний индекс качества: {avg_quality:.3f}")
            print(f"   Высокое качество: {high_quality} ({high_quality/len(X_test)*100:.1f}%)")
            print(f"   Подозрительные:   {suspicious} ({suspicious/len(X_test)*100:.1f}%)")
            print(f"   Низкое качество:  {low_quality} ({low_quality/len(X_test)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ ОШИБКА при создании признаков: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ============================================================
    # 5. Генерация предсказаний
    # ============================================================
    print(f"\n🎯 Генерация предсказаний (порог: {threshold})...")
    try:
        # Получаем вероятности
        probabilities = model.predict_proba(X_test)[:, 1]
        
        # Конвертируем в done/cancel
        # ============================================================
        # 🎯 ВАЖНО: Используем data_quality_index для финального решения
        # ============================================================
        # Модель обучена на "примет ли водитель заказ", но нам нужно "качественные ли данные"
        # Поэтому финальное решение принимаем на основе индекса качества данных
        
        if 'data_quality_index' in X_test.columns:
            print(f"\n🎯 Используем data_quality_index для определения качества данных")
            quality_index = X_test['data_quality_index'].values
            
            # Правило: quality_index >= 0.5 → DONE (качественные данные)
            predictions = np.where(quality_index >= 0.5, 'done', 'cancel')
            
            print(f"   Статистика quality_index:")
            print(f"      Минимум:  {quality_index.min():.4f}")
            print(f"      Среднее:  {quality_index.mean():.4f}")
            print(f"      Максимум: {quality_index.max():.4f}")
        else:
            print(f"\n⚠️  data_quality_index не найден, используем вероятности модели")
            predictions = np.where(probabilities >= threshold, 'done', 'cancel')
        
        # Статистика
        n_done = (predictions == 'done').sum()
        n_cancel = (predictions == 'cancel').sum()
        
        print(f"\n✅ Предсказания созданы:")
        print(f"   done:   {n_done:5d} ({n_done/len(predictions)*100:5.1f}%)")
        print(f"   cancel: {n_cancel:5d} ({n_cancel/len(predictions)*100:5.1f}%)")
        
        # Распределение вероятностей
        print(f"\n📈 Распределение вероятностей:")
        print(f"   Минимум:  {probabilities.min():.3f}")
        print(f"   Q1:       {np.percentile(probabilities, 25):.3f}")
        print(f"   Медиана:  {np.median(probabilities):.3f}")
        print(f"   Q3:       {np.percentile(probabilities, 75):.3f}")
        print(f"   Максимум: {probabilities.max():.3f}")
        print(f"   Среднее:  {probabilities.mean():.3f}")
        
    except Exception as e:
        print(f"❌ ОШИБКА при генерации предсказаний: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ============================================================
    # 6. Сохранение результатов
    # ============================================================
    print(f"\n💾 Сохранение результатов в {output_path}...")
    try:
        # Создаём DataFrame с одной колонкой is_done
        result = pd.DataFrame({
            'is_done': predictions
        })
        
        # Сохраняем
        result.to_csv(output_path, index=False)
        print(f"✅ Файл сохранён: {output_path}")
        print(f"   Формат: {len(result)} строк × 1 колонка")
        
        # Проверяем результат
        print(f"\n🔍 Проверка сохранённого файла:")
        saved = pd.read_csv(output_path)
        print(f"   Колонки: {list(saved.columns)}")
        print(f"   Строк: {len(saved)}")
        print(f"   Уникальные значения: {saved['is_done'].unique()}")
        
        # Дополнительно сохраняем версию с вероятностями для анализа
        detailed_output = output_path.replace('.csv', '_detailed.csv')
        result_detailed = pd.DataFrame({
            'is_done': predictions,
            'probability': probabilities,
            'data_quality_index': X_test['data_quality_index'].values if 'data_quality_index' in X_test.columns else np.nan
        })
        result_detailed.to_csv(detailed_output, index=False)
        print(f"📊 Детальный файл (для анализа): {detailed_output}")
        
    except Exception as e:
        print(f"❌ ОШИБКА при сохранении: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ============================================================
    # 7. Итоговая информация
    # ============================================================
    print("\n" + "="*70)
    print("✅ ПРЕДСКАЗАНИЯ УСПЕШНО СОЗДАНЫ!")
    print("="*70)
    print(f"\n📄 Итоговый файл: {output_path}")
    print(f"   Формат: CSV с колонкой 'is_done'")
    print(f"   Значения: 'done' (качественный) или 'cancel' (не качественный)")
    print(f"   Записей: {len(result)}")
    print(f"\n💡 Для анализа смотрите: {detailed_output}")
    print(f"   (содержит вероятности и индекс качества данных)")
    
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Генерация предсказаний для тестовых данных"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="test.csv",
        help="Путь к тестовому файлу"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model_enhanced.joblib",
        help="Путь к обученной модели"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="PREDICT.csv",
        help="Путь для сохранения предсказаний"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Порог для классификации (0.0-1.0)"
    )
    parser.add_argument(
        "--no_train",
        action="store_true",
        help="Не обучать модель, если она отсутствует"
    )
    
    args = parser.parse_args()
    
    try:
        result = predict_test_data(
            test_path=args.test_path,
            model_path=args.model_path,
            output_path=args.output_path,
            threshold=args.threshold,
            train_if_missing=not args.no_train
        )
        
        if result is not None:
            print("\n🎉 Готово! Файл PREDICT.csv создан успешно!")
        else:
            print("\n❌ Не удалось создать предсказания")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

