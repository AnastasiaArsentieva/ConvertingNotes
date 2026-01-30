import os
import cv2
from pathlib import Path
from ultralytics import YOLO  # Если используете современный интерфейс YOLOv5/v8


# Для классического YOLOv5 обычно используется вызов train.py через терминал,
# но ниже представлен универсальный код на Python

def check_dataset(img_dir):
    """Проверка на битые изображения"""
    print(f"--- Проверка файлов в {img_dir} ---")
    valid_extensions = ('.jpg', '.jpeg', '.png')
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]

    corrupt_files = 0
    for f in files:
        path = os.path.join(img_dir, f)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Битый файл удален: {path}")
            os.remove(path)
            # Также удаляем файл разметки, если он есть
            lbl_path = path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
            corrupt_files += 1

    print(f"Проверка завершена. Удалено битых файлов: {corrupt_files}")


# 1. Проверяем папки перед обучением
for split in ['train', 'val']:
    check_dataset(f'dataset/images/{split}')

# 2. Запуск обучения
# Мы используем YOLOv5nu (nano-version) для скорости или yolov5s.pt
model = YOLO('yolov5s.pt')

results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=1280,  # Если объекты ОЧЕНЬ мелкие, можно поднять до 1280
    batch=16,  # Уменьшите, если не хватает памяти GPU
    device='cpu',  # 0 для GPU, 'cpu' для процессора
    workers=14,         # Используем 14 ядер для подготовки данных, елси на cpu
    mosaic=1.0,  # Склеивает 4 фото в одно (отлично для мелких объектов)
    mixup=0.1,  # Помогает при множественной разметке
    patience=20,  # Ранняя остановка, если метрики не растут
    project='my_yolo_project',
    name='small_objects_run'
)


# 3. Расчет метрик на тестовой выборке
print("--- Финальная валидация ---")
metrics = model.val()
print(f"mAP@50: {metrics.box.map50}")
print(f"mAP@50-95: {metrics.box.map}")
