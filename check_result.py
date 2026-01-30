import os
from ultralytics import YOLO


def predict_and_save():
    # 1. Путь к весам (используем best.pt, если обучение кончилось, или last.pt, если еще идет)
    weights_path = 'runs/detect/my_yolo_project/small_objects_run/weights/best.pt'

    # Если обучение еще не завершено, возьмем промежуточный файл last.pt
    if not os.path.exists(weights_path):
        weights_path = 'runs/detect/my_yolo_project/small_objects_run/weights/last.pt'

    if not os.path.exists(weights_path):
        print("❌ Файл весов не найден! Подождите еще пару эпох.")
        return

    # 2. Загружаем модель
    model = YOLO(weights_path)

    # 3. Запускаем предсказание
    # source: путь к папке с картинками, которые хотим проверить
    # imgsz: ставим 1280, как при обучении
    # conf: порог уверенности (0.25 значит показывать объекты, в которых модель уверена на 25%+)
    results = model.predict(
        source='dataset/images/val',
        imgsz=1280,
        conf=0.25, #conf=0.1,  # Снижаем порог, чтобы увидеть "черновики" предсказаний
        save=True,  # Сохранить картинки с рамками
        project='val_results',
        name='check_epoch'
    )

    print(f"✅ Готово! Результаты сохранены в папку: val_results/check_epoch")


if __name__ == '__main__':
    predict_and_save()



