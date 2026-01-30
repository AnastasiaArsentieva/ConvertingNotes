import os
from ultralytics import YOLO
from tkinter import filedialog, Tk
from PIL import Image
import numpy as np

# 1. ВАШ СЛОВАРЬ (Маппинг ID класса в название ноты)
PIANO_TO_GLUCO = {
    0: "3.", 1: "4.", 2: "5.", 3: "6.", 4: "7.", 5: "1", 6: "2",
    7: "3", 8: "4", 9: "5", 10: "6", 11: "7", 12: ".1", 13: ".2",
    14: ".3", 15: "0"
}


def process_and_save():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    img_path = filedialog.askopenfilename(title="Выберите лист с нотами")
    if not img_path: return

    model_path = 'runs/detect/my_yolo_project/small_objects_run/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"❌ Ошибка: Модель не найдена!")
        return

    # Загружаем модель
    model = YOLO(model_path)

    # 2. Распознавание
    results = model.predict(source=img_path, imgsz=1280, conf=0.25)
    result = results[0]  # Берем первый результат из списка

    # 3. Сбор данных и подмена имен для текстового файла
    detections = []
    for box in result.boxes:
        # Получаем координаты как список чисел
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # Берем имя ноты из нашего словаря
        note_name = PIANO_TO_GLUCO.get(cls_id, f"ID_{cls_id}")

        detections.append({
            'x': x1,
            'y': y1,
            'name': note_name,
            'h': y2 - y1,
            'conf': conf
        })

    if not detections:
        print("Объекты не найдены.")
        return

    # 4. СОРТИРОВКА ПО СТРОКАМ И ГОРИЗОНТАЛИ
    detections.sort(key=lambda d: d['y'])
    rows = []
    current_row = [detections[0]]

    for i in range(1, len(detections)):
        # Группировка в строки (допуск 70% высоты)
        if abs(detections[i]['y'] - current_row[-1]['y']) < (current_row[-1]['h'] * 0.7):
            current_row.append(detections[i])
        else:
            current_row.sort(key=lambda d: d['x'])
            rows.append(current_row)
            current_row = [detections[i]]

    current_row.sort(key=lambda d: d['x'])
    rows.append(current_row)

    # 5. ЗАПИСЬ В ФАЙЛ
    output_file = "notes_sequence.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            line = " -> ".join([f"{d['name']} ({int(d['conf'] * 100)}%)" for d in row])
            f.write(f"Строка {i + 1}: {line}\n")

    print(f"✅ Результаты записаны в '{output_file}'")

    # 6. ВЫВОД КАРТИНКИ С ПРАВИЛЬНЫМИ ИМЕНАМИ
    # Чтобы на картинке были ноты, а не ID, мы временно подменяем словарь имен в объекте результата
    result.names = PIANO_TO_GLUCO

    res_bgr = result.plot()  # Отрисовка
    res_rgb = Image.fromarray(res_bgr[..., ::-1])
    res_rgb.show()


if __name__ == "__main__":
    process_and_save()





