import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os
import threading

# --- ВАШ СЛОВАРЬ НОТ ---
PIANO_TO_GLUCO = {
    0: "3.", 1: "4.", 2: "5.", 3: "6.", 4: "7.", 5: "1", 6: "2",
    7: "3", 8: "4", 9: "5", 10: "6", 11: "7", 12: ".1", 13: ".2",
    14: ".3", 15: "0"
}


class NoteScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Piano Note Scanner v1.0")
        self.root.geometry("1000x900")
        self.root.configure(bg="#2c3e50")

        # --- ЗАГРУЗКА МОДЕЛИ ---
        self.model_path = 'runs/detect/my_yolo_project/small_objects_run/weights/best.pt'
        self.model = None
        self._load_model()

        # --- ИНТЕРФЕЙС ---
        self.header = tk.Label(root, text="Сканер и Конвертер Нот", font=("Arial", 20, "bold"), bg="#2c3e50",
                               fg="#ecf0f1")
        self.header.pack(pady=15)

        self.btn_frame = tk.Frame(root, bg="#2c3e50")
        self.btn_frame.pack(pady=10)

        self.btn_load = tk.Button(self.btn_frame, text="📁 Загрузить картинку", command=self.open_file,
                                  font=("Arial", 12, "bold"), bg="#3498db", fg="white", padx=20, pady=10)
        self.btn_load.grid(row=0, column=0, padx=10)

        self.btn_save = tk.Button(self.btn_frame, text="💾 Сохранить результат", command=self.save_to_file,
                                  font=("Arial", 12, "bold"), bg="#27ae60", fg="white", padx=20, pady=10,
                                  state=tk.DISABLED)
        self.btn_save.grid(row=0, column=1, padx=10)

        self.canvas = tk.Canvas(root, width=800, height=450, bg="#34495e", highlightthickness=0)
        self.canvas.pack(pady=10)

        self.text_area = tk.Text(root, height=10, width=90, font=("Courier New", 12), bg="#ecf0f1", fg="#2c3e50",
                                 padx=10, pady=10)
        self.text_area.pack(pady=15)
        self.text_area.insert(tk.END, "Результат распознавания появится здесь...")

        self.current_result_text = ""

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
            # ВАЖНО: Мы не меняем self.model.names здесь, это вызовет ошибку
        else:
            messagebox.showerror("Ошибка", f"Модель не найдена!\n{self.model_path}")

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Изображения", "*.jpg *.jpeg *.png *.webp")])
        if not file_path:
            return

        self.btn_load.config(state=tk.DISABLED)
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, "Обработка... Пожалуйста, подождите.")

        threading.Thread(target=self.process_image, args=(file_path,), daemon=True).start()

    def process_image(self, img_path):
        try:
            # 1. Детекция
            results = self.model.predict(source=img_path, imgsz=1280, conf=0.25)
            result = results[0]  # Берем первый результат из списка

            # 2. ПОДМЕНЯЕМ ИМЕНА в объекте результата для корректной отрисовки рамками
            result.names = PIANO_TO_GLUCO

            # 3. Сбор данных для сортировки
            detections = []
            for box in result.boxes:
                coords = box.xyxy.tolist()[0]  # Исправлено получение координат
                x1, y1, x2, y2 = coords
                cls_id = int(box.cls)
                conf = float(box.conf)
                note_name = PIANO_TO_GLUCO.get(cls_id, f"ID_{cls_id}")
                detections.append({'x': x1, 'y': y1, 'name': note_name, 'h': y2 - y1, 'conf': conf})

            if not detections:
                self.root.after(0, lambda: self.show_no_objects())
                return

            # 4. Сортировка по строкам
            detections.sort(key=lambda d: d['y'])
            rows = []
            if detections:
                curr_row = [detections[0]]
                for i in range(1, len(detections)):
                    if abs(detections[i]['y'] - curr_row[-1]['y']) < (curr_row[-1]['h'] * 0.7):
                        curr_row.append(detections[i])
                    else:
                        curr_row.sort(key=lambda d: d['x'])
                        rows.append(curr_row)
                        curr_row = [detections[i]]
                curr_row.sort(key=lambda d: d['x'])
                rows.append(curr_row)

            # 5. Формирование текста
            output_lines = []
            for i, row in enumerate(rows):
                line = " -> ".join([f"{d['name']} ({int(d['conf'] * 100)}%)" for d in row])
                output_lines.append(f"Строка {i + 1}: {line}")

            self.current_result_text = "\n".join(output_lines)

            # 6. Отрисовка картинки (plot теперь использует наши имена из result.names)
            res_bgr = result.plot()
            res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(res_rgb)

            self.root.after(0, self.update_ui, img_pil)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Детали: {str(e)}"))
            self.root.after(0, lambda: self.btn_load.config(state=tk.NORMAL))

    def update_ui(self, img_pil):
        img_pil.thumbnail((800, 450))
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(400, 225, image=self.tk_img)

        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, self.current_result_text)

        self.btn_load.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)

    def show_no_objects(self):
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, "Объекты не найдены.")
        self.btn_load.config(state=tk.NORMAL)

    def save_to_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt")],
                                                 initialfile="converted_notes.txt")
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.current_result_text)
                messagebox.showinfo("Успех", "Файл успешно сохранен!")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = NoteScannerApp(root)
    root.mainloop()

