import os
import shutil
from sklearn.model_selection import train_test_split

# --- НАСТРОЙКИ ---
SRC_IMG = 'source_data/images'
SRC_LBL = 'source_data/labels'
OUT_DIR = 'dataset'
RATIOS = (0.8, 0.1, 0.1)  # train, val, test

MAPPING = {
    "3.": 0, "4.": 1, "5.": 2, "6.": 3, "7.": 4, "1": 5, "2": 6,
    "3": 7, "4": 8, "5": 9, "6": 10, "7": 11, ".1": 12, ".2": 13,
    ".3": 14, "0": 15
}


def convert_labels(file_path, dest_path):
    if not os.path.exists(file_path): return
    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.split()
        if not parts: continue
        class_name = parts[0]
        if class_name in MAPPING:
            parts[0] = str(MAPPING[class_name])
            new_lines.append(" ".join(parts) + "\n")

    with open(dest_path, 'w') as f:
        f.writelines(new_lines)


# Получаем список файлов
files = [os.path.splitext(f)[0] for f in os.listdir(SRC_IMG) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

train, temp = train_test_split(files, train_size=RATIOS[0], random_state=42)
val, test = train_test_split(temp, train_size=RATIOS[1] / (RATIOS[1] + RATIOS[2]), random_state=42)


def process_set(file_list, split_name):
    img_dest_dir = os.path.join(OUT_DIR, 'images', split_name)
    lbl_dest_dir = os.path.join(OUT_DIR, 'labels', split_name)
    os.makedirs(img_dest_dir, exist_ok=True)
    os.makedirs(lbl_dest_dir, exist_ok=True)

    for name in file_list:
        for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
            img_path = os.path.join(SRC_IMG, name + ext)
            if os.path.exists(img_path):
                shutil.copy(img_path, os.path.join(img_dest_dir, name + ext))
                break
        convert_labels(os.path.join(SRC_LBL, name + '.txt'), os.path.join(lbl_dest_dir, name + '.txt'))


process_set(train, 'train')
process_set(val, 'val')
process_set(test, 'test')
print("✅ Данные готовы для YOLOv5")





