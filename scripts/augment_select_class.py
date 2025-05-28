import os
import cv2
import shutil
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import defaultdict
import albumentations as A

# --------------------------------
# 🔧 CONFIG
# --------------------------------
TARGET_CLASS_ID = 3  # motorcycle
CLASS_NAMES = {0: "mycar", 1: "person", 2: "car", 3: "motorcycle"}
OUTPUT_DIR = "D:/GhostTrack_YOLOv12n_Train/datasets"
AUG_IMG_DIR = os.path.join(OUTPUT_DIR, "images/train")
AUG_LABEL_DIR = os.path.join(OUTPUT_DIR, "labels/train")

# 💥 Augmented output
AUG_TIMES = 6
MAX_AUG = 50  # จำนวน motorcycle ที่ต้องการเพิ่ม

IGNORE_FILES = ['classes.txt']  # ไฟล์ที่ต้องการข้าม

# ตรวจสอบว่าเป็นไฟล์ที่ต้องการข้ามหรือไม่
def is_valid_label_file(file_name):
    return file_name not in IGNORE_FILES and file_name.endswith('.txt')

# --------------------------------
# 📦 รวมไฟล์จาก train เดิม (ถ้ามี)
# --------------------------------
combined_img_dir = os.path.join(OUTPUT_DIR, 'images_combined')
os.makedirs(combined_img_dir, exist_ok=True)

for split in ['train']:
    split_path = os.path.join(OUTPUT_DIR, f'images/{split}')
    if os.path.exists(split_path):
        for f in os.listdir(split_path):
            full_src = os.path.join(split_path, f)
            full_dst = os.path.join(combined_img_dir, f)
            if not os.path.exists(full_dst):
                shutil.copy(full_src, full_dst)

IMG_DIR = combined_img_dir
image_files = [f for f in os.listdir(IMG_DIR) if f.endswith((".jpg", ".png"))]

if len(image_files) == 0:
    raise ValueError(f"❌ ไม่พบไฟล์ภาพใน {IMG_DIR}")

# --------------------------------
# ✂️ SPLIT TRAIN/VAL (90/10)
# --------------------------------
train_files, val_files = train_test_split(image_files, test_size=0.1, random_state=42)

# ✅ LOG
print(f"📊 Train images: {len(train_files)}")
print(f"📊 Val images: {len(val_files)}")

# --------------------------------
# 📁 MAKE DIRS
# --------------------------------
for split in ['train', 'val']:
    os.makedirs(os.path.join(OUTPUT_DIR, f'images/{split}'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, f'labels/{split}'), exist_ok=True)

# ✅ คัดลอกภาพ + label ไปตาม split
for split_name, split_files in [('train', train_files), ('val', val_files)]:
    for img_file in split_files:
        src_img = os.path.join(IMG_DIR, img_file)
        dst_img = os.path.join(OUTPUT_DIR, f'images/{split_name}', img_file)
        shutil.copy(src_img, dst_img)

        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_label = os.path.join(LABEL_DIR, label_file)
        dst_label = os.path.join(OUTPUT_DIR, f'labels/{split_name}', label_file)
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)

# --------------------------------
# 🧠 AUGMENT ONLY MOTORCYCLES
# --------------------------------
transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.4),
    A.CLAHE(p=0.3),
    A.ToGray(p=0.1),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

AUG_IMG_DIR = os.path.join(OUTPUT_DIR, "images/train")
AUG_LABEL_DIR = os.path.join(OUTPUT_DIR, "labels/train")

aug_count = 0
motorcycle_done = 0

# ✅ หา label ที่มี motorcycle
motorcycle_files = []
for label_file in os.listdir(AUG_LABEL_DIR):
    if not label_file.endswith(".txt") or label_file in IGNORE_FILES:
        continue
    with open(os.path.join(AUG_LABEL_DIR, label_file), "r") as f:
        if any(int(line.split()[0]) == TARGET_CLASS_ID for line in f):
            motorcycle_files.append(label_file)

# ✅ ทำ augment ซ้ำจนกว่าจะครบ MAX_AUG
aug_count = 0
motorcycle_done = 0
rounds = 0

while motorcycle_done < MAX_AUG and rounds < 10:
    random.shuffle(motorcycle_files)
    for label_file in motorcycle_files:
        image_name = label_file.replace(".txt", ".jpg")
        img_path = os.path.join(AUG_IMG_DIR, image_name)
        if not os.path.exists(img_path):
            image_name = label_file.replace(".txt", ".png")
            img_path = os.path.join(AUG_IMG_DIR, image_name)
        if not os.path.exists(img_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        with open(os.path.join(AUG_LABEL_DIR, label_file), "r") as f:
            lines = f.readlines()

        bboxes = []
        labels = []
        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            if cls_id == TARGET_CLASS_ID:
                x, y, w, h = map(float, parts[1:])
                bboxes.append([x, y, w, h])
                labels.append(cls_id)

        if not bboxes:
            continue

        for _ in range(AUG_TIMES):
            if motorcycle_done >= MAX_AUG:
                break

            try:
                aug = transform(image=image, bboxes=bboxes, class_labels=labels)
            except Exception as e:
                print(f"⚠️ Augment failed: {e}")
                continue

            aug_img = aug["image"]
            aug_bboxes = aug["bboxes"]
            aug_labels = aug["class_labels"]

            if not aug_bboxes:
                continue  # ไม่มี bbox เหลือหลังแปลง

            new_image_name = f"aug_motorcycle_{aug_count}_{image_name}"
            new_label_name = f"aug_motorcycle_{aug_count}_{label_file}"

            cv2.imwrite(os.path.join(AUG_IMG_DIR, new_image_name), aug_img)

            with open(os.path.join(AUG_LABEL_DIR, new_label_name), "w") as f:
                for box, cls_id in zip(aug_bboxes, aug_labels):
                    x, y, w, h = box
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            motorcycle_done += 1
            aug_count += 1
    rounds += 1

print(f"✅ Augmented motorcycle เพิ่มทั้งหมด {motorcycle_done} ภาพ")
