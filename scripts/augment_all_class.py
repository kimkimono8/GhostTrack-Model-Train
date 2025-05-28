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
IMG_DIR = "D:/GhostTrack_YOLOv12n_Train/datasets/images"
LABEL_DIR = "D:/GhostTrack_YOLOv12n_Train/datasets/labels"
OUTPUT_DIR = "D:/GhostTrack_YOLOv12n_Train/datasets"

# 💥 Augmented output
AUG_TIMES = 4

IGNORE_FILES = ['classes.txt']  # ไฟล์ที่ต้องการข้าม

# ตรวจสอบว่าเป็นไฟล์ที่ต้องการข้ามหรือไม่
def is_valid_label_file(file_name):
    return file_name not in IGNORE_FILES and file_name.endswith('.txt')

image_files = []
for root, _, files in os.walk(IMG_DIR):
    for f in files:
        if f.endswith((".jpg", ".png")):
            full_path = os.path.join(root, f)
            image_files.append(os.path.relpath(full_path, IMG_DIR))

if len(image_files) == 0:
    raise ValueError(f"❌ ไม่พบไฟล์ภาพใน {IMG_DIR} — โปรดเช็กว่า path ถูกต้องและมีภาพอยู่จริง")


combined_img_dir = os.path.join(OUTPUT_DIR, 'images_combined')
os.makedirs(combined_img_dir, exist_ok=True)

for split in ['train']:  # ❗️รวมเฉพาะ train
    split_path = os.path.join(OUTPUT_DIR, f'images/{split}')
    if os.path.exists(split_path):
        for f in os.listdir(split_path):
            full_src = os.path.join(split_path, f)
            full_dst = os.path.join(combined_img_dir, f)
            if not os.path.exists(full_dst):
                shutil.copy(full_src, full_dst)

IMG_DIR = combined_img_dir  # 🔁 เปลี่ยน path ที่ใช้สำหรับแบ่ง train/val

# 📦 ดึงรายการภาพใหม่จากโฟลเดอร์ที่รวมแล้ว
image_files = [f for f in os.listdir(IMG_DIR) if f.endswith((".jpg", ".png"))]

# ✅ จากนั้นก็แบ่ง train/val ได้ตามปกติ
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)


# --------------------------------
# 📁 MAKE DIRS
# --------------------------------
for split in ['train', 'val']:
    os.makedirs(os.path.join(OUTPUT_DIR, f'images/{split}'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, f'labels/{split}'), exist_ok=True)

# --------------------------------
# ✂️ SPLIT TRAIN/VAL
# --------------------------------
image_files = [f for f in os.listdir(IMG_DIR) if f.endswith((".jpg", ".png"))]
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

if len(os.listdir(IMG_DIR)) == 0:
    for split in ['train', 'val']:
        split_img_dir = os.path.join(OUTPUT_DIR, f'images/{split}')
        for f in os.listdir(split_img_dir):
            shutil.copy(os.path.join(split_img_dir, f), os.path.join(IMG_DIR, f))
    print("🔄 รวมไฟล์ภาพกลับมาที่ images/ แล้ว")

# --------------------------------
# 🧠 AUGMENTATION PIPELINE
# --------------------------------
transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.RandomScale(scale_limit=0.1, p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.CLAHE(clip_limit=2.0, p=0.2),
    A.ChannelShuffle(p=0.1),
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),
    A.ToGray(p=0.05),
    A.Rotate(limit=30, p=0.3),  # การหมุนภาพ
    A.Perspective(p=0.2),  # การบิดเบือน
])

# --------------------------------
# ✅ FIX จำนวน augment ต่อ class ตามตาราง
# --------------------------------
AUG_PER_CLASS = {
    0: 0,   # mycar
    1: 40,   # person
    2: 60,   # car
    3: 100    # motorcycle
}

AUG_IMG_DIR = os.path.join(OUTPUT_DIR, "images/train")
AUG_LABEL_DIR = os.path.join(OUTPUT_DIR, "labels/train")

aug_count = 0
aug_done_per_class = defaultdict(int)
original_class_counts = defaultdict(int)

def count_classes(label_dir, exclude_augmented=False):
    class_counts = defaultdict(int)
    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue
        if exclude_augmented and file.startswith("aug_"):
            continue
        with open(os.path.join(label_dir, file)) as f:
            for line in f:
                try:
                    class_id = int(line.strip().split()[0])
                    class_counts[class_id] += 1
                except ValueError:
                    print(f"⚠️ ข้ามบรรทัดในไฟล์ {file} เพราะ class_id ไม่ใช่ตัวเลข: {line.strip()}")
    return class_counts
    
def class_name_to_id(class_name):
    class_mapping = {
        "mycar": 0,
        "person": 1,
        "car": 2,
        "motorcycle": 3
    }
    
    # ถ้าชื่อ class ไม่ตรงกับที่กำหนดใน class_mapping ให้คืนค่า None
    return class_mapping.get(class_name, None)

# นับจำนวน class ใน training set ก่อน
original_counts = count_classes(AUG_LABEL_DIR, exclude_augmented=True)
original_class_counts.update(original_counts)

# สำหรับแต่ละ class ที่ต้องการ augment
for label_file in os.listdir(AUG_LABEL_DIR):
    if not label_file.endswith(".txt"):
        continue

    if label_file == "classes.txt":  # ✅ ข้ามไฟล์ classes.txt
        continue

    label_path = os.path.join(AUG_LABEL_DIR, label_file)
    with open(label_path, "r") as f:
        lines = f.readlines()

    # ✅ แก้ตรงนี้ให้ใช้ try-except ป้องกัน error
    present_classes = set()
    for line in lines:
        try:
            class_id = int(line.strip().split()[0])
            present_classes.add(class_id)
        except ValueError:
            print(f"⚠️ ข้ามบรรทัดในไฟล์ {label_file} เพราะ class_id ไม่ใช่ตัวเลข: {line.strip()}")

    target_classes = [cid for cid in present_classes if aug_done_per_class[cid] < AUG_PER_CLASS[cid]]

    if not target_classes:
        continue

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

    for _ in range(AUG_TIMES):
        for cid in target_classes:
            if aug_done_per_class[cid] >= AUG_PER_CLASS[cid]:
                continue

            aug = transform(image=image)
            aug_img = aug["image"]
            new_image_name = f"aug_{aug_count}_{image_name}"
            new_label_name = f"aug_{aug_count}_{label_file}"

            cv2.imwrite(os.path.join(AUG_IMG_DIR, new_image_name), aug_img)
            shutil.copy(label_path, os.path.join(AUG_LABEL_DIR, new_label_name))

            aug_done_per_class[cid] += 1
            aug_count += 1

print(f"✅ Augmented เพิ่มทั้งหมด {aug_count} ภาพ (ตามที่กำหนดจำนวนต่อ class)")
print("📈 จำนวน augment ต่อ class:", dict(aug_done_per_class))

# --------------------------------
# 📈 CLASS DISTRIBUTION PLOT
# --------------------------------
def plot_class_distribution():
    # นับจำนวน class ใน train และ val
    train_counts = count_classes(os.path.join(OUTPUT_DIR, 'labels/train'), exclude_augmented=True)
    val_counts = count_classes(os.path.join(OUTPUT_DIR, 'labels/val'))

    # รวมข้อมูลจาก train และ val
    combined_counts = defaultdict(int)
    for cid in set(train_counts.keys()).union(val_counts.keys()):
        combined_counts[cid] = train_counts.get(cid, 0) + val_counts.get(cid, 0)

    all_class_ids = sorted(set(train_counts.keys()).union(val_counts.keys()))
    labels = [CLASS_NAMES.get(cid, f"class_{cid}") for cid in all_class_ids]

    x = range(len(all_class_ids))

    # กราฟเปรียบเทียบก่อนหลังการ augment
    plt.figure(figsize=(10,5))
    plt.bar(x, [train_counts.get(cid, 0) for cid in all_class_ids], width=0.3, label="Original Train", align="center", color='skyblue')
    plt.bar([i + 0.3 for i in x], [val_counts.get(cid, 0) for cid in all_class_ids], width=0.3, label="Val", align="center", color='orange')
    plt.xticks([i + 0.3 for i in x], labels)
    plt.legend()
    plt.title("Class Distribution: Original vs Augmented vs Val")
    plt.tight_layout()
    plt.savefig("class_distribution_after_augment.png")
    plt.show(block=False)  # เปิดกราฟ แต่ไม่บล็อกโปรแกรม
    plt.pause(3)           # แสดง 3 วินาที
    plt.close()            # ปิดกราฟอัตโนมัติ

    aug_counts = count_classes(os.path.join(OUTPUT_DIR, 'labels/train'), exclude_augmented=False)

    plt.figure(figsize=(10,5))
    plt.bar(x, [train_counts.get(cid, 0) for cid in all_class_ids], width=0.3, label="Original Train", align="center", color='skyblue')
    plt.bar([i + 0.3 for i in x], [aug_counts.get(cid, 0) for cid in all_class_ids], width=0.3, label="Train+Augmented", align="center", color='green')
    plt.bar([i + 0.6 for i in x], [val_counts.get(cid, 0) for cid in all_class_ids], width=0.3, label="Val", align="center", color='orange')
    plt.xticks([i + 0.3 for i in x], labels)
    plt.legend()
    plt.title("Class Distribution: Original vs Augmented vs Val")
    plt.tight_layout()
    plt.savefig("class_distribution_after_augment.png")
    plt.show()

# เรียกใช้ฟังก์ชันสร้างกราฟ
plot_class_distribution()

print("✅ สร้างกราฟ class_distribution ก่อนและหลังการ augment เรียบร้อยแล้ว")
