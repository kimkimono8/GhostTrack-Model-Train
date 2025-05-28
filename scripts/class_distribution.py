import os
import matplotlib.pyplot as plt
from collections import Counter

LABEL_DIR = "D:/GhostTrack_YOLOv12n_Train/datasets/labels"

# อ่านไฟล์ทั้งหมดใน LABEL_DIR
label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.txt')]

# เก็บ count ของแต่ละ class ID
class_counts = Counter()

for file_name in label_files:
    file_path = os.path.join(LABEL_DIR, file_name)
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            parts = line.strip().split()
            class_id = int(parts[0])
            class_counts[class_id] += 1

# แม็ปชื่อคลาส (ตามที่คุณใช้ใน YOLO)
class_names = {
    0: 'mycar',
    1: 'person',
    2: 'car',
    3: 'motorcycle'
}

# เตรียมข้อมูลสำหรับกราฟ
ids = sorted(class_counts.keys())
labels = [class_names.get(i, f'class_{i}') for i in ids]
counts = [class_counts[i] for i in ids]

# วาดกราฟ
plt.figure(figsize=(8,5))
plt.bar(labels, counts, color='orange')
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Instances")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# แสดงผลบน console ด้วย
print("🔍 Class Distribution:")
for i, label in zip(ids, labels):
    print(f"  🔸 {label:12} (ID {i}): {class_counts[i]}")
