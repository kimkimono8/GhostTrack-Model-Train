import os
import random
import shutil

# ==== CONFIG ====
source_folder = 'D:/GhostTrack_YOLOv12n_Train/datasets'         # ที่เก็บรูปและ .txt เดิม
output_folder = 'D:/GhostTrack_YOLOv12n_Train/datasets'    # ที่ต้องการแยกเก็บ train/val
train_ratio = 0.8                           # 80% สำหรับ train

# ==== เตรียมโฟลเดอร์ ====
train_img_dir = os.path.join(output_folder, 'train')
val_img_dir = os.path.join(output_folder, 'val')

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)

# ==== รวบรวมไฟล์ .jpg ====
image_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.jpg')]
image_files.sort()

# ==== สุ่มแบ่ง ====
random.shuffle(image_files)
split_idx = int(len(image_files) * train_ratio)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# ==== ฟังก์ชันคัดลอกไฟล์ (.jpg + .txt) ====
def move_pair(files, dest_folder):
    for img_file in files:
        base_name = os.path.splitext(img_file)[0]
        txt_file = base_name + '.txt'

        # path เดิม
        img_src = os.path.join(source_folder, img_file)
        txt_src = os.path.join(source_folder, txt_file)

        # path ใหม่
        img_dst = os.path.join(dest_folder, img_file)
        txt_dst = os.path.join(dest_folder, txt_file)

        shutil.copy(img_src, img_dst)
        if os.path.exists(txt_src):
            shutil.copy(txt_src, txt_dst)
        else:
            print(f"⚠️ ไม่พบไฟล์ txt สำหรับ: {img_file}")

# ==== ย้ายไฟล์ ====
move_pair(train_files, train_img_dir)
move_pair(val_files, val_img_dir)

print(f"✅ แบ่งเสร็จแล้ว: Train {len(train_files)} รูป / Val {len(val_files)} รูป")
