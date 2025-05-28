import os
import cv2

# ---------------------------
# 🔧 CONFIG
# ---------------------------
BASE_IMG_DIR = "E:/GhostTrack_YOLOv12n_Train/datasets/images"
BASE_LABEL_DIR = "E:/GhostTrack_YOLOv12n_Train/datasets/labels"
CLASS_NAMES = {0: "mycar", 1: "person", 2: "car", 3: "motorcycle"}

splits = ['train', 'val', 'test']  # ✅ วนดูหลาย split

# ---------------------------
# 🔁 Loop ทีละ Split
# ---------------------------
for split in splits:
    IMG_DIR = os.path.join(BASE_IMG_DIR, split)
    LABEL_DIR = os.path.join(BASE_LABEL_DIR, split)

    if not os.path.exists(IMG_DIR):
        continue

    for img_file in os.listdir(IMG_DIR):
        if not img_file.endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(IMG_DIR, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(LABEL_DIR, label_file)

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, box_w, box_h = map(float, parts[1:])

                # แปลงเป็นพิกัดจริง
                x1 = int((x_center - box_w / 2) * w)
                y1 = int((y_center - box_h / 2) * h)
                x2 = int((x_center + box_w / 2) * w)
                y2 = int((y_center + box_h / 2) * h)

                color = (0, 255, 0) if class_id != 3 else (0, 0, 255)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, CLASS_NAMES.get(class_id, str(class_id)), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # แสดงภาพ
        cv2.imshow("Check BBox", image)
        print(f"🖼️ [{split}] {img_file} | กด 'd' เพื่อลบ, กด 'q' เพื่อออก, ปุ่มอื่นข้าม")

        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == ord('d'):
            try:
                os.remove(img_path)
                if os.path.exists(label_path):
                    os.remove(label_path)
                print(f"❌ ลบ {img_file} และ label แล้ว")
            except Exception as e:
                print(f"❌ ไม่สามารถลบได้: {e}")
        elif key == ord('q'):
            print("🚪 ออกจากโปรแกรม")
            exit()
        else:
            print(f"✅ เก็บ {img_file} ไว้")
