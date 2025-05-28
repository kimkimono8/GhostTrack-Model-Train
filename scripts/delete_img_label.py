from pathlib import Path

label_root = Path("D:/yolov8n_train/labels")
target_class_id = "5"

found = False

for split in ["train", "val", "test"]:
    label_dir = label_root / split
    if not label_dir.exists():
        continue

    for file in label_dir.glob("*.txt"):
        if file.name == "classes.txt":
            continue  # ข้ามไฟล์นี้

        with file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if parts and parts[0] == target_class_id:
                    print(f"❗ พบ class_id=5 ที่ไฟล์: {file} | บรรทัด {i+1}: {line.strip()}")
                    found = True

if not found:
    print("✅ ไม่พบ class_id=5 ในไฟล์ label ใด ๆ")
