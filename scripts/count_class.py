from pathlib import Path
from collections import defaultdict
import yaml

DATASET_YAML_PATH = Path("E:/GhostTrack_YOLOv12n_Train/dataset.yaml")

def load_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def count_labels(label_dir: Path, split_name: str, class_names: list):
    counts = defaultdict(int)
    total_instances = 0
    total_files = 0

    if not label_dir.exists():
        print(f"[⚠️] ไม่พบโฟลเดอร์ {split_name}: {label_dir}")
        return counts, 0, 0

    for txt_file in label_dir.glob("*.txt"):
        if txt_file.name == "classes.txt":
            continue  # ✅ ข้าม classes.txt

        with txt_file.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        valid_lines = 0
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
                counts[class_id] += 1
                total_instances += 1
                valid_lines += 1
            except ValueError:
                print(f"[⚠️] {txt_file.name}: ข้ามบรรทัดไม่ถูกต้อง -> {line.strip()}")

        if valid_lines > 0:
            total_files += 1

    print(f"\n📁 Split: {split_name}")
    print(f"🔍 พบทั้งหมด {total_instances} instance จาก {total_files} ไฟล์ annotation")
    for class_id in sorted(counts.keys()):
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        print(f"  🔸 {class_name:<12} (ID {class_id}): {counts[class_id]}")

    return counts, total_instances, total_files

def main():
    data = load_yaml(DATASET_YAML_PATH)
    class_names = data.get("names", [])
    print(f"✅ โหลด dataset.yaml สำเร็จ\n👀 คลาสทั้งหมด:", ", ".join([f"{i}: {n}" for i, n in enumerate(class_names)]))

    for split in ["train", "val", "test"]:
        split_path = data.get(split)
        if split_path:
            split_path = Path(split_path)
            label_dir = split_path.parent.parent / 'labels' / split_path.name
            count_labels(label_dir, split, class_names)

if __name__ == "__main__":
    main()
