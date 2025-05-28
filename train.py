from ultralytics import YOLO

# -------------------------------
# ⚙️ CONFIG
# -------------------------------
MODEL_NAME = "yolo12n"
DATA_YAML = "E:/GhostTrack_YOLOv12n_Train/dataset.yaml"

EPOCHS = 94
IMG_SIZE = 640
BATCH_SIZE = 14
DEVICE = 0

PROJECT_NAME = "runs"
RUN_NAME = "train_GhostTrack_DEMO_yolo12n_2"

# -------------------------------
# 🚀 TRAINING
# -------------------------------
def train():
    print("🚀 เริ่มการเทรน YOLO ...")

    model = YOLO(MODEL_NAME)

    try:
        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            pretrained=False,
            lr0=0.001,                # learning rate
            lrf=0.0001,               # learning rate final
            warmup_epochs=15,         # warmup epochs
            warmup_bias_lr=0.05,      # warmup bias learning rate
            weight_decay=0.0008,      # weight decay
            cos_lr=True,              # cosine learning rate scheduler
            momentum=0.97,            # momentum for SGD
            dropout=0.2,              # dropout probability
            patience=0,               # number of epochs with no improvement before stopping
            multi_scale=True,        # ไม่เปิด multi scale (ช่วยลดการใช้ VRAM)
            rect=True,                # ใช้การปรับขนาดภาพ
            hsv_h=0.015,              # HUE adjustment range
            hsv_s=0.6,               # SAT adjustment range
            hsv_v=0.3,               # VALUE adjustment range
            scale=0.4,               # Scaling factor for multi-scale
            fliplr=0.5,              # Horizontal flip probability
            flipud=0.1,              # Vertical flip probability
            mosaic=0.05,              # ใช้ Mosaic augmentation แต่เบา ๆ
            mixup=0.05,              # Mixup augmentation
            cutmix=0.05,             # CutMix augmentation
            cache="disk",            # ใช้ disk cache (ประหยัด RAM)
            plots=True,              # เปิดการแสดงผลกราฟ
            seed=42,                 # การตั้งค่า random seed
            save=True,               # บันทึกโมเดล
            save_period=1,           # บันทึกโมเดลทุก 1 epoch
            save_json=True,          # บันทึกข้อมูลผลลัพธ์เป็น JSON
            project=PROJECT_NAME,    # พื้นที่บันทึกโปรเจค
            name=RUN_NAME,           # ชื่อ run
            resume=False,            # ไม่ resume จากโมเดลเก่า
            val=True,                # เปิดการ validation
            verbose=True,            # แสดงข้อมูลการเทรนแบบละเอียด
        )

    except Exception as e:
        print(f"[🔥 ERROR] Training failed: {e}")
        return

    print(f"🔥 เทรนโหดเสร็จแล้ว! ตรวจการบ้านที่: {PROJECT_NAME}/{RUN_NAME}/weights")

if __name__ == "__main__":
    train()
