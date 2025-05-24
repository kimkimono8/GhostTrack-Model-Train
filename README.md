# 🧠 Custom YOLOv12n Model (896x896)

This is a high-performance object detection model trained using **YOLOv12n** on custom data at **896×896** resolution.  
The model achieves **exceptional accuracy** while remaining lightweight — ideal for edge devices, RTSP cameras, and real-time applications.

---

## 🚀 Performance

| Metric          | Value  |
|-----------------|--------|
| 📐 Input Size    | 896 × 896 px |
| 🎯 mAP@0.50      | **0.99**     |
| 📊 mAP@0.50:0.95 | **0.893**    |
| ⚡ Model         | YOLOv12n (Nano) |

---

## 📦 Highlights

- 🧠 **Custom-trained YOLOv12n** with excellent generalization
- 💡 Optimized for **low-latency inference**
- 🔧 Suitable for:
  - Real-time surveillance systems
  - Embedded AI edge devices
  - Drones, robots, and IoT
- 🔋 Ultra-lightweight: Runs efficiently even on low-spec hardware

---

## 📂 Usage

1. Clone the inference repo (or use in your own codebase)
2. Place the `.pt` model file inside the `models/` directory
3. Example usage (Python):

```python
from ultralytics import YOLO

model = YOLO("models/yolov12n_custom.pt")
results = model("input.jpg", imgsz=896)
results.show()
