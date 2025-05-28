from ultralytics import YOLO

model = YOLO('runs/train/GhostTrack_v2/weights/best.pt')

# Export เป็น ONNX หรือ TensorRT ถ้าจะใช้กับ edge device
model.export(format='onnx')
# model.export(format='engine')  # สำหรับ TensorRT (ต้องติดตั้ง plugin เพิ่ม)
