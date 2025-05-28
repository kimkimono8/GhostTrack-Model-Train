from ultralytics import YOLO

def main():
    # โหลดน้ำหนักของโมเดล
    model = YOLO("D:/GhostTrack_YOLOv12n_Train/runs/train_GhostTrack_DEMO_yolo12n_5/weights/best.pt") 
    
    # ประเมินผล
    metrics = model.val(data="D:/GhostTrack_YOLOv12n_Train/dataset.yaml")
    
    # แสดงผล
    print(metrics)

if __name__ == "__main__":
    main()
