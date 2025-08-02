from ultralytics import YOLO

if __name__ == "__main__":
    # 初始训练
    model = YOLO("yolo11-seg.yaml").load("yolo11n-seg.pt")  # 加载预训练模型，如果本地没有会自动下载

    # 进行训练
    results = model.train(data="data-seg.yaml", epochs=100, imgsz=640, batch=4, workers=8)
