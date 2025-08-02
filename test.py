from ultralytics import YOLO

if __name__ == "__main__":
    # 加载预训练的 YOLOv11n 模型
    model = YOLO('yolo11n.pt')
    source = 'cat.png'

    results = model.predict(source)

    for i, r in enumerate(results):
        r.show()
