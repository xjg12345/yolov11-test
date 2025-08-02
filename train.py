from ultralytics import YOLO

if __name__ == '__main__':
    # 初始训练
    model = YOLO("yolo11.yaml").load("yolo11n.pt")  # 加载预训练模型，如果本地没有会自动下载

    results = model.train(
        data="data.yaml",  # 数据集配置文件的路径（例如 coco8.yaml）。该文件包含数据集特定的参数，包括训练和验证数据的路径、类名和类数。
        optimizer='auto',  # 训练使用优化器，可选 auto,SGD,Adam,AdamW 等
        epochs=200,  # 总训练周期数。每个周期代表对整个数据集的一次完整遍历。调整此值会影响训练时长和模型性能。
        imgsz=640,  # 训练目标图像大小。所有图像在输入模型之前都会被调整为这个尺寸。影响模型精度和计算复杂度。
        device=0,  # 指定训练的计算设备：单个 GPU（device=0）、多个 GPU（device=0,1）、CPU（device=cpu），或 Apple Silicon 的 MPS（device=mps）。
        batch=4,  # 批量大小，即单次输入多少图片训练，有三种模式：设置为整数（例如 batch=16），自动模式为60% GPU内存利用率（batch=-1），或指定利用率的自动模式（batch=0.70）。
        workers=8,  # 数据加载的工作线程数（每个 RANK 如果是多 GPU 训练）。影响数据预处理和输入模型的速度，尤其在多 GPU 设置中非常有用。
        patience=100  # 在验证指标无改进的情况下等待的周期数，超过该周期后提前停止训练。帮助防止过拟合，当性能停滞时停止训练。
    )
