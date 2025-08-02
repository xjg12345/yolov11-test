from ultralytics import YOLO

# 加载前面训练的模型
model = YOLO('runs/detect/train3/weights/best.pt')

img_list = ['cat.png']

for img in img_list:
    # 运行推理，并附加参数 save:是否保存文件
    model.predict(img, save=True, conf=0.5, )
