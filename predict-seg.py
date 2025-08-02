from ultralytics import YOLO

# 加载训练的模型
model = YOLO('runs/segment/train/weights/best.pt')

img_list = ['cat.png']

for img in img_list:
    # 运行推理，并附加参数 save:是否保存文件 retina_masks：返回高分辨率分割掩码
    model.predict(img, save=True, conf=0.5, retina_masks=True)
