from ultralytics import YOLO
import numpy as np
import cv2

# 加载训练的模型
model = YOLO('runs/segment/train/weights/best.pt')

results = model.predict(source="cat.png", retina_masks=True)

for result in results:
    if not hasattr(result, 'masks') or result.masks is None:
        continue

    img = result.orig_img.copy()
    orig_h, orig_w = result.orig_shape
    print(f'宽：{orig_w}，高：{orig_h}')
    masks = result.masks
    boxes = result.boxes

    for index, (mask, box) in enumerate(zip(masks, boxes)):
        # 获取检测框坐标
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy()) # 如果模型在 GPU 上运行，必须调用 .cpu() 才能转为 NumPy
        print(f"目标 {index + 1} 边框坐标: ({x1}, {y1}) ({x2}, {y2})")
        width = x2 - x1
        height = y2 - y1
        # 计算实例中心点
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        # 绘制边界框和中心点
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
        # 显示宽高和中心点信息
        info_text = f"W:{width:.1f} H:{height:.1f} Center:({center_x},{center_y})"
        cv2.putText(img, info_text, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # 绘制中心点到边界框的线
        cv2.line(img, (x1 + 5, center_y), (center_x - 10, center_y), (255, 0, 0), 2)
        cv2.line(img, (x2 - 5, center_y), (center_x + 10, center_y), (255, 0, 0), 2)

        # 绘制掩膜轮廓
        mask_xy = mask.xy[0]
        print(f"目标 {index + 1} 轮廓点数: {len(mask_xy)}")  # 每个目标的轮廓点数量
        print(f'目标 {index + 1} 轮廓面积：', cv2.contourArea(mask_xy))  # 计算轮廓面积
        contours = [np.array(mask_xy, dtype=np.int32)]  # 转换为int32类型
        img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)  # 绘制轮廓

        # 绘制掩膜区域
        mask_data = mask.data.cpu().numpy()
        mask_data = (mask_data > 0.5).astype(np.uint8)
        mask_resized = cv2.resize(mask_data[0], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)  # 调整掩膜尺寸

        new_img = img.copy()
        y_coords, x_coords = np.where(mask_resized == 1)
        print(f'掩膜点数：', len(y_coords))
        for x, y in zip(x_coords, y_coords):
            cv2.circle(new_img, (x, y), 1, (255, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(img, alpha, new_img, 1 - alpha, 0, img)

    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 