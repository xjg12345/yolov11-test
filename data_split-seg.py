import os
import random
import shutil


def split_dataset(input_image_folder, input_label_folder, output_folder, test_ratio=0.2):
    # 创建训练集和验证集文件夹
    train_images_folder = os.path.join(output_folder, 'train', 'images')
    train_labels_folder = os.path.join(output_folder, 'train', 'labels')
    val_images_folder = os.path.join(output_folder, 'val', 'images')
    val_labels_folder = os.path.join(output_folder, 'val', 'labels')

    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)

    # 获取所有图像文件列表
    images = [f for f in os.listdir(input_image_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # 随机打乱图像文件列表
    random.shuffle(images)

    # 计算验证集的数量
    val_size = int(len(images) * test_ratio)

    # 划分验证集和训练集
    val_images = images[:val_size]
    train_images = images[val_size:]

    # 复制验证集图像和标签
    for image in val_images:
        label = os.path.splitext(image)[0] + '.txt'
        if os.path.exists(os.path.join(input_label_folder, label)):
            shutil.copy(os.path.join(input_image_folder, image), os.path.join(val_images_folder, image))
            shutil.copy(os.path.join(input_label_folder, label), os.path.join(val_labels_folder, label))
        else:
            print(f"Warning: Label file {label} not found for image {image}")

    # 复制训练集图像和标签
    for image in train_images:
        label = os.path.splitext(image)[0] + '.txt'
        if os.path.exists(os.path.join(input_label_folder, label)):
            shutil.copy(os.path.join(input_image_folder, image), os.path.join(train_images_folder, image))
            shutil.copy(os.path.join(input_label_folder, label), os.path.join(train_labels_folder, label))
        else:
            print(f"Warning: Label file {label} not found for image {image}")


input_image_folder = 'data-seg/images'  # 图片路径
input_label_folder = 'data-seg/labels'  # 标签路径
output_folder = 'datasets-seg'  # 输出目录
split_dataset(input_image_folder, input_label_folder, output_folder, test_ratio=0.2)
