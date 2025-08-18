import json
import os

label_to_class_id = {
    "cat": 0,  # 从0开始
    "dog": 1,
    # 其他类别...
}


def convert_labelme_json_to_yolo(json_file, output_dir):
    try:
        with open(json_file) as f:
            labelme_data = json.load(f)

        img_width = labelme_data["imageWidth"]
        img_height = labelme_data["imageHeight"]

        file_name = os.path.splitext(os.path.basename(json_file))[0]
        txt_path = os.path.join(output_dir, f"{file_name}.txt")

        with open(txt_path, "w") as txt_file:
            for shape in labelme_data["shapes"]:
                label = shape["label"]
                points = shape["points"]

                if not points:
                    continue

                class_id = label_to_class_id.get(label)
                if class_id is None:
                    print(f"Warning: 跳过未定义标签 '{label}'")
                    continue

                # 检查多边形是否闭合
                if points[0] != points[-1]:
                    points.append(points[0])

                normalized = [(x / img_width, y / img_height) for x, y in points]
                line = f"{class_id} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in normalized)
                txt_file.write(line + "\n")

    except Exception as e:
        print(f"处理文件 {json_file} 时出错: {str(e)}")


if __name__ == "__main__":
    json_dir = "data-seg/json"  # labelme标注存放的目录
    output_dir = "data-seg/labels"  # 输出目录

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            json_path = os.path.join(json_dir, json_file)
            convert_labelme_json_to_yolo(json_path, output_dir)
