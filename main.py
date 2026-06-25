import os
import shutil

if __name__ == "__main__":
    # ====================== 路径配置 ======================
    # 项目根目录（如果你在 Py0408 里直接运行，这里写 '.' 即可）
    BASE_DIR = "."
    

    # 原始数据集路径
    OXFORD_DIR = os.path.join(BASE_DIR, "oxford-102-flowers")
    JPG_DIR = os.path.join(OXFORD_DIR, "jpg")

    # 三个划分文件
    SPLIT_FILES = {
        "train":    os.path.join(OXFORD_DIR, "train.txt"),
        "validate": os.path.join(OXFORD_DIR, "validate.txt"),
        "test":     os.path.join(OXFORD_DIR, "test.txt")
    }

    # 输出根目录
    DATA_ROOT = os.path.join(BASE_DIR, "data")

    # 类别总数 0~101
    CLASSES_NUMBER = 52

    # ====================== 创建目录结构 ======================
    for split in ["train", "validate", "test"]:
        for cls_idx in range(CLASSES_NUMBER):
            dir_path = os.path.join(DATA_ROOT, split, f"class{cls_idx}")
            os.makedirs(dir_path, exist_ok=True)

    # ====================== 逐个文件复制 ======================
    for split_name, txt_path in SPLIT_FILES.items():
        print(f"正在处理 {split_name} ...")

        if not os.path.exists(txt_path):
            print(f"警告：未找到 {txt_path}")
            continue

        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 格式：jpg/image_01234.jpg 类别号
            img_rel_path, cls_str = line.split()
            cls = int(cls_str)

            # 原图完整路径
            src_img = os.path.join(OXFORD_DIR, img_rel_path)

            # 目标路径
            dst_dir = os.path.join(DATA_ROOT, split_name, f"class{cls}")
            dst_img = os.path.join(dst_dir, os.path.basename(src_img))

            # 复制
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)

    print("=" * 60)
    print("✅ 数据集划分完成！最终结构：")
    print(f"{DATA_ROOT}/")
    print("  ├─ train/")
    print("  │   └─ class0 ~ class101")
    print("  ├─ validate/")
    print("  │   └─ class0 ~ class101")
    print("  └─ test/")
    print("  │   └─ class0 ~ class101")

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
