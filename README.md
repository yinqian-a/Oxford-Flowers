
# 牛津102花卉分类 - TensorFlow CNN实现

基于 TensorFlow 2.x 和卷积神经网络（CNN）实现的牛津102花卉数据集分类项目。包含数据集预处理、模型训练、评估和可视化完整流程。

## 🌸 项目简介

牛津102花卉数据集（Oxford 102 Flowers）包含102种英国常见花卉，每种花卉有40-258张图片，总计约8000张图片。本项目使用自定义CNN模型对该数据集进行分类，实现从数据准备到模型部署的完整流程。

## ✨ 功能特点

- **数据集自动划分**：根据官方提供的 train/validate/test 划分文件，自动将图片整理到对应目录
- **GPU内存优化**：支持GPU训练，并自动配置显存增长策略
- **自定义CNN模型**：4层卷积层 + 全连接层，带Dropout防止过拟合
- **训练过程可视化**：实时绘制准确率和损失曲线，便于监控训练状态
- **模型保存与复用**：训练完成后自动保存为 H5 格式，方便后续部署

## 🛠️ 技术栈

| 组件 | 技术 |
| :--- | :--- |
| 深度学习框架 | TensorFlow 2.x + Keras |
| 图像处理 | Pillow / OpenCV（通过tf.image） |
| 数据处理 | NumPy |
| 可视化 | Matplotlib |
| GPU加速 | CUDA + cuDNN（可选） |

## 📦 项目结构

```
.
├── main.py                      # 数据集预处理脚本
├── train.py                     # 模型训练脚本
├── requirements.txt             # 项目依赖清单
├── oxford-102-flowers/          # 原始数据集目录（需自行下载）
│   ├── jpg/                     # 所有图片
│   ├── train.txt                # 训练集划分
│   ├── validate.txt             # 验证集划分
│   └── test.txt                 # 测试集划分
├── data/                        # 预处理后数据集（自动生成）
│   ├── train/                   # 训练集
│   │   ├── class0/              # 类别0图片
│   │   ├── class1/              # 类别1图片
│   │   └── ...
│   ├── validate/                # 验证集
│   └── test/                    # 测试集
├── models/                      # 训练好的模型（自动生成）
│   └── Classifier_for_Oxford_102Flowers_YYYYMMDDHHMMSS.h5
└── README.md                    # 项目说明文档
```

## 📥 数据集下载

牛津102花卉数据集需要手动下载，推荐以下方式：

### 方式一：官网下载（推荐）

1. 访问 [Oxford 102 Flowers 官网](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
2. 下载以下三个文件：
   - `102flowers.tgz`（图片压缩包）
   - `setid.mat`（划分文件，MATLAB格式）
   - `imagelabels.mat`（标签文件）

### 方式二：使用第三方数据集工具

```bash
# 使用 torchvision（需安装PyTorch）
import torchvision.datasets as datasets
dataset = datasets.Flowers102(root='./oxford-102-flowers', download=True)
```

> ⚠️ **注意**：本项目使用的 `train.txt` / `validate.txt` / `test.txt` 是基于官方划分生成的文本文件。如果你使用 `setid.mat`，需要先通过MATLAB或Python脚本转换格式。

### 数据目录结构要求

```
oxford-102-flowers/
├── jpg/
│   ├── image_00001.jpg
│   ├── image_00002.jpg
│   └── ... (约8000张图片)
├── train.txt          # 格式：jpg/image_00001.jpg 0
├── validate.txt       # 格式：jpg/image_00052.jpg 1
└── test.txt           # 格式：jpg/image_00103.jpg 2
```

其中 `.txt` 文件每行格式为：`图片相对路径 类别索引(0-101)`

## 🚀 使用指南

### 环境配置

#### 1. 一键安装所有依赖

在项目根目录下执行：

```bash
pip install -r requirements.txt
```

如果下载速度太慢，使用国内镜像源：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 2. 轻量版安装（如果不需要MAT文件转换）

如果你不需要从 `.mat` 文件转换数据（直接用已有的 `train.txt` 等文件），可以只安装核心依赖：

```bash
pip install tensorflow numpy Pillow matplotlib
```

#### 3. GPU版本配置（可选）

如果有NVIDIA GPU，需要安装对应CUDA版本的TensorFlow：

```bash
# CUDA 11.2 或更高版本
pip install tensorflow-gpu

# 或者指定版本（推荐）
pip install tensorflow==2.10.0
```

> ⚠️ **注意**：TensorFlow 2.11 及以上版本不再区分CPU/GPU版本，会自动检测硬件。如果安装最新版，直接 `pip install tensorflow` 即可。

### 第一步：数据集预处理

运行 `main.py`，将原始图片按类别和划分分配到 `data/` 目录：

```bash
python main.py
```

执行后会在 `data/` 目录下生成以下结构：
```
data/
├── train/
│   ├── class0/
│   ├── class1/
│   └── ...
├── validate/
│   ├── class0/
│   ├── class1/
│   └── ...
└── test/
    ├── class0/
    ├── class1/
    └── ...
```

### 第二步：训练模型

运行 `train.py` 开始训练：

```bash
python train.py
```

训练过程中的输出包括：
- 各数据集加载信息（图片数量、类别数）
- 模型结构摘要（总参数量）
- 每个epoch的训练/验证准确率和损失
- 测试集最终准确率

### 第三步：查看训练结果

训练完成后：
1. **模型文件**：保存在 `models/` 目录，文件名包含时间戳
2. **可视化图表**：自动弹出准确率和损失曲线图

```
训练准确率曲线                损失曲线
    1.0 ───────                   2.5 ──────
   0.8    ╱                     2.0   ╱
   0.6  ╱                      1.5 ╱
   0.4 ╱                       1.0╱
     ─────────────────             ─────────────
     0   10  20  30               0   10  20  30
```

## ⚙️ 配置调整

你可以修改 `train.py` 开头的参数来调整训练配置：

```python
IMAGE_SIZE = (128, 128)    # 图片尺寸（可调大以提高精度）
BATCH_SIZE = 32            # 批次大小（显存不足时减小）
EPOCHS = 30                # 训练轮数
CLASSES_NUMBER = 102       # 类别总数
```

### 模型架构调整

如果你想要更深的网络或不同的结构，可以修改 `model = models.Sequential([...])` 部分：

```python
# 添加更多卷积层
layers.Conv2D(512, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
```

## 📊 预期结果

在默认配置（128x128输入，30轮训练）下，预期能达到：

| 数据集 | 准确率 |
| :--- | :--- |
| 训练集 | 85% - 92% |
| 验证集 | 70% - 78% |
| 测试集 | 68% - 75% |

> 💡 **提示**：可以通过增加图片尺寸（如224x224）、加深网络或使用预训练模型（如ResNet）来提升准确率。

## ❓ 常见问题

### Q1: 报错 `No module named 'tensorflow'`

请安装 TensorFlow：
```bash
pip install tensorflow
```

如果有GPU，可以安装GPU版本：
```bash
pip install tensorflow-gpu
```

### Q2: GPU显存不足（OOM）

解决方法：
1. 减小 `BATCH_SIZE`（如从32改为16）
2. 减小 `IMAGE_SIZE`（如从128改为64）
3. 使用CPU训练（代码会自动检测，无GPU时使用CPU）

### Q3: 数据集路径找不到

检查 `oxford-102-flowers/` 目录是否存在，且包含：
- `jpg/` 文件夹
- `train.txt` / `validate.txt` / `test.txt` 三个文件

如果缺少 `.txt` 文件，请参考数据集下载部分生成。

### Q4: 训练过程太慢

- 启用GPU加速（需CUDA支持）
- 增加 `BATCH_SIZE` 提升GPU利用率（显存允许情况下）
- 减少 `EPOCHS` 训练轮数

### Q5: 中文图表显示为方框

代码已配置中文字体支持：
```python
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
```
如果仍然显示异常，请确保系统安装了中文字体（如SimHei）。

### Q6: 如何加载保存的模型进行预测？

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# 加载模型
model = tf.keras.models.load_model('./models/Classifier_for_Oxford_102Flowers_20240101120000.h5')

# 预测单张图片
img_path = 'test.jpg'
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
print(f'预测类别: {predicted_class}')
```

## 📄 许可证

本项目遵循 MIT 许可证。数据集版权归牛津大学视觉几何组（VGG）所有，请遵守其使用条款。

## 📚 参考文献

- [Oxford 102 Flowers Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [TensorFlow 官方文档](https://www.tensorflow.org/)
- Nilsback, M-E., & Zisserman, A. (2008). Automated flower classification over a large number of classes.

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进本项目！

## 📧 联系方式

如有问题，请通过 GitHub Issues 联系。

---

现在这份 README 已经包含了完整的 `requirements.txt` 内容，你可以直接**全部复制**到项目根目录的 `README.md` 文件中使用！需要我再帮你调整其他内容吗？
