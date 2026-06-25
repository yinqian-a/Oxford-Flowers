import tensorflow as tf
from tensorflow.keras import layers, models
import os
from datetime import datetime
import matplotlib.pyplot as plt

# ======================
# 0. 基础配置
# ======================
TRAIN_DIR = './data/train'
VALIDATE_DIR = './data/validate'
TEST_DIR = './data/test'

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
CLASSES_NUMBER = 102

if __name__ == '__main__':
    # ======================
    # 1. GPU 内存优化
    # ======================
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('✅ GPU 已配置')

    # ======================
    # 2. 加载训练集
    # ======================
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True
    )

    # ======================
    # 3. 加载验证集
    # ======================
    validate_ds = tf.keras.utils.image_dataset_from_directory(
        directory=VALIDATE_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )

    # ======================
    # 4. 加载测试集
    # ======================
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )

    # ======================
    # 5. 数据预处理
    # ======================
    def process(image, label):
        return image / 255.0, label

    train_ds = train_ds.map(process).prefetch(tf.data.AUTOTUNE)
    validate_ds = validate_ds.map(process).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(process).prefetch(tf.data.AUTOTUNE)

    # ======================
    # 6. 构建CNN模型
    # ======================
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(CLASSES_NUMBER, activation='softmax')
    ])

    # ======================
    # 7. 编译
    # ======================
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # ======================
    # 8. 训练（带验证）
    # ======================
    diary = model.fit(
        train_ds,
        validation_data=validate_ds,
        epochs=EPOCHS
    )

    # ======================
    # 9. 测试集评估
    # ======================
    print('\n===== 测试集评估 =====')
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f'测试集准确率: {test_accuracy:.4f}')

    # ======================
    # 10. 保存模型
    # ======================
    os.makedirs('./models', exist_ok=True)
    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    model_path = f'./models/Classifier_for_Oxford_102Flowers_{time_stamp}.h5'
    model.save(model_path)
    print(f'\n✅ 模型已保存至：{model_path}')

    # ======================
    # 11. 绘图
    # ======================
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    train_accuracy = diary.history['accuracy']
    validate_accuracy = diary.history['val_accuracy']
    train_loss = diary.history['loss']
    validate_loss = diary.history['val_loss']

    my_canvas = plt.figure(figsize=(14, 5))

    area1 = my_canvas.add_subplot(1, 2, 1)
    area1.plot(train_accuracy, label='训练准确率')
    area1.plot(validate_accuracy, label='验证准确率')
    area1.set_title('准确率曲线')
    area1.set_xlabel('Epoch')
    area1.set_ylabel('Acc')
    area1.legend()

    area2 = my_canvas.add_subplot(1, 2, 2)
    area2.plot(train_loss, label='训练损失')
    area2.plot(validate_loss, label='验证损失')
    area2.set_title('损失曲线')
    area2.set_xlabel('Epoch')
    area2.set_ylabel('Loss')
    area2.legend()

    plt.tight_layout()
    plt.show()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
