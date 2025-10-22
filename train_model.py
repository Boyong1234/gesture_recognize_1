import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import data_load_explore


def setup_environment():
    """设置环境"""
    tf_data_root = 'D:/python3/anaconda/PythonMyFirstWork/datasetHere'
    os.environ['TFDS_DATA_DIR'] = os.path.join(tf_data_root, 'datasets')
    os.makedirs(os.environ['TFDS_DATA_DIR'], exist_ok=True)

    model_dir = os.path.join(tf_data_root, 'models')
    os.makedirs(model_dir, exist_ok=True)

    return tf_data_root, model_dir


def load_and_preprocess():
    (ds_train, ds_test), ds_info = tfds.load(
        'rock_paper_scissors',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    print(f"✅ 数据集加载成功!")
    print(f"   训练集: {ds_info.splits['train'].num_examples} 样本")
    print(f"   测试集: {ds_info.splits['test'].num_examples} 样本")
    print(f"   类别: {ds_info.features['label'].names}")

    def preprocess_image(image, label):
        """预处理单张图片"""
        # 调整大小到 150x150（减少计算量）
        image = tf.image.resize(image, [150, 150])
        # 归一化到 [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # 应用预处理
    ds_train = ds_train.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # 批量处理和优化
    batch_size = 32
    ds_train = ds_train.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print(f"✅ 数据预处理完成!")
    print(f"   批量大小: {batch_size}")
    print(f"   图像尺寸: 150x150")

    return ds_train, ds_test, ds_info


def create_cnn_model():
    """创建CNN模型"""
    print("🤖 构建CNN模型...")

    model = tf.keras.Sequential([
        # 第一卷积块
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        # 第二卷积块
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        # 第三卷积块
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        # 展平层
        tf.keras.layers.Flatten(),

        # 全连接层
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # 防止过拟合

        # 输出层
        tf.keras.layers.Dense(3, activation='softmax')  # 3个类别
    ])

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("✅ 模型构建完成!")
    model.summary()

    return model


def train_model(model, train_data, test_data, model_dir):
    """训练模型"""
    print("🎯 开始训练模型...")

    # 设置回调函数
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.0001,
            verbose=1
        )
    ]

    # 开始训练
    history = model.fit(
        train_data,
        epochs=20,
        validation_data=test_data,
        callbacks=callbacks,
        verbose=1
    )

    return history


def evaluate_model(model, test_data, ds_info):
    """评估模型"""
    print("📊 评估模型性能...")

    # 计算测试集准确率
    test_loss, test_accuracy = model.evaluate(test_data, verbose=0)
    print(f"🎯 测试集准确率: {test_accuracy:.2%}")
    print(f"📉 测试集损失: {test_loss:.4f}")

    # 生成预测
    y_true = []
    y_pred = []

    for images, labels in test_data:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    # 分类报告
    print("\n📋 分类报告:")
    print(classification_report(y_true, y_pred,
                                target_names=ds_info.features['label'].names))

    # 混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=ds_info.features['label'].names,
                yticklabels=ds_info.features['label'].names)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    def plot_training_history(history):
        """绘制训练历史"""
        print("📈 绘制训练历史...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 准确率曲线
        ax1.plot(history.history['accuracy'], label='训练准确率')
        ax1.plot(history.history['val_accuracy'], label='验证准确率')
        ax1.set_title('模型准确率')
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('准确率')
        ax1.legend()
        ax1.grid(True)

        # 损失曲线
        ax2.plot(history.history['loss'], label='训练损失')
        ax2.plot(history.history['val_loss'], label='验证损失')
        ax2.set_title('模型损失')
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('损失')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

        def main():
            """主函数"""
            print("🚀 开始完整的CNN手势识别项目")
            print("=" * 50)

            # 1. 设置环境
            tf_data_root, model_dir = setup_environment()

            # 2. 加载和预处理数据
            ds_train, ds_test, ds_info = data_load_explore()

            # 3. 创建模型
            model = create_cnn_model()

            # 4. 训练模型
            history = train_model(model, ds_train, ds_test, model_dir)

            # 5. 保存最终模型
            final_model_path = os.path.join(model_dir, 'final_gesture_model.h5')
            model.save(final_model_path)
            print(f"💾 模型已保存: {final_model_path}")

            # 6. 评估模型
            evaluate_model(model, ds_test, ds_info)

            # 7. 绘制训练历史
            plot_training_history(history)

            print("\n🎉 项目完成! 下一步可以运行实时手势识别。")

        if __name__ == "__main__":
            main()