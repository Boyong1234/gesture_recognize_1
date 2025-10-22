import os

# 设置根目录
tf_data_root = 'D:/python3/anaconda/PythonMyFirstWork/datasetHere'

# 设置各个子目录
os.environ['TFDS_DATA_DIR'] = os.path.join(tf_data_root, 'datasets')
os.environ['TFHUB_CACHE_DIR'] = os.path.join(tf_data_root, 'hub_modules')

# 创建这些目录
os.makedirs(os.environ['TFDS_DATA_DIR'], exist_ok=True)
os.makedirs(os.environ['TFHUB_CACHE_DIR'], exist_ok=True)

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

model_dir = os.path.join(tf_data_root, 'models')
log_dir = os.path.join(tf_data_root, 'logs')
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# 加载数据集，它会自动下载到TFDS_DATA_DIR指定的目录
(ds_train, ds_test), ds_info = tfds.load(
    'rock_paper_scissors',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

print("\n 数据集信息：")
print(f"训练集样本数：{ds_info.splits['train'].num_examples}")
print(f"测试集样本数: {ds_info.splits['test'].num_examples}")
print(f"类别: {ds_info.features['label'].names}")
print(f"图像形状: {ds_info.features['image'].shape}")

#可视化
def show_dataset_images(dataset,num_samples=6):
    plt.figure(figsize=(12,8))
    for i,(image,label) in enumerate(dataset.take(num_samples)):
        plt.subplot(2,3,i+1)
        plt.imshow(image.numpy())
        plt.title(f"{ds_info.features['label'].names[label.numpy()]} (标签: {label.numpy()})")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(tf_data_root, 'data_samples.png'))
    plt.show()

    print("\n📸 显示数据样本...")
    show_dataset_images(ds_train)