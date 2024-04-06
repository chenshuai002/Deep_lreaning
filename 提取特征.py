import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 定义一个函数来提取图像的特征并进行降维
def extract_and_reduce_features(image_dir, features_dir, reduced_dim=100):
    features = []  # 用于保存所有图像的特征
    labels = []    # 用于保存图像对应的标签（目录名）

    # 遍历目录中的子目录，每个子目录代表一个人的图像
    for person_dir in os.listdir(image_dir):
        person_path = os.path.join(image_dir, person_dir)

        if not os.path.isdir(person_path):
            continue

        # 遍历每个人的图像文件
        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)

            # 使用OpenCV加载图像
            image = cv2.imread(image_path)

            # 检查图像是否成功加载
            if image is None:
                print(f"Error loading image: {image_path}")
                continue

            # 提取图像的直方图作为特征向量（示例特征提取方法）
            hist = cv2.calcHist([image], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256])

            # 将特征保存为Numpy数组
            feature = np.array(hist).flatten()

            # 添加特征和标签到列表
            features.append(feature)
            labels.append(person_dir)

    # 将特征和标签转换为Numpy数组
    features = np.array(features)
    labels = np.array(labels)

    # 创建用于保存特征的子目录
    os.makedirs(features_dir, exist_ok=True)

    # 保存特征和标签到文件
    np.save(os.path.join(features_dir, 'features.npy'), features)
    np.save(os.path.join(features_dir, 'labels.npy'), labels)

    # 标准化特征
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 使用PCA进行降维
    pca = PCA(n_components=reduced_dim)
    reduced_features = pca.fit_transform(features)

    # 保存降维后的特征
    np.save(os.path.join(features_dir, 'reduced_features.npy'), reduced_features)

    return reduced_features, labels

# 示例用法
image_dir = "F:\ziyuan\photo\lfw"  # 包含图像文件的目录
features_dir = "features_output_directory_model"  # 用于保存特征的目录
reduced_features, labels = extract_and_reduce_features(image_dir, features_dir, reduced_dim=100)

# 打印提取和降维后的特征和标签的形状
print("Reduced Features shape:", reduced_features.shape)
print("Labels shape:", labels.shape)
