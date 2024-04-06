
import cv2
import numpy as np
import joblib

# 步骤1: 加载模型
model = joblib.load('model.pkl')

# 步骤2: 准备测试图像并提取特征
def extract_features_from_image(image_path):
    # 使用OpenCV加载图像
    image = cv2.imread(image_path)

    # 检查图像是否成功加载
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    # 提取图像的直方图作为特征向量（示例特征提取方法）
    hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

    # 将特征保存为Numpy数组
    feature = np.array(hist).flatten()

    return feature

# 步骤3: 进行识别
def recognize_faces(test_image_path):
    test_feature = extract_features_from_image(test_image_path)

    if test_feature is not None:
        predicted_label = model.predict([test_feature])[0]

        # 步骤4: 显示结果
        print(f"Predicted Label: {predicted_label}")
    else:
        print("Feature extraction failed for the test image.")

# 用于测试的图像路径
test_image_path = 'F:\pycharm\data\ceshiji\Aaron_Eckhart\Aaron_Eckhart_0001.jpg'

# 进行人脸识别
recognize_faces(test_image_path)
