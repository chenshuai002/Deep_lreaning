import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

# 步骤2: 模型训练
def train_model(features_dir, model_save_path, max_iterations=20):
    # 加载特征和标签
    features = np.load(os.path.join(features_dir, 'features.npy'))
    labels = np.load(os.path.join(features_dir, 'labels.npy'))

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 初始化记录正确率和迭代次数的列表
    iteration_values = []
    accuracy_values = []

    for iteration in range(1, max_iterations + 1):
        # 训练分类模型（使用支持向量机SVM）
        model = SVC(kernel='linear', max_iter=iteration)
        model.fit(X_train, y_train)

        # 预测测试集
        y_pred = model.predict(X_test)

        # 计算识别正确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Iteration {iteration}, Accuracy: {accuracy}")

        # 记录正确率和迭代次数
        iteration_values.append(iteration)
        accuracy_values.append(accuracy)

    # 保存训练好的模型到指定路径
    joblib.dump(model, model_save_path)

    # 绘制正确率曲线图
    plt.figure(figsize=(8, 6))
    plt.plot(iteration_values, accuracy_values, marker='o')
    plt.title('Accuracy vs Training Iterations')
    plt.xlabel('Training Iterations')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('accuracy_vs_iterations_plot.png')
    plt.show()

    return model

# 主流程
features_dir = "features_output_directory_model"
model_save_path = "model.pkl"  # 模型保存路径

# 步骤2: 模型训练并保存
model = train_model(features_dir, model_save_path, max_iterations=20)
