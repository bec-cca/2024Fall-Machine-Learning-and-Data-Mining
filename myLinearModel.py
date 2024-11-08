import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 加载数据
train_data = pd.read_csv('mnist_01_train.csv')
test_data = pd.read_csv('mnist_01_test.csv')

# 分离特征和标签
X_train = train_data.iloc[:, 1:].values  # 输入特征
y_train = train_data.iloc[:, 0].values  # 标签
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Sigmoid 函数
def sigmoid(x):
    x = np.clip(x, -500, 500)  # 限制 x 的范围，避免过大或过小的值导致溢出
    return 1 / (1 + np.exp(-x))

# 损失和梯度计算：Hinge Loss
def hinge_loss(w, b, X, y, lambda_reg=0.001):
    margin = 1 - y * (np.dot(X, w) + b)  # 批量处理
    loss = np.mean(np.maximum(0, margin)) + lambda_reg * np.sum(w ** 2)  # 正则化项
    # 梯度
    dw = - np.mean((y * (margin > 0))[:, None] * X, axis=0) + 2 * lambda_reg * w
    db = - np.mean(y * (margin > 0))
    return loss, dw, db

# 损失和梯度计算：Cross-Entropy Loss
def cross_entropy_loss(w, b, X, y, lambda_reg=0.001):
    m = X.shape[0]
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)
    loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) + lambda_reg * np.sum(w ** 2)  # 正则化项
    # 梯度
    dw = np.dot(X.T, (y_hat - y)) / m + 2 * lambda_reg * w
    db = np.sum(y_hat - y) / m
    return loss, dw, db

# 梯度下降训练模型
def train_model(X, y, loss_function, learning_rate=0.01, num_iterations=200, lambda_reg=0.001):
    w = np.zeros(X.shape[1])
    b = 0
    losses = []  # 用于保存每次迭代的损失
    for i in range(num_iterations):
        loss, dw, db = loss_function(w, b, X, y, lambda_reg)
        w -= learning_rate * dw
        b -= learning_rate * db
        losses.append(loss)  # 保存每次迭代的损失值
        if loss_function == hinge_loss:
            loss_name = 'Hinge Loss'
        elif loss_function == cross_entropy_loss:
            loss_name = 'Cross-Entropy Loss'
        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1}: {loss_name} = {loss}")
    return w, b, losses  # 返回损失值

def predict(X, w, b, loss_function='hinge'):
    if loss_function == 'hinge':
        return np.sign(np.dot(X, w) + b)
    elif loss_function == 'cross_entropy':
        return (sigmoid(np.dot(X, w) + b) >= 0.5).astype(int)

# 计算准确率
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# 训练模型：Hinge Loss
w_hinge, b_hinge, losses_hinge = train_model(X_train, y_train * 2 - 1, hinge_loss, learning_rate=0.01, num_iterations=300)

# 训练模型：Cross-Entropy Loss
w_ce, b_ce, losses_ce = train_model(X_train, y_train, cross_entropy_loss, learning_rate=0.01, num_iterations=300)

# 预测测试集标签
y_pred_hinge = predict(X_test, w_hinge, b_hinge, 'hinge')
y_pred_ce = predict(X_test, w_ce, b_ce, 'cross_entropy')

acc_hinge = accuracy(y_test, (y_pred_hinge + 1) / 2)
acc_ce = accuracy(y_test, y_pred_ce)

print(f"Accuracy (Hinge Loss): {acc_hinge:.6f}")
print(f"Accuracy (Cross-Entropy Loss): {acc_ce:.6f}")

# 绘制训练过程中的损失曲线
plt.figure(figsize=(10, 6))
plt.plot(losses_hinge, label='Hinge Loss', color='blue')
plt.plot(losses_ce, label='Cross-Entropy Loss', color='red')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss vs Iterations')
plt.legend()
plt.grid(True)
plt.show()