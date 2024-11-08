import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
 
# 数据读取
train_data = pd.read_csv('mnist_01_train.csv')
test_data = pd.read_csv('mnist_01_test.csv')

# 特征和标签
X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# 线性核SVM
linear_svm = svm.SVC(kernel='linear', probability=True)
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)
linear_accuracy = accuracy_score(y_test, y_pred_linear)

# 高斯核SVM
gaussian_svm = svm.SVC(kernel='rbf', probability=True)
gaussian_svm.fit(X_train, y_train)
y_pred_gaussian = gaussian_svm.predict(X_test)
gaussian_accuracy = accuracy_score(y_test, y_pred_gaussian)

# 输出SVM性能结果
print(f'Linear SVM Accuracy: {linear_accuracy:.6f}')
print(f'Gaussian SVM Accuracy: {gaussian_accuracy:.6f}')
