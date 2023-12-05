from lr_utils import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import mpmath


def sigmoid(z):
    a = 1./(1+np.exp(-z))
    return a


def initialize_parameters(dim):
    w = np.zeros(shape=(dim, 1), dtype=np.float32)
    b = 0

    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]

    # Forward propagation
    A = sigmoid(np.dot(w.T, X) + b)

    epsilon = 1e-15
    A = np.maximum(A, epsilon)
    A = np.minimum(A, 1 - epsilon)

    cost = (-1. / m) * np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A)), axis=1)

    # Backward propagation
    dw = (1. / m) * np.dot(X, ((A - Y).T))
    db = (1. / m) * np.sum(A - Y, axis=1)

    cost = np.squeeze(cost)

    return dw, db, cost


def optimized(w, b, X, Y, train_times, learning_rate):
    costs = []

    for i in range(train_times):

        dw, db, cost = propagate(w, b, X, Y)

        # 参数迭代
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录成本函数
        if i % 100 == 0:
            costs.append(cost)

    return w, b, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    # 确保矩阵维数匹配
    w = w.reshape(X.shape[0], 1)

    # 计算 Logistic 回归
    A = sigmoid(np.dot(w.T, X) + b)

    # 如果结果大于0.5，打上标签"1"，反之记录为"0"
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    # 确保一下矩阵维数正确
    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def TestMyModel(X_train, Y_train, X_test, Y_test, train_times=100, learning_rate=0.005):
    # 初始化参数
    w, b = initialize_parameters(X_train.shape[0])

    # 开始梯度下降训练
    w, b, costs = optimized(w, b, X_train, Y_train, train_times, learning_rate)

    # 训练完成，测试一下效果
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # 输出效果
    print("Accuracy on train_set: " + str(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100) + "%")
    print("Accuracy on test_set: " + str(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100) + "%")
    print(Y_prediction_test)
    return w, b, costs, Y_prediction_test


train_X_orig, train_Y, test_X_orig, test_Y, classes = load_dataset()

# print("Shape of train_set_x_orig: " + str(train_X_orig.shape))
# print("Shape of train_set_y: " + str(train_Y.shape))
# print("Shape of test_set_x_orig: " + str(test_X_orig.shape))
# print("Shape of test_set_y: " + str(test_Y.shape))
# 显示训练集中的第 index 个训练图像
# index = 1
# plt.imshow(train_X_orig[index])
# plt.show()
# print("y = " + str(train_Y[0, index]) + ", it's a '" + str(classes[int(train_Y[0, index])]) + "' picture.")


train_set_x_flatten = train_X_orig.reshape(train_X_orig.shape[0], -1).T
test_set_x_flatten = test_X_orig.reshape(test_X_orig.shape[0], -1).T

train_set_x = 1.0 * train_set_x_flatten / 255
test_set_x = 1.0 * test_set_x_flatten / 255
w, b, costs, Y_prediction_test = TestMyModel(train_set_x, train_Y, test_set_x, test_Y, 20000, 0.002)
plot_costs = np.squeeze(costs)
print(costs)
plt.plot(plot_costs)
plt.show()
