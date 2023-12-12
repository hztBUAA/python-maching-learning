import pandas as pd
import torch
import numpy as np
import cv2
import os
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


def plot_metrics(loss_rates, acc_train, acc_val, epochs):
    # 绘制损失率图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_rates, label='Loss Rate')
    plt.title('Loss Rate Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Rate')
    plt.legend()

    # 绘制正确率图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_train, label='Train Accuracy')
    plt.plot(epochs, acc_val, label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 展示图表
    plt.tight_layout()
    plt.show()


def DataProcess():
    # 数据预处理
    # 将label与人脸数据作拆分
    """
    在这段代码中，使用 [['label']] 和 [['feature']] 的双层嵌套是为了选择特定的列，并且返回的数据结构是 DataFrame 而不是 Series。
单层的 ['label'] 或 ['feature'] 会返回一个 Pandas Series 对象，而双层的 [['label']] 和 [['feature']] 会返回一个 Pandas DataFrame 对象。这是因为双层嵌套表示你选择的是一个列的子集，而不是一个单独的列。
在这个例子中，[['label']] 和 [['feature']] 的目的是确保 df_y 和 df_x 是 DataFrame 对象，即使选择的是单个列。这对于后续的数据处理和操作来说可能更方便，因为 DataFrame 提供了更多的灵活性和功能，而不仅仅是存储数据的一列。
    """

    path = 'D:\\code\\python_p\\train.csv'  # 文件路径
    df = pd.read_csv(path)  # pd阅读器打开csv文件
    df = df.fillna(0)  # 空值填充

    # 分别提取标签和特征数据
    df_y = df[['label']]
    df_x = df[['feature']]

    # 将label,feature数据写入csv文件
    df_y.to_csv('label.csv', index=False, header=False)  # 不保存索引(0-N),不保存列名('label')
    df_x.to_csv('data.csv', index=False, header=False)

    # 指定存放图片的路径
    path = 'D:\\code\\python_p\\images'
    # 读取像素数据
    data = np.loadtxt('data.csv')

    # 按行取数据
    for i in range(2000):  # 按行读取
        face_array = data[i, :].reshape((48, 48))  # reshape 转成图像矩阵给cv2处理
        if i < 1800:
            cv2.imwrite(path + '//train_data//' + '{0}.jpg'.format(i), face_array)  # csv文件转jpg写图片
        else:
            cv2.imwrite(path + '//test_data//' + '{0}.jpg'.format(i), face_array)  # csv文件转jpg写图片


def data_label(path):
    # 读取label文件
    df_label = pd.read_csv('label.csv', header=None)

    # 查看文件夹下所有文件
    files_dir = os.listdir(path)

    # 用于存放图片名
    path_list = []

    # 用于存放图片对应的label
    label_list = []

    # 遍历该文件夹下的所有文件
    for files_dir in files_dir:
        # 如果某文件是图片,则其文件名以及对应的label取出,分别放入path_list和label_list这两个列表中
        if os.path.splitext(files_dir)[1] == ".jpg":  # 路径切割,将文件名和后缀名作切割后保存为列表形式
            path_list.append(files_dir)  # 如果是.jpg文件就添加入path_list 路径列表
            index = int(os.path.splitext(files_dir)[0])  # 将图片文件名按数值类型转存
            label_list.append(df_label.iat[index, 0])  # 将文件编号填入

    # 将两个列表写进dataset.csv文件
    path_s = pd.Series(path_list)
    label_s = pd.Series(label_list)
    df = pd.DataFrame()
    df['path'] = path_s
    df['label'] = label_s
    df.to_csv(path + '\\dataset.csv', index=False, header=False)  # df保存,命名为dataset.csv


# 验证模型在验证集上的正确率
def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size)
    result, num = 0.0, 0
    for images, labels in val_loader:
        pred = model.forward(images)
        pred = np.argmax(pred.data.numpy(), axis=1)
        labels = labels.data.numpy()
        result += np.sum((pred == labels))
        num += len(images)

    acc = result / num
    return acc


class FaceDataset(data.Dataset):  # 父类继承,注意继承dataset父类必须重写getitem,len否则报错.
    # 初始化
    def __init__(self, root):  # root为train,val文件夹地址
        super(FaceDataset, self).__init__()  # 调用父类的初始化函数
        self.root = root
        # 读取data - label 对照表中的内容
        df_path = pd.read_csv(root + '\\dataset.csv', header=None, usecols=[0])
        df_label = pd.read_csv(root + '\\dataset.csv', header=None, usecols=[1])
        # 将其中内容放入numpy, 方便后期索引
        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0]

    # 读取某幅图片, item为索引号
    def __getitem__(self, item):
        face = cv2.imread(self.root + '\\' + self.path[item])  # 读取图片

        # 读取单通道灰度图
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # 单通道=灰度,三通道-RGB彩色

        # 高斯模糊
        # face_Gus = cv2.GaussianBlur(face_gray, (3,3), 0)

        # 直方图均衡化
        face_hist = cv2.equalizeHist(face_gray)

        # 像素值标准化,0-255的像素范围转成0-1范围来描述
        face_normalized = face_hist.reshape(1, 48, 48) / 255.0

        # 用于训练的数据需要为tensor类型
        face_tensor = torch.from_numpy(face_normalized)  # 将numpy中的ndarray转换成pytorch中的tensor
        face_tensor = face_tensor.type('torch.FloatTensor')  # Tensor转FloatTensor
        label = self.label[item]
        return face_tensor, label

    # 获取数据集样本个数
    def __len__(self):
        return self.path.shape[0]


class FaceCNN(nn.Module):
    # 初始化网络结构
    def __init__(self):
        super(FaceCNN, self).__init__()

        # 第一次卷积， 池化
        self.conv1 = nn.Sequential(
            # 输入通道数in_channels，输出通道数(即卷积核的通道数)out_channels，
            # 卷积核大小kernel_size，步长stride，对称填0行列数padding
            # input:(bitch_size, 1, 48, 48),
            # output:(bitch_size, 64, 48, 48), (48-3+2*1)/1+1 = 48
            # 卷积层
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),

            # 数据归一化处理，使得数据在Relu之前不会因为数据过大而导致网络性能不稳定
            # 做归一化让数据形成一定区间内的正态分布
            # 不做归一化会导致不同方向进入梯度下降速度差距很大
            nn.BatchNorm2d(num_features=64),  # 归一化可以避免出现梯度散漫的现象，便于激活。
            nn.RReLU(inplace=True),  # 激活函数

            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大值池化# output(bitch_size, 64, 24, 24)
        )

        # 第二次卷积， 池化
        self.conv2 = nn.Sequential(
            # input:(bitch_size, 64, 24, 24), output:(bitch_size, 128, 12, 12),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 第三次卷积， 池化
        self.conv3 = nn.Sequential(
            # input:(bitch_size, 128, 12, 12), output:(bitch_size, 256, 12, 12),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            # 最后一层不需要添加激活函数
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.RReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1024),
            nn.RReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=256),
            nn.RReLU(inplace=True),

            nn.Linear(in_features=256, out_features=7),
        )

    # 前向传播
    # 使用sequential模块后无需再在forward函数中添加激活函数
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # 数据扁平化
        x = x.view(x.shape[0], -1)  # 输出维度，-1表示该维度自行判断
        y = self.fc(x)
        return y


# 训练模型
def train(model, train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay):
    # 如果传入的模型为None，则实例化一个新模型
    if model is None:
        model = FaceCNN()
    # 载入数据并分割batch
    train_loader = data.DataLoader(train_dataset, batch_size)
    # # 构建模型
    # model = FaceCNN()
    # 损失函数
    loss_function = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    # 学习率衰减
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 10, gamma= 0.8)

    # 逐轮训练
    for epoch in range(epochs):
        # 记录损失值
        loss_rate = 0
        # scheduler.step()
        # 注意dropout网络结构下训练和test模式下是不一样的结构
        model.train()  # 模型训练，调用Modlue类提供的train()方法切换到train状态

        # for images, labels in train_loader:
        #     # 梯度清零
        #     optimizer.zero_grad()
        #     # 前向传播
        #     output = model.forward(images)
        #     # 误差计算
        #     loss_rate = loss_function(output, labels)
        #     # 误差的反向传播
        #     loss_rate.backward()
        #     # 更新参数
        #     optimizer.step()

        # 梯度清零
        optimizer.zero_grad()
        # 随机选择一个batch的数据
        images, labels = next(iter(train_loader))
        # 前向传播
        output = model.forward(images)
        # 误差计算
        loss_rate = loss_function(output, labels)
        # 误差的反向传播
        loss_rate.backward()
        # 更新参数
        optimizer.step()

        # 打印每轮的损失
        print('After {} epochs , '.format(epoch + 1))
        print('After {} epochs , the loss_rate is : '.format(epoch + 1), loss_rate.item())
        # 在每轮结束后记录损失率和正确率
        loss_rates.append(loss_rate.item())
        acc_train = validate(model, train_dataset, batch_size)
        acc_val = validate(model, val_dataset, batch_size)
        acc_train_list.append(acc_train)
        acc_val_list.append(acc_val)
        # if epoch % 5 == 0:
        #     model.eval()  # 模型评估,切换到test状态继续执行
        #     acc_train = validate(model, train_dataset, batch_size)
        #     acc_val = validate(model, val_dataset, batch_size)
        #     print('After {} epochs , the acc_train is : '.format(epoch + 1), acc_train)
        #     print('After {} epochs , the acc_val is : '.format(epoch + 1), acc_val)

    return model


def main():
    # 数据预处理
    # 可以添加一个检查数据预处理是否已经执行的条件
    images_dir = 'D:\\code\\python_p\\images'

    # if not os.path.exists(images_dir) or not os.listdir(images_dir):
    # DataProcess()
    #
    # # if not os.path.exists('D:\\code\\python_p\\images\\train_data\\dataset.csv'):
    # train_path = 'D:\\code\\python_p\\images\\train_data'
    # data_label(train_path)
    #
    # # 检查数据预处理是否已经执行
    # # if not os.path.exists('D:\\code\\python_p\\images\\test_data\\dataset.csv'):
    # val_path = 'D:\\code\\python_p\\images\\test_data'
    # data_label(val_path)
    #
    # # 数据集的使用
    train_dataset = FaceDataset(root='D:\\code\\python_p\\images\\train_data')
    val_dataset = FaceDataset(root='D:\\code\\python_p\\images\\test_data')

    # 加载已保存的模型
    model = None
    # if os.path.exists('model_net1.pk1'):
    # model = torch.load('model_net1.pk1')

    # 继续训练
    model = train(model, train_dataset, val_dataset, batch_size=128, epochs=200, learning_rate=0.1, wt_decay=0)

    # 保存模型
    torch.save(model, 'model_net1.pk1')


if __name__ == '__main__':
    loss_rates = []  # 记录每轮的损失率
    acc_train_list = []  # 记录每轮的训练集正确率
    acc_val_list = []  # 记录每轮的测试集正确率

    main()

    # 绘制图表
    epochs = list(range(1, 201))  # 你的迭代次数范围
    plot_metrics(loss_rates, acc_train_list, acc_val_list, epochs)
