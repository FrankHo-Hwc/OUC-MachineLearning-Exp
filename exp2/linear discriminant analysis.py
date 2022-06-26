import numpy as np
import matplotlib.pyplot as plt
import random

def loadData(dir):
    firstdata = np.loadtxt(dir, dtype=np.float32, delimiter=',')
    feature_name = ['R', 'F', 'M', 'T', 'D']
    feature_len = len(feature_name)

    data = firstdata.reshape([firstdata.shape[0], feature_len])
    data = np.array(data)


    training_data = []
    test_data = []
    list_1 = random.sample(range(0, data.shape[0]), 600)
    for j in range(0, data.shape[0]):
        if j in list_1:
            training_data.append(data[j, :])
        else:
            test_data.append(data[j, :])
    training_data = np.array(training_data)
    test_data = np.array(test_data)
    # print(training_data.shape)
    # print(test_data.shape)
    return training_data, test_data



def LDA(x1, x2):
    '''
    :param x1: 类别1  （num,d）
    :param x2: 类别2   (num,d)
    :return: 投影向量w
    '''
    u1 = np.mean(x1, axis=0)
    u2 = np.mean(x2, axis=0)
    Sw = np.dot((x1 - u1).T, (x1 - u1)) + np.dot((x2 - u2).T, (x2 - u2))
    Swmat = np.mat(Sw)
    w = np.dot(Swmat.I, (u1 - u2))
    return w


def LDA_2(x1, x2):
    u1 = np.mean(x1, axis=0)
    u2 = np.mean(x2, axis=0)

    s1 = 0
    s2 = 0
    for i in range(0, len(x1)):
        s1 = s1 + np.dot((x1[i, :] - u1).T, (x1[i, :] - u1))
    for i in range(0, len(x2)):
        s2 = s2 + np.dot((x2[i, :] - u2).T, (x2[i, :] - u2))
    Sw = s1 + s2
    Sw = np.mat(Sw)
    w = Sw.I * (u1 - u2).T
    return w


def predict(w, X):
    y_pred = []
    for sample in X:
        h = sample.dot(w.T)
        y = 1 * (h < 0)
        y_pred.append(y)
    return y_pred

if __name__ == "__main__":

    train_data,test_data = loadData("G:\\machine learning\\exp2\\blood_data.txt")
    y = train_data[:, -1]
    C1 = train_data[y == 0, :-1]
    C2 = train_data[y == 1, :-1]
    w = LDA(C1, C2)
    #预测测试集
    y_pre = predict(w, test_data[:, :-1])
    y_pre = np.asarray(y_pre)
    y_pre = np.reshape(y_pre,(148,1))
    #print(y_pre)
    acc = np.sum(y_pre == test_data[:, -1:]) / test_data[:, -1:].shape[0]
    print("正确率:", acc)

    #画图
    x_values = range(-5, 5)
    # 映射适量的xielv
    rate = w[0, 1] / w[0, 0]
    # 对应垂直于这个直线的斜率
    rate2 = - w[0, 0] / w[0, 1]
    y_values = [rate * x for x in x_values]

    # y = kx+b 中的b
    b = train_data[:, 1] - train_data[:, 0] * rate2
    # b = feature_data[:, 0] - feature_data[:, 1]*rate2
    # 计算出焦点x
    x_ = b / (rate - rate2)
    # 计算出焦点y
    y_ = x_ * rate2 + b

    plt.plot(x_values, y_values)
    plt.scatter(train_data[:, 0], train_data[:, 1], s=10)
    w = np.array(w)

    plt.scatter(x_, y_, s=10)


    # 设置图表标题并给坐标轴加上标签
    plt.title('LDA', fontsize=24)
    plt.xlabel('feature_1', fontsize=14)
    plt.ylabel('feature_2', fontsize=14)

    # 设置刻度标记的大小
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()
