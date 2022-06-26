import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False  # 负号显示


#数据预处理，将数据归一化
class StandardScaler:
    def fit(self,X):
        X = np.asarray(X)
        self.std_ = np.std(X,axis=0)
        self.mean = np.mean(X,axis = 0)

    def transform(self,X):
        return (X-self.mean)/self.std_

    def fit_transform(self ,X):
        self.fit(X)
        return self.transform(X)

#读取数据
def load_data(file_name):
    data = np.loadtxt(file_name)
    data = pd.DataFrame(data,columns=["0","1","2"])
    data.loc[data["2"].isin([0]),"2"] = -1
    data = np.asarray(data)
    return data


class SMO:
    def __init__(self, X, y, C, kernel, alphas, b, errors, user_linear_optim):

        self.X = X  # 训练样本
        self.y = y  # 类别 label
        self.C = C  # r  正则化常量，用于调整（过）拟合的程度
        self.kernel = kernel  # 核函数，实现了两个核函数，线性和高斯（RBF）
        self.alphas = alphas  # 拉格朗日乘子，与样本一一相对
        self.b = b  #bias 标量，偏移量# self.b = 0
        self.errors = errors  # E  用于存储alpha值实际与预测值得差值，与样本数量一一相对
        self.m, self.n = np.shape(self.X)  # 训练样本的个数和每个样本的w维度
        self.user_linear_optim = user_linear_optim  # 判断模型是否使用线性核函数
        self.w = np.zeros(self.n)  # 初始化权重w的值，主要用于线性核函数


def linear_kernel(x, y, b = 1):
    #线性核函数
    end = np.dot(x, y.T) - b
    return end


def gaussian_kernel(x, y, sigma=1):
    # 高斯核函数
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        end = np.exp(-(np.linalg.norm(x - y, 2)) ** 2 / (2 * sigma ** 2))
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
        end = np.exp(-(np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma ** 2))
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        end = np.exp(-(np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2) / (2 * sigma ** 2))
    return end



def decision_function_plot(alphas, y, kernel, x1, x2, b):
    end = np.dot(alphas * y, kernel(x1, x2)) - b

    return end

#计算向量机方程的输出值
def decision_f_output(model, i):
    if model.user_linear_optim:
        f_xi = float(np.dot(model.w.T, model.X[i])) - model.b
        return f_xi
    else:
        for j in range(model.m):
            f_xi = np.sum(model.alphas[j] * model.y[j] * model.kernel(model.X[j], model.X[i])) - model.b
        return f_xi

#计算误差
def get_error(model, i):
    if 0 < model.alphas[i] < model.C:  # 为支持向量时
        return model.errors[i]
    else:
        return decision_f_output(model, i) - model.y[i]

#选择一对alpha的办法，先选alpha2再选alpha1.
def fit(model):
    if_Success = 0  # 判断优化是否成功0代表失败
    examineAll = 1  # 所有的alpha是否都遍历过了
    epoch = 0  # 计数器，记录优化时的循环次数
    loopnum1 = 0
    loopnum2 = 0
    while (if_Success > 0) or (examineAll):
        # examineAll=1
        # 当numChanged = 0 and examineAll = 0时 循环退出
        if_Success = 0
        '''需要设置终止条件，不能一直循环下去。
        因为一般情况下，numChanged是大于0的，只有少数情况才是这样
        最下面的判断就会无限满足，设置迭代次数让它停止循环
        '''
        if epoch == 24:
            break
        epoch = epoch + 1  # 记录总循环次数
        #这个时候是一开始随机选择alpha2进行更新
        if examineAll:
            loopnum1 = loopnum1 + 1  # 记录顺序一个一个选择alpha2时的循环次数
            for i in range(len(model.alphas)):
                examine_result, model = examine_example(i, model,epoch)
                # 优化成功examine_result为1不成功为0
                if_Success += examine_result
                # 非极端情况下numChanged>0
        else:  # 上面if里m(m-1)执行完的后执行
            loopnum2 = loopnum2 + 1
            #当处于极端情况时就使用此处循环
            for i in np.where((model.alphas != 0) & (model.alphas != model.C))[0]:
                # a中不等于0和C的所有元素
                # 即支持向量上的点
                examine_result, model = examine_example(i, model,epoch)
                if_Success += examine_result
        # 无论如何怎样到这里时examineAll始终是1
        if examineAll == 1:
            examineAll = 0
        elif if_Success == 0:  # 防止极端情况
            examineAll = 1
            # 优化不成功numchanged=0继续循环
    return model


# 由alpha2确定alpha1,上面也提到了
def examine_example(i2, model,iteration):
    y2 = model.y[i2]  # 取y2
    alph2 = model.alphas[i2]  # 取old_a2
    E2 = get_error(model, i2)  # 去差值
    r2 = E2 * y2
    # 下面条件之一满足，进入if开始找第二个alpha，送到take_step进行优化
    if ((r2 < -tol and alph2 < model.C) or (r2 > tol and alph2 > 0)):

        for i1 in np.roll(np.where((model.alphas != 0) & (model.alphas != model.C))[0],
                          np.random.choice(np.arange(model.m))):

            step_result, model = take_step(i1, i2, model)
            if step_result:  # 优化成功
                print("iteration:{},changed pairs of alpha:a{},a{}".format(iteration,i1,i2))
                return 1, model
        # a2_old确定的情况下，循环所有(m-1) alphas, 随机选择起始点优化所有alpha
        for i1 in np.roll(np.arange(model.m), np.random.choice(np.arange(model.m))):
            step_result, model = take_step(i1, i2, model)
            if step_result:  # 优化成功
                print("iteration:{},changed pairs of alpha:a{},a{}".format(iteration, i1, i2))
                return 1, model
    #如果都满足KKT条件，不需要优化就正常退出
    return 0, model  # 退出0表示不需要优化


def take_step(i1, i2, model):
    if i1 == i2:
        return 0, model
    alph1 = model.alphas[i1]  # a1_old
    alph2 = model.alphas[i2]  # a2_old
    y1 = model.y[i1]
    y2 = model.y[i2]
    E1 = get_error(model, i1)
    E2 = get_error(model, i2)
    s = y1 * y2
    #根据y1和y2选择上界和下界H和L
    if (y1 != y2):

        L = max(0, alph2 - alph1)
        H = min(model.C, model.C + alph2 - alph1)
    elif (y1 == y2):

        L = max(0, alph1 + alph2 - model.C)
        H = min(model.C, alph1 + alph2)
    if (L == H):
        return 0, model
    #核函数
    k11 = model.kernel(model.X[i1], model.X[i1])
    k12 = model.kernel(model.X[i1], model.X[i2])
    k22 = model.kernel(model.X[i2], model.X[i2])
    # 计算eta
    eta = k11 + k22 - 2 * k12
    #此处为裁剪变量
    if (eta > 0):
        a2 = alph2 + y2 * (E1 - E2) / eta

        if L < a2 < H:
            a2 = a2
        elif (a2 <= L):
            a2 = L
        elif (a2 >= H):
            a2 = H

    if (np.abs(a2 - alph2) < eps * (a2 + alph2 + eps)):
        return 0, model
    # 更新部分
    # 更新b
    a1 = alph1 + s * (alph2 - a2)  # 根据新a2计算新a1
    b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + model.b  # 更新 bias b的值
    b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + model.b
    if 0 < a1 and a1 < C:
        b_new = b1
    elif 0 < a2 and a2 < C:
        b_new = b2
    else:
        b_new = (b1 + b2) * 0.5
    model.b = b_new  # 更新b
    #如果 0 < a_1 < c 选b_1
    #如果 0 <a_2 < c 选b_2
    #都不满足就取二者平均值
    #线性核函数的更新方式
    if model.user_linear_optim:
        model.w = model.w + y1 * (a1 - alph1) * model.X[i1] + y2 * (a2 - alph2) * model.X[i2]
    #将更新a1, a2的值存入model中
    model.alphas[i1] = a1
    model.alphas[i2] = a2
    # 更新误差
    model.errors[i1] = 0
    model.errors[i2] = 0
    # 更新差值 主要针对0<a<c的情况
    for i in range(model.m):
        if 0 < model.alphas[i] < model.C:
            model.errors[i] += y1 * (a1 - alph1) * model.kernel(model.X[i1], model.X[i]) + \
                               y2 * (a2 - alph2) * model.kernel(model.X[i2], model.X[i]) + model.b - b_new

    return 1, model


def plot_decision_boundary(model, resolution=100, colors=('b', 'k', 'r'), levels=(-1, 0, 1)):
    plt.title('训练结果')
    xrange = np.linspace(model.X[:, 0].min(), model.X[:, 0].max(), resolution)
    yrange = np.linspace(model.X[:, 1].min(), model.X[:, 1].max(), resolution)

    grid = [[decision_function_plot(model.alphas, model.y, model.kernel, model.X,
                                    np.array([xr, yr]), model.b) for xr in xrange] for yr in yrange]

    grid = np.array(grid).reshape(len(xrange), len(yrange))

    plt.contour(xrange, yrange, grid, levels=levels, linewidths=(1, 1, 1),
                linestyles=('--', '-', '--'), colors=colors)

    plt.scatter(model.X[model.y == 1, 0], model.X[model.y == 1, 1], label='样本1', c='r', alpha=0.25)
    plt.scatter(model.X[model.y == -1, 0], model.X[model.y == -1, 1], label='样本2', c='b', alpha=0.25)
    mask = np.round(model.alphas, decimals=2) != 0.0
    #支持向量
    plt.scatter(model.X[mask, 0], model.X[mask, 1], label='支持向量', c='y', alpha=0.8)
    plt.legend()
    return grid





dir ='G:\machine learning\exp5\data.txt'
train_data = load_data(dir)
X_train = train_data[:,:-1]
y = train_data[:,-1]
#print(train_data.shape)
scaler = StandardScaler()
X = scaler.fit_transform(X_train)
# print(np.ndim(X))

C = 1
m = len(X)
#print(m)
#每个样本都有一个alpha
initial_alphas = np.zeros(m)
# print(len(initial_alphas))
#print(initial_alphas.shape[0])

initial_b = 0.0
tol = 0.011  #差值矩阵的容忍差
eps = 0.011  #阿尔法的容忍差

model = SMO(X, y, C, linear_kernel, initial_alphas, initial_b, np.zeros(m), user_linear_optim=True)

initial_error = decision_function_plot(model.alphas, model.y, model.kernel, model.X, model.X, model.b) - model.y

model.errors = initial_error
np.random.seed(0)
#设置随机数初始化变量
output = fit(model)
print("parameters:")
print("w:{}".format(output.w))
print("b:{}".format(output.b))

grid = plot_decision_boundary(output)
plt.show()