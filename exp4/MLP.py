import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_name):
    data = np.loadtxt(file_name)
    data = pd.DataFrame(data,columns=["0","1","2"])
    #data.loc[data["2"].isin([0]),"2"] = -1
    data = np.asarray(data)
    return data



class layer(object):
    def __init__(self,input_size,output_size,lr):
        self.input_size = input_size
        self.output_size =output_size
        self.lr = lr

    def init_params(self):
        self.w_ = np.random.normal(loc=0.2, scale=0.01, size=(self.input_size, self.output_size))
        self.bias = np.zeros([1, self.output_size])
        #print(self.w_.shape)

    def forward(self,data):
        self.input = data
        # 全连接层的前向传播，计算输出结果
        self.output = self.input.dot(self.w_) + self.bias
        return self.output


    def backforwad(self,top_diff):
        # print(self.input.shape)
        # print(top_diff)
        self.input = self.input.reshape(1, -1)
        self.d_w = np.matmul(self.input.T, top_diff)
        #print(self.d_w)
        self.d_bias = np.matmul(np.ones([1, top_diff.shape[0]]), top_diff)
        bottom_diff = np.matmul(top_diff, self.w_.T)
        return bottom_diff

    def update(self,lr):
        self.w_ = self.w_ - lr * self.d_w
        self.bias = self.bias - lr * self.d_bias


class ReLu(object):
    def forward(self,input):
        self.input = input
        output = np.maximum(0, self.input)
        return output

    def backward(self,top_diff):
        return top_diff * (self.input>=0.)

class Sigmoid(object):
    def forwrd(self,input):
        self.inputs = input
        output = 1/(1 + np.exp(-self.inputs))
        return output

    def backward(self,top_diff):
        index = self.inputs[0][0]
        f = 1 / (1 + np.exp(-index))
        return top_diff * f * (1 - f)


class Loss(object):

    def forward(self,inputs,label):
        self.input = inputs
        self.label = label
        loss = ((inputs - label) ** 2) / 2
        return loss

    def  backward(self):
        #print(self.input)
        #print(self.label)
        bottom_diff = self.input - self.label
        return bottom_diff

dir ='G:\machine learning\exp4\perceptron_data.txt'


def sigmoid(input):
    output = 1 / (1 + np.exp(-input))
    return output

train_data = load_data(dir)

epoch = 2000
lr = 0.01

layer1 = layer(input_size=2,output_size=2,lr=lr)
layer1.init_params()
relu1 = ReLu()
layer2 = layer(input_size=2,output_size=2,lr=lr)
layer2.init_params()
relu2 = ReLu()
layer3 = layer(input_size=2,output_size=1,lr=lr)
layer3.init_params()
sigmoid1 = Sigmoid()
lossfun = Loss()

for i in range(epoch):
    for vec in train_data:
        data = vec[:-1]
        label = vec[-1]
        #前向传播
        h1 =layer1.forward(data=data)
        #print("h1_1",h1)
        h1 = relu1.forward(h1)
        #print("h1_2", h1)
        h2 = layer2.forward(h1)
        #print("h2_1", h2)
        h2 = relu2.forward(h2)
        #print("h2_2", h2)
        h3 = layer3.forward(h2)
        #print("h3_1", h3)
        prob = sigmoid1.forwrd(h3)
        #print("h3_2", h3)
        #计算损失
        loss = lossfun.forward(prob,label)

        #反向传播进行更新
        dloss = lossfun.backward()
        dh3 = sigmoid1.backward(dloss)
        #print("dh3_1",dh3)

        dh3 = layer3.backforwad(dh3)
        #print("dh3_2", dh3)
        layer3.update(lr)
        dh2 = relu2.backward(dh3)
        #print("dh2_1", dh2)
        dh2 = layer2.backforwad(dh2)
        #print("dh2_2", dh2)
        layer2.update(lr)
        dh1 = relu1.backward(dh2)
        #print("dh1_1", dh1)
        dh1 = layer1.backforwad(dh1)
        #print("dh1_2", dh1)
        layer1.update(lr)

    print('Epoch %d,loss: %.6f' % (i+1, loss))



# y = train_data[:, -1]
# C1 = train_data[y == 0, :-1]
# C2 = train_data[y == 1, :-1]
#
#
# x_values = np.arange(-5, 5)
# y_values = -(w[0][0] * x_values + b)/w[0][1]
# y_values = np.squeeze(y_values)
# print(x_values)
# print(y_values)
#
#
#
#
#
#
#
#
# plt.plot(x_values, y_values)
# plt.scatter(C1[:, 0],C1[:, 1], s=100,color='b')
# plt.scatter(C2[:, 0],C2[:, 1], s=100,color='r')
# plt.legend()
# plt.show()

# a = np.asarray([1.5666874,2])
# print(a)
# print(type(a[0]))
# b = sigmoid(a)
# print(b)

