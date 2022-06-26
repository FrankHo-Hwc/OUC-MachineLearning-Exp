import numpy as np
import pandas as pd
from  random import randrange
import random
import matplotlib
import matplotlib.pyplot as plt

#读入数据
def openreadtxt(file_name):
    # data = []
    # file = open(file_name,'r')  #打开文件
    # file_data = file.readlines() #读取所有行
    # file_data = np.array(file_data)
    # print(file_data.shape)
    data = pd.read_csv(file_name,sep='\s+',names=["CRIM","ZN", "INDUS", "CHAS", "NOX" ,"RM","AGE" ,"DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"])
    return data


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


#sgd版本
class LinearRegression:
    def __init__(self, lr,epoch):
        #epoch:训练轮数
        #lr:学习率
        self.lr =lr
        self.epoch = epoch

    def fit(self,X,y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.loss_ = []
        self.w_ = np.zeros(1 + X.shape[1])

        for i in range(self.epoch):
            y_hat = np.dot(X,self.w_[1:])+self.w_[0]
            loss =  y - y_hat
            self.w_[0] += self.lr * np.sum(loss)
            self.w_[1:] += self.lr * np.dot(X.T,loss)
            self.loss_.append(np.sum((loss**2)/2))
            print("epochs:{},loss:{}".format(i+1,np.sum((loss**2)/2)))


    def predict(self,X):
        X = np.asarray(X)
        result = np.dot(X,self.w_[1:])+self.w_[0]
        return result

lr = 0.0005
epochs = 50

dir = 'G:\machine learning\housing_data.txt'
data = openreadtxt(dir)
model = LinearRegression(lr,epochs)
t = data.sample(len(data),random_state=0)
#将数据集划分为训练集和测试集
train_x = t.iloc[:450,:-1]
train_y = t.iloc[:450,-1]
test_x = t.iloc[450:,:-1]
test_y = t.iloc[450:,-1]

s = StandardScaler()
train_x = s.fit_transform(train_x)
test_x = s.fit_transform(test_x)
s2 = StandardScaler()
train_y = s2.fit_transform(train_y)
test_y = s2.fit_transform(test_y)

model.fit(train_x,train_y)
result = model.predict(test_x)

print("parameters:{},square loss:{}".format(model.w_,np.mean((result - test_y)**2)))

plt.figure(figsize=(10,10))

plt.plot(result,'ro-',label='predict')

plt.plot(test_y.values,'go--',label ='real')

plt.xlabel('labels')

plt.ylabel('price')

plt.legend()
plt.show()

plt.plot(range(1,epochs+1),model.loss_,'o-')
plt.xlabel('epochs')

plt.ylabel('Loss')
plt.legend()
plt.show()