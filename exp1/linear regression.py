import numpy as np
import pandas as pd
from  random import randrange
import random
import matplotlib
import matplotlib.pyplot as plt
def openreadtxt(file_name):
    # data = []
    # file = open(file_name,'r')  #打开文件
    # file_data = file.readlines() #读取所有行
    # file_data = np.array(file_data)
    # print(file_data.shape)
    data = pd.read_csv(file_name,sep='\s+',names=["CRIM","ZN", "INDUS", "CHAS", "NOX" ,"RM","AGE" ,"DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"])
    return data






#训练
class LinearRegression:
    def  fit(self,X,y):
        X = np.asmatrix(X.copy())
        y = np.asmatrix(y).reshape(-1,1)

        self.w_ = (X.T * X).I * X.T*y

    def predict(self,X):
        X = np.asmatrix(X.copy())
        result = X *self.w_
        return np.array(result).ravel()

#损失函数
def loss(y,y_hat):
    print('Loss:{}'.format(np.mean((y_hat - y )**2)/2))



dir = 'G:\machine learning\housing_data.txt'
data = openreadtxt(dir)
#print(data)
data.info()
t = data.sample(len(data),random_state=0)
#将数据集划分为训练集和测试集
train_x = t.iloc[:450,:-1]
train_y = t.iloc[:450,-1]
test_x = t.iloc[450:,:-1]
test_y = t.iloc[450:,-1]
model = LinearRegression()
model.fit(train_x,train_y)
result = model.predict(test_x)
print('parameters:{}'.format(model.w_))
loss(test_y,result)

plt.figure(figsize=(10,10))

plt.plot(result,'ro-',label='预测值')

plt.plot(test_y.values,'go--',label ='真实值')

plt.xlabel('样本序号')

plt.ylabel('房价')

plt.legend()
plt.show()