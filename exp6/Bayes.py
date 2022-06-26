import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


Label_num = 10

train_dir = 'G:\\machine learning\\exp6\\traindata'
train_num = [189,198,195,199,186,187,195,201,180,204]


test_dir  = 'G:\\machine learning\\exp6\\testdata'
test_num = [87,97,92,85,114,108,87,96,91,89]

'''
m是超参数
由于我们是通过训练的数据计算出来的一系列概率，
如果数据的某项特征在训练数据中从未出现，而需要预测标签的数据出现了这种特征，那
通过贝叶斯分类器计算出来的先验概率就会为0，从而导致分类器出现问题。
此处就是采用m-估计来进行优化
'''
P = 0.5
m = 3

Train_data = []
Train_label = []


Test_data = []
Test_label = []


#读入训练集
def load_Train_Data(dir):
    #cnt = 0
    train_data = []
    train_label = []
    for i in range(0,10): #分数字读
        for j in range(0,train_num[i]): #分数字的样本读
            train_label.append(i)
            file_dir = dir+ '\\'+ str(i) +'_' + str(j) +'.txt'
            data = np.loadtxt(file_dir,dtype=str)
            #print(data)
            #cnt = cnt + 1
            tmp = []
            for m in range(0,32):
                #print(data[m])
                for n in data[m]:
                    num = int(n)
                    tmp.append(num)

            train_data.append(tmp)
    #print(cnt)
    return train_data,train_label


#读测试集
def load_Test_Data(dir):
    # cnt = 0
    test_data = []
    test_label = []
    for i in range(0, 10):  # 分数字读
        for j in range(0, test_num[i]):  # 分数字的样本读
            test_label.append(i)
            file_dir = dir + '\\' + str(i) + '_' + str(j) + '.txt'
            data = np.loadtxt(file_dir, dtype=str)
            # print(data)
            # cnt = cnt + 1
            tmp = []
            for m in range(0, 32):
                # print(data[m])
                for n in data[m]:
                    num = int(n)
                    tmp.append(num)

            test_data.append(tmp)
    # print(cnt)
    return test_data, test_label




def cal_Likelihood(train_data,train_label):
    train_num = train_data.shape[0]
    train_dim = train_data.shape[1]
    label_prob =  np.zeros((Label_num,))
    feature = np.zeros((Label_num,train_dim))
    #print(label_prob.shape)
    label_stat = Counter(train_label)
    #print(label_stat)

    for i in range(Label_num):
        label_prob[i] = label_stat[i] / train_num

    for i in range(Label_num):
        indexes = np.where(train_label == i)
        num_of_i = indexes[0].shape[0]
        for j in range(0,train_dim):
            times = 0
            #print(j)
            for index in indexes[0]:
                #print(train_data[index][j])
                if(train_data[index][j] == 0): #反正一个像素点的位置不过取值0或1，随便取一个进行计算即可
                    times += 1
                    #print(times)

            feature[i][j] = (times  + m*P ) / (num_of_i  + m)


    return label_prob, feature



def predict(test_data,label_prob,feature_prob):
    Possi = np.ones((Label_num,))
    #print(Possi)
    Possi = Possi * label_prob
    #print(Possi)
    for i in range(Label_num):
        feature_id = 0
        for data in test_data:
            '''
            被注释掉的是普通连乘
            没被注释的是对数似然
            二者结果一样，因为此处连乘没有出现下溢
            '''
            if(data == 0):
                #Possi[i] = Possi[i] * feature_prob[i][feature_id]
                Possi[i] = Possi[i] + np.log(feature_prob[i][feature_id])
            else:
                #Possi[i] = Possi[i] * (1 - feature_prob[i][feature_id])
                Possi[i] = Possi[i] + np.log((1 - feature_prob[i][feature_id]))
            feature_id += 1
    predict_label = np.where(Possi == max(Possi))
    #print(Possi)
    return predict_label[0][0]


Train_data,Train_label = load_Train_Data(train_dir)
#print(Train_data)

Train_data = np.asarray(Train_data)
Train_label = np.asarray(Train_label)


#print(Train_data.shape)
#print(Train_label.shape)


'''
label_prob: 每一个标签出现的概率，即先验概率
feature_prob：各个标签的每个像素的条件概率
             每一行的各列乘起来就是这一个标签的极大似然估计
'''


label_prob,feature_prob = cal_Likelihood(Train_data,Train_label)
print("各个标签出现的先验概率:{}".format(label_prob))
print("各个标签的每个像素的条件概率:{}".format(feature_prob))
Test_data,Test_label = load_Test_Data(test_dir)
Test_data = np.asarray(Test_data)
Test_label = np.asarray(Test_label)


#print(Test_label)
acc = 0

for i in range(Test_data.shape[0]):
    final_label = predict(Test_data[i], label_prob, feature_prob)
    if(final_label == Test_label[i]):
       acc = acc + 1


accuracy = (acc)/(Test_data.shape[0])
print("Accuracy rate: ",accuracy)



#print(type(train_num[0]))
# file_dir =  train_dir+ '\\'+ str(0) +'_' + str(0) +'.txt'
# print(file_dir)
# file = np.loadtxt(file_dir,dtype=str)
#print(len(file[0]))


