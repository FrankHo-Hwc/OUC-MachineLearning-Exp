import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from math import log

def load_data(file_name):
    # data = []
    # file = open(file_name,'r')  #打开文件
    # file_data = file.readlines() #读取所有行
    # file_data = np.array(file_data)
    # print(file_data.shape)
    data = pd.read_csv(file_name,sep='\s+',index_col=0,names=["Age","spectacle","astigmatic","tear","label"])
    data = np.asarray(data)
    data = data.tolist()
    return data

class DesicionTree:
    #计算信息熵
    def calEntropy(self, data):
        numofData = len(data)
        labelCounts = {}
        for i in data:
            label =  i[-1]
            if label not in labelCounts.keys():
                labelCounts[label] = 0
            labelCounts[label] += 1
        self.Ent = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numofData
            self.Ent -= prob * log(prob, 2)
        return self.Ent




    def spiltData(self,data,loc,label):
        #抽出数据子集，截取的原因是按照分支建子树时，已经用过的属性那一列已经没用了，可以删除那一列
        dataset =  []
        for  vec in data:
            if( vec[loc] == label):
                newvec = vec[:loc]
                newvec.extend(vec[loc + 1:])
                dataset.append(newvec)
        return dataset



    def spiltTree(self,data):
        #如何划分树，即找到一个最佳的属性去划分
        numofFeatures = len(data[0]) -1
        #print(numofFeatures)
        InitEnt  =  self.calEntropy(data)
        #print(InitEnt)
        self.bestFeature = -1
        self.bestGain_ratio = 0.0

        for i in  range(numofFeatures):
            newEnt = 0.0
            attribute = [vec[i] for vec in data]
            _class = set(attribute)
            IV = 0.0
            for label in _class:
                sonData = self.spiltData(data,i,label)
                #print(sonData)
                prob = len(sonData) / float(len(data))
                newEnt += prob * self.calEntropy(sonData)
                IV -= prob *log(2,prob)
            Gain = InitEnt - newEnt
            Gain_ratio = Gain / IV
            if(Gain_ratio > self.bestGain_ratio):
                self.bestGain_ratio = Gain_ratio
                self.bestFeature = i
        return self.bestFeature



    def createTree(self,data,feature):

        classList = [example[-1] for example in data]
        uniqueVals = set(classList)
        uniqueVals  = list(uniqueVals)
        #子结点所有样本类别相同时就停止划分
        if len(uniqueVals) == 1:
            return uniqueVals[0]
        bestFeat = self.spiltTree(data)
        bestFeatName = feature[bestFeat]
        Tree = {bestFeatName: {}}
        del (feature[bestFeat])
        featValues = [vec[bestFeat] for vec in data]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = feature[:]
            Tree[bestFeatName][value] = self.fit(self.spiltData(data, bestFeat, value), subLabels)
        return Tree

    def fit(self, data, feature):
        self.Tree = self.createTree(data,feature)
        self.feature_size = len(feature)
        return self.Tree

    def predict(self,data,features):
        feature = list(self.Tree.keys())[0]
        value = self.Tree[feature]
        featIndex = features.index(feature)
        for key in value.keys():
            if data[featIndex] == key:
                if type(value[key]).__name__ == 'dict':
                    classLabel = self.predict(value[key], data, features)
                else:
                    classLabel = value[key]
        return classLabel


dir = 'G:\machine learning\exp3\lenses_data.txt'
train_data = load_data(dir)
#print(train_data)

lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']

model = DesicionTree()
d_tree = model.fit(train_data,lensesLabels)
print(d_tree)


