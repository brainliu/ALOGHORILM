# -*- coding: UTF-8 -*-
# Author: brain 
# Created_time:2018/6/11 14:07
# file:GaussianNB.py
# location :china chengdu 61000
from Bayesmodel.Bacic import *
class GaussianNB(NativaBsyes):
    def feed_data(self,x,y,sample_weight=None):
        x=np.array([list(map(lambda c: float(c),sample)) for sample in x])
        #数值化类别向量
        labels=list(set(y))
        #将标签字符等转化为数字
        label_dic={label:i for i,label in enumerate(labels)}
        y=np.array([label_dic[yy] for yy in y]) #y是从0,1,2,3,4这种开始的
        cat_counter=np.bincount(y) #统计Y的出现次数和类别
        labels=[y==value for value in range(len(cat_counter))] #调用了y的特性
        labeled_x=[x[label].T for label in labels] #相当于y有几类，就分成了几类x的数据
        #更新属性
        self._x,self._y=x.T,y
        self._labelled_x,self._labeled_zip=labeled_x,labels
        self._cat_counter,self.label_dict=cat_counter,{i:_l for _l,i in label_dic.items()}
        self.feed_sample_weight(sample_weight)
    #定义处理样本的权重函数
    def feed_sample_weight(self,sample_weight=None):
        if sample_weight is not None:
            loca_weigths=sample_weight*len(sample_weight)
            for i,label in enumerate(self._labeled_zip):
                self._labelled_x[i]*=loca_weigths[label]

    def _fit(self,lb):
        n_category=len(self._cat_counter)
        p_catefory=self.get_prior_probablity(lb)
        #利用极大释然函数计算条件概率的函数，用数组变量进行保存
        data=[
            NBFunctions.gauss_maximum_likehood(
                self._labelled_x,n_category,dim
            ) for dim in range(len(self._x))
        ]
        self._data=data

        def func(input_x,tar_category):
            rs=1
            for d,xx in enumerate(input_x):
                rs*=data[d][tar_category](xx)
            return rs*p_catefory[tar_category]
        return func
    @staticmethod
    def _transfer(X):
        return X
if __name__ == '__main__':
    import time
    from Util.util import DataUtil

    for dataset in ["test"]:
        print("=" * 20)
        print(dataset)
        print("-" * 20)
        _X, _Y = DataUtil.get_dataset(dataset, "../data/{}.txt".format(dataset))
        learinning_time = time.time()
        nb = GaussianNB()
        nb.fit(_X, _Y)
        learinning_time = time.time() - learinning_time
        estiamtime = time.time()
        nb.evaluate(_X, _Y)
        estiamtime = time.time() - estiamtime
        # print output
        print(
            "model bulding: {:12.6}s\n"
            "Estimation: {:12.6}s\n"
            "Toatl ：{:12.6}".format(
                learinning_time, estiamtime, learinning_time + estiamtime
            )
        )
        print(" " * 20)

