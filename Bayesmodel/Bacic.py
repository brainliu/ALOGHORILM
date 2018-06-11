#-*-coding:utf8-*-
#user:brian
#created_at:2018/6/8 11:18
# file: Bacic.py
#location: china chengdu 610000
import numpy as np
import  time
#定义朴素贝叶模型的基类
class NativaBsyes:
    """
    _x，_y:存储的训练集变量
    _data:核心数组，存储实际使用的条件概率的相关信息
    _func：模型核心，决策函数，能够根据输入的x，y输出对应的后验概率
    _n_possibilities:记录各个维度特征取值个数的数组
    _lanelled_x:记录按照类别分开后的输入数据的数组
    lanelel_zip:记录按照类别相关信息的数组，视具体算法，定义会有所不同
    _cat_counter:核心数组，记录第i类数据的个数，也就是category
    _con_counter:核心数组，记录数据条件概率的原始极大使然估计
    _label_dict:核心字典，用于记录数值变化类别时的转换关系
    _feat_dict:核心字典，用于记录数值变化各维度特征转换的关系
    """
    def __init__(self):
        self._x=self._y=None
        self._data=self._func=None
        self._n_possibilities=None
        self._labeled_x=self._labeled_zip=None
        self._cat_counter=self._con_counter=None
        self.label_dict=self._feat_dics=None
    #重载_getitem_避免大量定义
    def __getitem__(self, item):
        if isinstance(item,str):
            return getattr(self,"_"+item)
    def feed_data(self,x,y,sample_weight=None):
        pass
    def feed_sample_weight(self,sample_weight=None):
        pass
    def get_prior_probablity(self,lb=1):
        return[(_c_num +lb)/(len(self._y)+lb*len(self._cat_counter))  for _c_num in self._cat_counter]

    def fit(self,x=None,y=None,sample_weigth=None,lb=1):
        if x is not  None and y is not  None:
            self.feed_data(x,y,sample_weigth)
        self._func=self._fit(lb)
    def _fit(self,lb):
        pass
    def predict_one(self,x,get_raw_result=False):
        if isinstance(x,np.ndarray):
            x=x.tolist()
        else:
            x=x[:]
        x=self._transfer(x)
        m_arg, m_probability = 0, 0
        for i in range(len(self._cat_counter)):
            p = self._func(x, i)
            if p > m_probability:
                m_arg, m_probability = i, p
        if not get_raw_result:
            return self.label_dict[m_arg]
        return m_probability

    def _transfer(self,X):
        pass

    def predict(self,x,get_raw_result=False):
        return np.array([self.predict_one(xx,get_raw_result) for xx in x])

    def evaluate(self,x,y):
        y_pred=self.predict(x)
        print("ACC: {:12.6}%".format(100*np.sum(y_pred==y)/len(y)))
import numpy as np
from math import pi,exp
sqrt_pi=(2*pi)**0.5
class NBFunctions:
    #定义正太分布的密度函数import time
    @staticmethod
    def gaussian(x,mu,sigma):
        return exp(-(x-mu)**2/ (2*sigma**2))/(sqrt_pi*sigma)
    @staticmethod
    def gauss_maximum_likehood(labeled_x,n_category,dim):
        mu=[np.sum(
            labeled_x[c][dim])/len(labeled_x[c][dim]) for c in range(n_category)
        ]
        sigma=[np.sum((labeled_x[c][dim]-mu[c])**2)/len(labeled_x[c][dim]) for c in range(n_category)]

        def func(_c):
            def sub(xx):
                return NBFunctions.gaussian(xx,mu[_c],sigma[_c])
            return sub
        return [func(_c=c)for c in range(n_category)]