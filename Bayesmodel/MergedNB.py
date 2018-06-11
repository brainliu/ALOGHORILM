# -*- coding: UTF-8 -*-
# Author: brain 
# Created_time:2018/6/11 15:10
# file:MergedNB.py
# location :china chengdu 61000
from Bayesmodel.Bacic import *
from Bayesmodel.MultionormialNB import MultinormialNB
from Bayesmodel.GaussianNB import GaussianNB
from  Util.util import DataUtil
class MergedNB(NativaBsyes):

    """
    self._whether_discrete:记录是否是离散型变量
    self._whether_continus:记录是否是连续性变量
    self._mutiomial,self_gaussian:离散型、联系性朴素贝叶斯模型
    """
    def __init__(self,whether_continus):
        super(MergedNB, self)
        self._mutionmial,self._gaussion=MultinormialNB(),GaussianNB()
        if whether_continus is None:
            self._whether_discrete=self._whether_continus=None
        else:
            self._whether_continus=np.array(whether_continus)
            self._whether_discrete=~self._whether_continus
    def feed_data(self,x,y,sample_weight=None):
        if sample_weight is not  None:
            sample_weight=np.array(sample_weight)
        x,y,wc,features,feat_discs,label_dics=DataUtil.quantize_data(x,y,wc=self._whether_continus,separate=True)
        if self._whether_continus is None:
            self._whether_continus=wc
            self._whether_discrete=~self._whether_continus
        self.label_dict=label_dics
        discrete_x,continus_x=x
        cat_conter=np.bincount(y)
        self._cat_counter=cat_conter
        labels=[y==value for value in range(len(cat_conter))]

        #训练离散型朴素贝叶斯

        labell_x=[discrete_x[ci].T for ci in labels]
        self._mutionmial._x,self._mutionmial._y=x,y
        self._mutionmial._labelled_x,self._mutionmial._labeled_zip=(labell_x,list(zip(labels,labell_x)))
        self._mutionmial._cat_counter=cat_conter
        self._mutionmial._feat_dics=[_dics for i ,_dics in enumerate(features) if self._whether_discrete[i]]
        self._mutionmial._n_possibilities=[len(feats) for i,feats in enumerate(features) if self._whether_discrete[i]]

        self._mutionmial.label_dict=label_dics

        labelled_x = [continus_x[label].T for label in labels]
        self._gaussion._x, self._gaussion._y = continus_x.T, y
        self._gaussion._labelled_x, self._gaussion._label_zip = labelled_x, labels
        self._gaussion._cat_counter, self._gaussion.label_dict = cat_conter, label_dics

        self.feed_sample_weight(sample_weight)
    def feed_sample_weight(self, sample_weight=None):
        self._mutionmial.feed_sample_weight(sample_weight)
        self._gaussion.feed_sample_weight(sample_weight)
    def _fit(self, lb):
        self._mutionmial.fit()
        self._gaussion.fit()
        p_category = self._mutionmial.get_prior_probablity(lb)
        discrete_func, continuous_func = self._mutionmial["func"], self._gaussion["func"]
        def func(input_x, tar_category):
            input_x = np.asarray(input_x)
            return discrete_func(
                input_x[self._whether_discrete].astype(np.int), tar_category) * continuous_func(
                input_x[self._whether_continus], tar_category) / p_category[tar_category]
        return func
    def _transfer_x(self, x):
        feat_dicts = self._mutionmial["_feat_dics"]
        idx = 0
        for d, discrete in enumerate(self._whether_discrete):
            if not discrete:
                x[d] = float(x[d])
            else:
                x[d] = feat_dicts[idx][x[d]]
            if discrete:
                idx += 1
        return x
if __name__ == '__main__':
    import time

    # whether_discrete = [True, False, True, True]
    # x = DataUtil.get_dataset("balloon2.0", "../../_Data/{}.txt".format("balloon2.0"))
    # y = [xx.pop() for xx in x]
    # learning_time = time.time()
    # nb = MergedNB(whether_discrete)
    # nb.fit(x, y)
    # learning_time = time.time() - learning_time
    # estimation_time = time.time()
    # nb.evaluate(x, y)
    # estimation_time = time.time() - estimation_time
    # print(
    #     "Model building  : {:12.6} s\n"
    #     "Estimation      : {:12.6} s\n"
    #     "Total           : {:12.6} s".format(
    #         learning_time, estimation_time,
    #         learning_time + estimation_time
    #     )
    # )

    whether_continuous = [False] * 16
    continuous_lst = [0, 5, 9, 11, 12, 13, 14]
    for cl in continuous_lst:
        whether_continuous[cl] = True

    train_num = 40000
    data_time = time.time()
    (x_train, y_train), (x_test, y_test) = DataUtil.get_dataset(
        "bank1.0", "../data/bank1.0.txt", n_train=train_num)
    data_time = time.time() - data_time
    learning_time = time.time()
    nb1 = MergedNB(whether_continus=whether_continuous)
    nb1.fit(x_train, y_train)
    learning_time = time.time() - learning_time
    estimation_time = time.time()
    nb1.evaluate(x_train, y_train)
    nb1.evaluate(x_test, y_test)
    estimation_time = time.time() - estimation_time
    print(
        "Data cleaning   : {:12.6} s\n"
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            data_time, learning_time, estimation_time,
            data_time + learning_time + estimation_time
        )
    )