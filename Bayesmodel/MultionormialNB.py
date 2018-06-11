#-*-coding:utf8-*-
#user:brian
#created_at:2018/6/9 10:18
# file: MultionormialNB.py
#location: china chengdu 610000
from Bayesmodel.Bacic import *

class MultinormialNB(NativaBsyes):
    def feed_data(self,x,y,sample_weight=None):
        if isinstance(x,list):
            features=map(list,zip(*x))
        else:
            features=x.T
        #采用bincout来计算
        features=[set(feat) for feat in features] #一共有多少种组合
        feat_dics=[{_l:i for i,_l in enumerate(feats)}for feats in features]  #许多个字典
        label_dics={_l:i for i,_l in enumerate(set(y))} #一个字典
        x=np.array([[feat_dics[i][_l]for i ,_l in enumerate(sample) ]for sample in x])
        y= np.array([label_dics[yy] for yy in y])
        cat_counter = np.bincount(y)
        n_possibilities = [len(feats) for feats in features]
        labels = [y == value for value in range(len(cat_counter))]
        labelled_x = [x[ci].T for ci in labels]
        #更新数据
        self._x, self._y = x, y
        self._labelled_x, self._label_zip = labelled_x, list(zip(labels, labelled_x))
        self._cat_counter, self._feat_dics, self._n_possibilities = cat_counter, feat_dics, n_possibilities
        self.label_dict = {i:_l for _l,i in label_dics.items()}
        self.feed_sample_weight(sample_weight)

    def feed_sample_weight(self, sample_weight=None):
        self._con_counter = []
        for dim, p in enumerate(self._n_possibilities):
            if sample_weight is None:
                self._con_counter.append([
                    np.bincount(xx[dim], minlength=p) for xx in self._labelled_x])
            else:
                local_weights = sample_weight * len(sample_weight)
                self._con_counter.append([
                    np.bincount(xx[dim], weights=local_weights[label], minlength=p)
                    for label, xx in self._label_zip])

    def _fit(self, lb):
        n_dim = len(self._n_possibilities)
        n_category = len(self._cat_counter)
        p_category = self.get_prior_probablity(lb)

        data = [[] for _ in range(n_dim)]
        for dim, n_possibilities in enumerate(self._n_possibilities):
            data[dim] = [
                [(self._con_counter[dim][c][p] + lb) / (self._cat_counter[c] + lb * n_possibilities)
                 for p in range(n_possibilities)] for c in range(n_category)]
        self._data = [np.asarray(dim_info) for dim_info in data]

        def func(input_x, tar_category):
            rs = 1
            for d, xx in enumerate(input_x):
                rs *= data[d][tar_category][xx]
            return rs * p_category[tar_category]

        return func

    def _transfer(self, x):
        for j, char in enumerate(x):
            x[j] = self._feat_dics[j][char]
        return x
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams["font.sans-serif"]=['Fangsong']
mpl.rcParams["axes.unicode_minus"]=False

# def plot_all(nb,dataset):
#     data=nb._data
#     colors={"不爆炸":"blue","爆炸":"red"}
#     #反字典化
#     _rev_feat_dics=[{_val:key for key ,_val in item.items()} for item in  nb._feat_dics]
#     for _j in range(nb._x.shape[1]):
#         sj=nb._n_possibilities[_j]
#         temp_x=np.arange(1,sj+1)
#         tittle="$j= {};s_j={}$".format(_j+1,sj)
#         plt.figure()
#         plt.title(tittle)
#         for _c in range(len(nb.label_dict)):
#             plt.bar(temp_x-0.35*_c,data[_j][_c,:],width=0.35, facecolor=colors[nb.label_dict[_c]],edgecolor="white"
#             ,label="class :{}".format(nb.label_dict[_c]))
#             plt.xticks([i for i in range(sj)], [""] + [_rev_feat_dics[_j]] + [""])
#             plt.ylim(0, 1.0)
#             plt.legend()
#             # 保存画好的图像
#             plt.savefig("../result/d{0}，{1}.png".format(dataset,_j + 1))


if __name__ == '__main__':
    import time
    from  Util.util import DataUtil
    for dataset in ["balloon1.0","balloon1.5"]:
        print("="*20)
        print(dataset)
        print("-"*20)
        _X,_Y=DataUtil.get_dataset(dataset,"../data/{}.txt".format(dataset))
        learinning_time=time.time()
        nb=MultinormialNB()
        nb.fit(_X,_Y)
        learinning_time=time.time()-learinning_time
        estiamtime=time.time()
        nb.evaluate(_X,_Y)
        estiamtime=time.time()-estiamtime
        #print output
        print(
            "model bulding: {:12.6}s\n"
            "Estimation: {:12.6}s\n"
            "Toatl ：{:12.6}".format(
                learinning_time,estiamtime,learinning_time+estiamtime
            )
        )
        print(" "*20)
        # plot_all(nb,dataset)

