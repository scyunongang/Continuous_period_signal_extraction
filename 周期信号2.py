# 增加了DTW
from scipy.stats import pearsonr
import numpy as np
from scipy.spatial.distance import euclidean  # 引入欧氏距离
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from data import get_data
import datetime
rf, rc, rt, sample= get_data()
# print('移动次数',len(x_sum2)/300)
corref_lis = []
p1 = []
loss2 = []

def generate_arr(x,y,path):  # point_num = len(path)-1为点数，此处可以设置要保留的点数
    point_num = len(path) - 1
    x_sample,y_sample = [],[]
    for i in range(point_num):
        ele  = path[i]
        x_sample.append(x[ele[0]])
        y_sample.append(y[ele[1]])
    return np.array(x_sample),np.array(y_sample)


def feature_extracing(x_sum=rf,pace = 3,T_start =52,T_stop = 58, p_1=0.02,pear=0.8,pic = True,dtw = False):  # p_1为相关系数 p_2为P值
    # x_sum 是输入的时序信号，pace是每次移动的步长，原则越短越好，但会变慢！，T_start ，T_stop 分别为周期包含点数的最小值，最大值
    # threhold 是累加距离的阈值，小于他说明经过调整后已经非常接近样本函数，可以认为是要提取的信号
    # Stability_factor 是稳定系数，可以理解为看小于threhold的要提出图的数量系数，越大要求就越苛刻，剔除出来的信就越少，但质量都很好
    open , line_n, n_point,distance,dot,symbol =  False,0,0,0,0,True
    save_window = []
    dict_set = dict()
    pi = pic
    save_sata=[]
    # np.where(pi==True,plt.ion(),plt.ioff())  # something about plotting动态图打开
    if pi==True:
        plt.ion()
    for t in range(int(len(x_sum)//pace)):  # 设置移动步伐的大小
        for j in range(T_start,T_stop):  # 周期信号长度
            y = x_sum[pace * t:(int(j) + pace * t)] # 待测样本
            if dtw==True:
                distance, path = fastdtw(y, sample, dist=euclidean)
                print('ditance',distance)
                s_sam1, s_am2 = generate_arr(y,sample,path)
            else:
                s_sam1 = y
                s_am2 = sample
            if len(s_sam1)==len(s_am2):
                p = (pearsonr(s_am2.ravel(), s_sam1.ravel()))
                p1.append(p)
                # print('相关系数：',p)
                x = np.linspace(pace * t, pace * t + j, j)  # 主要是计算x的坐标，无其他用途
                if p[1] < p_1 and p[0] > pear and distance<16: #P<0.05时表示相关显著，即在当前的样本下可以明显的观察到两变量的相关，两个变量的相关有统计学意义。
                    dot += 1
                    dict_set[p[0]] = y  # 对应待测信号中的点数  and
                    open = True
                else:
                    if dot>1 and open == True:
                        dot = 0
                        line_n += 1
                        n_point = j
                        open = False
                        lis = max(dict_set.keys()) # 求解distance最小值对应的图像
                        # global save_window
                        save_window = dict_set[lis]
                        save_sata.append(save_window.ravel())
                        dict_set.clear()
                    elif dot == 1 and open == True:
                        dot = 0
                        open = False
                        line_n += 1
                        n_point = j
                        lis = max(dict_set.keys())  # 求解distance最小值对应的图像
                        save_window = dict_set[lis]
                        save_sata.append(save_window.ravel())
                        dict_set.clear()

            if t < len(x_sum)/pace-j//pace and pi: # 结束点标志设置
                plt.subplot(1, 2, 1)
                plt.cla()
                plt.plot(x_sum,color = 'g',linestyle = '-') # 画总的曲线
                plt.plot(x,y,color = 'r',marker = 'x') # 画移动的曲线
                plt.text(0, 1.6, 'p=%.4f'% p[1], fontdict={'size': 10, 'color': 'red'})
                plt.text(10000, 1.6, 'n: %i' % line_n, fontdict={'size': 13, 'color': 'red'})
                plt.text(18000, 1.6, 'n_point: %i' % n_point, fontdict={'size': 13, 'color': 'red'})
                plt.text(30000, 1.6, 'pear: %.5f' % p[0], fontdict={'size': 13, 'color': 'red'})
                plt.subplot(1, 2, 2)
                # print('save_window////////////////////////////////////////////////////', (save_window))
                plt.plot(save_window)
                # plt.text(30, 1.25, 'Found a total of %i samples' % line_n, fontdict={'size': 10, 'color': 'b'})
                plt.show()
                plt.pause(0.00001)  # 延时0.01s

    plt.ioff()
    return save_sata

if __name__ == '__main__':
    starttime1 = datetime.datetime.now()
    save_sata1 = feature_extracing(x_sum=rf,pace = 3,T_start =51,T_stop = 55, p_1=0.01,pear=0.85,pic = False,dtw = True) # p_1为相关系数 p_2为P值,pic是True时会实时显示当前状态，但是会影响整个过程的提取速度
    plt.subplot(1, 2, 1)
    plt.plot(rf,linewidth=0.1)
    plt.subplot(1, 2, 2)
    for i in save_sata1:
        plt.plot(i)
    plt.text(0, 1.6, 'Extract the periodic signal number: %i' % len(save_sata1), fontdict={'size': 13, 'color': 'red'})
    endtime1 = datetime.datetime.now()
    time = (endtime1-starttime1)
    plt.text(0, 1.8, 'Total time:%s s' % time, fontdict={'size': 13, 'color': 'red'})
    plt.show()
    print('Total time：', (endtime1 - starttime1), '秒')