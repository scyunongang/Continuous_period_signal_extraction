import numpy as np
import pandas as pd
def get_data():
    data = pd.read_csv("./data_set/train_data3.csv", header=0,encoding='gbk')  # 要哪行就排除剩下的几行
    data = np.array(data)
    print(data.dtype)
    rc = data[:, 1:2]
    rt = data[:, 2:3]
    rf = data[:, 3:4]
    print(data.shape)
    sample = data[:53,4:5]
    print(sample.shape)
    return rf,rc,rt,sample
get_data()