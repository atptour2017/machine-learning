# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
ws_data=pd.read_csv("ori_data.csv") 

ws_data.describe ## 大致看看数据集的情况
ws_data.shape ## 看看有多少行和列
ws_data.dtypes ## 查看每一例的数据类型
ws_data.columns ## 查看每列的变量名
