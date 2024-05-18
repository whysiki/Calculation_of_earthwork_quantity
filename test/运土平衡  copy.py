from rich import print
from math import *

# 挖方区域土方量
(
    ss1,
    ss2,
    ss3,
    ss4,
    ss5,
) = [
    round(float(i), 2)
    for i in "115.27700919999998 40.252598 0.009445333333333333 52.455234000000004 24.461145833333333".strip().split(
        " "
    )
]
print(ss1, ss2, ss3, ss4, ss5)
all_ss = ss1 + ss2 + ss3 + ss4 + ss5

print("挖方区块总和", all_ss)

zbs = [(9.26, 24.1), (4.41, 75.1), (40.09, 1.2), (41.20, 42.6), (49.20, 74.6)]
# 计算土方平衡
# 各个挖土区域挖土量以及区域中心点坐标
s1, s2, s3, s4, s5 = [[v, p] for v, p in zip([ss1, ss2, ss3, ss4, ss5], zbs)]

print(s1, s2, s3, s4, s5)

# 各个需要填土的区域的中心点坐标
t1, t2, t3, t4, t5, t6 = [
    (x, y) for x, y in zip([30, 30, 10, 30, 50, 50], [10, 30, 60, 60, 60, 20])
]

print(t1, t2, t3, t4, t5, t6)

# 挖土区域需要向填土距离运土, 当所有挖土区域土方量运完时, 计算各个挖土区域向各个填土区域运土的最佳方案:
from scipy.optimize import linear_sum_assignment
import numpy as np

# 计算所有挖土区域到所有填土区域的距离\
ts = [t1, t2, t3, t4, t5, t6]
distances = np.zeros((len(zbs), len(ts)))
for i, p in enumerate(zbs):
    x1, y1 = p
    for j, p2 in enumerate(ts):
        x2, y2 = p2
        distances[i, j] = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

print(np.round(distances, 2))
# 使用匈牙利算法找到最优的运输方案
row_ind, col_ind = linear_sum_assignment(distances)

# 打印结果
for i, j in zip(row_ind, col_ind):
    print(f"从挖土区域{i+1}到填土区域{j+1}运输")
