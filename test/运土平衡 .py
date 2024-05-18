from rich import print
from math import *
import numpy as np

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
s1, s2, s3, s4, s5 = [(v, p) for v, p in zip([ss1, ss2, ss3, ss4, ss5], zbs)]

print(s1, s2, s3, s4, s5)


# 各个需要填土的区域的中心点坐标
t1, t2, t3, t4, t5, t6 = [
    (x, y) for x, y in zip([30, 20, 10, 30, 50, 50], [10, 50, 60, 60, 60, 20])
]

print(t1, t2, t3, t4, t5, t6)

# 计算运土距离乘以运土量总和的最小值时, 各个挖土区域向各个填土区域运土的最佳方案

# 当所有挖土区域土方量运完时, 计算各个挖土区域向各个填土区域运土的最佳方案
from scipy.optimize import linprog

# Calculate the distance between each dig area and each fill area
distances = np.zeros((5, 6))
for i, (v1, p1) in enumerate([s1, s2, s3, s4, s5]):
    for j, p2 in enumerate([t1, t2, t3, t4, t5, t6]):
        distances[i, j] = sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Flatten the distances array to 1D
c = distances.flatten()

# Create the equality constraints matrix
A_eq = np.zeros((5, 30))
for i in range(5):
    A_eq[i, i * 6 : i * 6 + 6] = 1

# Create the equality constraints vector
b_eq = np.array([ss1, ss2, ss3, ss4, ss5])

# Create the inequality constraints matrix
A_ub = np.zeros((6, 30))
for i in range(6):
    A_ub[i, i::6] = 1

# Create the inequality constraints vector
b_ub = np.array([all_ss, all_ss, all_ss, all_ss, all_ss, all_ss])

# Create the bounds for each variable in the solution
x0_bounds = (0, None)

# Solve the linear programming problem
res = linprog(
    c,
    A_eq=A_eq,
    b_eq=b_eq,
    A_ub=A_ub,
    b_ub=b_ub,
    bounds=[x0_bounds] * 30,
    method="highs",
)

# Set numpy print options
np.set_printoptions(precision=2, suppress=True)
# Print the optimal solution
solution = res.x.reshape((5, 6))
# print(np.around(solution, 2))
print(solution)
