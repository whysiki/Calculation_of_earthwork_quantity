import numpy as np
import math
from rich import print
from itertools import product

# 等高距
equidistant = 0.5  # m
# 纵坡
slope = 0.02  # m/m
# 横坡
cross_slope = 0.02  # m/m
# 网格数
number_net = 12
# 网格边长
lenth_net = 20  # m
# 网格形状
net_shape = (4, 5)


# 计算原场地标高
def c_biao_gao(lists: list[list]):
    results = []
    for ha, x, l in lists:
        results.append([ha + (x * equidistant) / l])

    return results


data = """
28.5 2.19 11.14, 28.5 3.9 7.96, 28.5 17.56 9.47, 28 21.61 21.48, 28.5 12.06 16.31,
27.5 -5.27 8.64, 28 5.01 10.18, 28 25.04 36.79, 28 0.96 23.97, 27.5 5.16 12.14,
27 9.13 12.48, 27.5 1.24 4.5, 28.5 0 1, 28 -5.66 9.7, 27 9.35 9.96,
27 -4.25 22.79, 27 2.82 9.36, 27.5 9.96 26, 27 8.7 25.33, 27 3.08 8.15
""".strip().replace(
    "\n", ""
)

datas = [[float(ii) for ii in i.split(" ") if ii] for i in data.split(",")]

# print(datas)

# for i in datas:
#     l = len(i)
#     if l != 3:
#         raise ValueError()
#     else:
#         print(True)

line = []
linea = []
# print(len(c_biao_gao(datas)), c_biao_gao(datas))
for index, i in enumerate(c_biao_gao(datas)):
    if (index + 1) % 5 == 0:
        line.append(round(i[0], 2))
        # print(line)
        linea.append(line)
        line = []
    else:
        line.append(round(i[0], 2))

# 原场地标高
print("原场地标高")
linea[0][3] = 28.49
print(linea)


# [[28.6, 28.74, 29.43, 28.5, 28.87], [27.2, 28.25, 28.34, 28.02, 27.71], [27.37, 27.64, 28.5, 27.71, 27.47], [26.91, 27.15, 27.69, 27.17, 27.19]]
# 计算平整标高
def c_h0(lists: list[list]):
    h1, h2, h3, h4 = 0, 0, 0, 0
    # 0 1 2 3 4   - > x
    # 0 1 2 3 4   |
    # 0 1 2 3 4   |
    # 0 1 2 3 4   y
    h1_point = set(list(product([0, 3], [0, 4])))
    h4_point = set(list(product([1, 2], [1, 2, 3])))
    h2_point = (
        set(list(product([i for i in range(4)], [i for i in range(5)])))  # all
        - h1_point
        - h4_point
    )
    # print(h1_point, h2_point, h4_point)
    for y, x_line in enumerate(lists):
        for x, data in enumerate(x_line):
            if (y, x) in h1_point:
                h1 += data
            elif (y, x) in h2_point:
                h2 += 2 * data
            elif (y, x) in h4_point:
                h4 += 4 * data
            else:
                raise ValueError("error point")
    # print(h1, h2, h4)
    result = (h1 + h2 + h4) / (4 * number_net)
    return round(result, 2)


# 平整标高
print("原场地平整标高")
print(c_h0(linea))
#


# 计算设计标高
def c_she_ji_biao_gao(lists: list[list], h0: float):
    results = []
    center_point = (0, 2)  # y x
    # 0 1 2 3 4   - > x
    # 0 1 2 3 4   |
    # 0 1 2 3 4   |
    # 0 1 2 3 4   y
    for y, x_line in enumerate(lists):
        line = []
        for x, __ in enumerate(x_line):

            data = round(
                h0
                - abs(x - center_point[-1]) * cross_slope * lenth_net
                - abs(y - center_point[0]) * slope * lenth_net,
                3,
            )
            # print(y, x, data)
            line.append(data)
        results.append(line)
    return results


# 计算设计标高的最高点
def c_x(lists: list[list]):
    # 假定最高点的x坐标
    init_x = min(lists[0])
    # 原场地平整标高
    origin_x = c_h0(linea)
    # 当设计标高的最高点的求得的设计标高的平整标高与原场地平整标高相等时
    # 返回设计标高的最高点的x坐标
    while abs(origin_x - c_h0(c_she_ji_biao_gao(lists, init_x))) != 0:
        init_x += 0.01
    return round(init_x, 2)


# 设计标高的最高点
print("设计标高的最高点")


print(c_x(linea))

# print("设计标高")
# print(np.array(c_she_ji_biao_gao(linea, c_x(linea))))
print("设计标高平整标高")
print(c_h0(c_she_ji_biao_gao(linea, c_x(linea))))


设计标高 = np.array(c_she_ji_biao_gao(linea, c_x(linea)))
原场地标高 = np.array(linea)
print("设计标高")
print(设计标高)
print("原场地标高")
print(原场地标高)
print("设计标高减去平整标高")
施工标高 = np.array(设计标高) - np.array(原场地标高)
print(施工标高)


# def find_zero_points(heights, grid_size=20):
#     zero_points = []
#     rows, cols = heights.shape

#     for i in range(rows - 1):
#         for j in range(cols - 1):
#             # Check if adjacent points have different signs
#             if np.sign(heights[i, j]) != np.sign(heights[i, j + 1]):
#                 # Calculate the zero point coordinate
#                 x_zero = j * grid_size + (grid_size / 2)
#                 y_zero = i * grid_size + (grid_size / 2)
#                 zero_points.append((x_zero, y_zero))

#             if np.sign(heights[i, j]) != np.sign(heights[i + 1, j]):
#                 x_zero = j * grid_size + (grid_size / 2)
#                 y_zero = (i + 1) * grid_size + (grid_size / 2)
#                 zero_points.append((x_zero, y_zero))

#     return zero_points


# # 计算零点坐标
# zero_points = find_zero_points(施工标高, grid_size=lenth_net)


# print("零点坐标")
# print(zero_points)
