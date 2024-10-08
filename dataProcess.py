import numpy as np
from rich import print
from itertools import product

# data数据：每个元素为一个角点的原始标高数据
# 坐标系, 左上角为原点坐标
# ---->x 从左往右为 x
# |
# |
# |
# y
# 从上往下为 y
# 可以使用二维数组输入数据
# data = [
#     [28.6, 28.74, 29.43, 28.49, 28.87],
#     [27.2, 28.25, 28.34, 28.02, 27.71],
#     [27.37, 27.64, 28.5, 27.71, 27.47],
#     [26.91, 27.15, 27.69, 27.17, 27.19],
# ]

# 或者使用字符串输入数据
data = """
230.63	230.16	230.55	230.86	230.78	230.65	230.98	230.61	230.15	230.17	230.08	229.35	229.38	229.16	228.76	228.18	227.55	227.65
229.68	229.56	229.58	230.74	230.75	230.73	230.71	230.73	230.56	230.34	230.25	229.56	229.33	229.56	228.78	228.56	228.23	227.83
229.54	229.26	229.67	230.77	230.65	230.57	230.88	230.76	230.75	230.54	230.37	229.75	229.47	229.37	229.45	228.62	228.75	228.81
227.71	228.45	229.43	229.52	230.48	230.65	230.63	230.72	230.57	230.25	229.75	229.74	229.45	229.43	228.56	228.57	228.41	228.43
227.53	228.25	228.28	229.62	230.23	230.27	230.34	230.37	230.28	229.86	229.93	229.52	229.36	229.34	228.64	228.62	228.24	228.22
227.41	228.85	228.25	229.87	229.89	229.88	230.15	230.24	230.41	229.65	229.86	229.48	228.82	228.56	228.82	228.25	228.27	227.94
227.35	228.84	228.85	229.46	229.49	229.86	229.87	229.89	229.78	229.75	229.77	229.84	228.64	228.44	227.68	227.83	227.53	227.88
227.93	228.83	228.82	229.46	229.45	229.35	229.65	229.63	229.47	229.49	229.43	229.38	228.88	227.25	227.68	227.73	226.75	226.73
227.94	228.16	228.18	228.74	228.85	229.31	229.27	229.28	229.33	229.17	229.16	229.18	228.35	227.35	226.15	225.73	225.45	225.47
227.92	227.83	228.34	228.32	228.31	228.64	228.62	229.32	229.17	229.16	228.92	228.91	228.35	227.95	226.84	225.74	225.45	224.45
227.63	227.58	227.56	227.64	228.54	228.28	228.64	228.66	228.83	228.54	228.66	228.24	227.46	227.44	227.42	226.54	225.44	224.56
226.94	227.14	227.16	227.18	227.66	228.26	228.24	228.22	228.22	228.28	227.82	227.84	227.88	227.64	227.15	226.84	225.62	225.34
226.54	226.94	226.53	226.92	226.91	227.84	227.86	227.84	227.62	227.58	227.84	227.82	227.56	227.55	227.14	227.16	226.36	226.38
226.4	226.16	226.17	226.36	226.38	227.42	227.44	227.62	227.34	227.58	227.46	226.74	226.75	227.42	227.41	227.45	226.52	226.54
"""

# 纵坡 从上往下下降坡度
slope = 0.03  # m/m ly 0.3%
# 横坡 从左往右下降坡度
cross_slope = 0.0025  # m/m lx 0.25%
# 网格边长
lenth_net = 20  # m
# 拟平整场地后的最高点坐标 ， 根据坡度下降方向和坡面的形状确定
center_point = (0, 0)  # y x
# 字符串数据分割符号
data_str_split_symbol = None  # 默认"\t" or " "

linea = (
    [
        [
            float(ii.strip())
            for ii in (
                (i.split("\t" if "\t" in data else i.split(" ")))
                if not data_str_split_symbol
                else i.split(data_str_split_symbol)
            )
            if ii.strip()
        ]
        for i in data.strip().split("\n")
    ]
    if isinstance(data, str)
    else data
)

# 网格数 （列角点数目 -1 ，行角点数目 -1 ） 个
number_net = (np.array(linea).shape[0] - 1) * (np.array(linea).shape[1] - 1)

# 网格形状 # y x （列角点数目，行角点数目）
net_shape = (np.array(linea).shape[0], np.array(linea).shape[1])


assert (
    np.array(linea).shape == net_shape
), "原场地标高数据格式错误，应该是一个网格数据，二维数组"

assert number_net == (net_shape[0] - 1) * (net_shape[1] - 1), "原场地标高数据格式错误"


# 计算平整标高
def c_h0(lists: list[list]):

    h1, h2, h3, h4 = 0, 0, 0, 0
    maxindexy = net_shape[0] - 1
    maxindexx = net_shape[1] - 1
    h1_point = set(list(product([0, maxindexy], [0, maxindexx])))  # 计算一次的点，角点
    h4_point = set(
        list(
            product([i for i in range(1, maxindexy)], [i for i in range(1, maxindexx)])
        )
    )  # 计算四次的点，非边界边上的点
    h2_point = (
        set(
            list(
                product(
                    [i for i in range(maxindexy + 1)], [i for i in range(maxindexx + 1)]
                )
            )
        )  # all
        - h1_point
        - h4_point
    )  # 计算两次的点，边界边上的点
    # h3_point = set(list(product([1, 2], [0, 4])))  # 计算三次的点，凹角点， 在四方的方格网上不会出现，具体情况具体分析
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
原场地平整标高 = c_h0(linea)
print(原场地平整标高)
#


# 计算设计标高
def c_she_ji_biao_gao(lists: list[list], h0: float):
    results = []
    global center_point
    center_point = center_point
    for y, x_line in enumerate(lists):
        line = []
        for x, __ in enumerate(x_line):

            data = round(
                h0
                - abs(x - center_point[-1]) * cross_slope * lenth_net  # 横坡下降的标高
                - abs(y - center_point[0]) * slope * lenth_net,  # 纵坡下降的标高
                3,
            )
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


print("设计标高平整标高")
设计标高平整标高 = c_h0(c_she_ji_biao_gao(linea, c_x(linea)))

print(设计标高平整标高)

# 原场地平整标高应该等于设计标高平整标高
assert 设计标高平整标高 == 原场地平整标高, "设计标高平整标高应该等于原场地平整标高"

设计标高 = np.array(c_she_ji_biao_gao(linea, c_x(linea)))
原场地标高 = np.array(linea)
print("设计标高")
print(设计标高)
print("原场地标高")
print(原场地标高)
print("施工标高 = 设计标高减去原场地标高")
施工标高 = np.array(设计标高) - np.array(原场地标高)
print(施工标高)

# print(施工标高.shape)

assert 施工标高.shape == net_shape, "施工标高.shape != net_shape"

# 最终得到施工标高 = 设计标高减去原场地标高
