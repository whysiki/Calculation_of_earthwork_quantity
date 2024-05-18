# import matplotlib.pyplot as plt

# # 输入路径点
# paths = [
#     [(0, 0), (8.8, 0), (20, 17.23), (20, 0), (20, 20), (0, 20), (0, 0)],
#     [(20, 0), (20, 17.23), (40, 0), (28.57, 20), (40, 20), (20, 20), (20, 0)],
# ]


# # 定义重排序函数
# def reorder_path(points):
#     # 去掉最后的闭合点
#     points = points[:-1]

#     # 找到左上角、右上角、右下角和左下角的点
#     left_upper = min(points, key=lambda p: (p[1], p[0]))
#     right_upper = min(points, key=lambda p: (p[1], -p[0]))
#     right_lower = max(points, key=lambda p: (p[1], p[0]))
#     left_lower = max(points, key=lambda p: (p[1], -p[0]))

#     # 将点分配到对应的边上
#     top_edge = sorted([p for p in points if p[1] == left_upper[1]], key=lambda p: p[0])
#     right_edge = sorted(
#         [p for p in points if p[0] == right_upper[0] and p[1] > right_upper[1]],
#         key=lambda p: p[1],
#     )
#     bottom_edge = sorted(
#         [p for p in points if p[1] == left_lower[1]], key=lambda p: p[0], reverse=True
#     )
#     left_edge = sorted(
#         [p for p in points if p[0] == left_upper[0] and p[1] > left_upper[1]],
#         key=lambda p: p[1],
#         reverse=True,
#     )

#     # 重排序路径
#     ordered_points = top_edge + right_edge + bottom_edge + left_edge + [left_upper]
#     return ordered_points


# # 重新排序路径
# reordered_paths = [reorder_path(path) for path in paths]

# # 打印重新排序后的路径
# for path in reordered_paths:
#     print(path)

# # 可视化路径
# for path in reordered_paths:
#     path_x, path_y = zip(*path)
#     plt.plot(path_x, path_y, marker="o")

# plt.gca().invert_yaxis()
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Reordered Paths")
# plt.grid(True)
# plt.show()



def sort_closepath(points):
    # 去掉重复的起点和终点
    points = list(dict.fromkeys(points))

    # 计算最小和最大 x 和 y
    min_x = min(points, key=lambda p: p[0])[0]
    max_x = max(points, key=lambda p: p[0])[0]
    min_y = min(points, key=lambda p: p[1])[1]
    max_y = max(points, key=lambda p: p[1])[1]

    # 分类点
    left_upper = sorted([p for p in points if p[0] == min_x and p[1] == min_y], key=lambda p: p[0])
    right_upper = sorted([p for p in points if p[1] == min_y and p[0] > min_x], key=lambda p: p[0])
    right_lower = sorted([p for p in points if p[0] == max_x and p[1] > min_y], key=lambda p: p[1])
    left_lower = sorted([p for p in points if p[1] == max_y and p[0] < max_x], key=lambda p: p[0], reverse=True)
    left_upper_from_bottom = sorted([p for p in points if p[0] == min_x and p[1] > min_y], key=lambda p: p[1], reverse=True)

    # 构建排序后的路径
    sorted_points = left_upper + right_upper + right_lower + left_lower + left_upper_from_bottom

    # 确保路径是闭合的
    if sorted_points[0] != sorted_points[-1]:
        sorted_points.append(sorted_points[0])

    return sorted_points

# 测试 closepath1
closepath1 = [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0), (8.8, 0), (20, 17.23)]
sorted_closepath1 = sort_closepath(closepath1)
print("Sorted closepath1:", sorted_closepath1)

# 测试 closepath2
closepath2 = [
    (20, 0),
    (40, 0),
    (40, 20),
    (20, 20),
    (20, 0),
    (20, 17.23),
    (28.57, 20),
    (39.65, 0),
    (40, 1.54),
]
sorted_closepath2 = sort_closepath(closepath2)
print("Sorted closepath2:", sorted_closepath2)
