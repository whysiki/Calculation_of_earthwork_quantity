import matplotlib.pyplot as plt
import numpy as np

# 创建一些示例数据
data = np.random.randn(50)  # 生成50个随机数

fig, ax = plt.subplots()

# 根据值的正负和零来分配颜色
for value in data:
    if value < 0:
        color = "red"  # 负值为红点
    elif value > 0:
        color = "blue"  # 正值为蓝点
    else:
        color = "green"  # 零值为绿点

    ax.plot(value, 0, "o", color=color)  # 0是y坐标，所有点都在x轴上

# 添加自定义图例
from matplotlib.lines import Line2D

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="负值 (Negative)",
        markersize=10,
        markerfacecolor="red",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="正值 (Positive)",
        markersize=10,
        markerfacecolor="blue",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="零 (Zero)",
        markersize=10,
        markerfacecolor="green",
    ),
]

ax.legend(handles=legend_elements, loc="best")

plt.show()
