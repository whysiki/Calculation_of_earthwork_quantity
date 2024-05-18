import matplotlib.pyplot as plt
import numpy as np

# 创建数据点
x = np.linspace(0, 10, 10)
y = np.zeros(10)
u = np.ones(10)
v = np.zeros(10)

# 生成随机颜色
colors = np.random.rand(10, 3)  # 生成10个RGB颜色

# 绘制箭头线
plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color=colors)
plt.xlim(0, 11)
plt.ylim(-1, 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
