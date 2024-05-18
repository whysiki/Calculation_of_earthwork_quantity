from shapely.geometry import Polygon
import matplotlib.pyplot as plt

# 定义两个多边形
polygon1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
polygon2 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

# 计算相交的多边形
intersection = polygon1.intersection(polygon2)

# 打印相交多边形的面积
print("相交面积:", intersection.area)

# 打印相交多边形的闭合路径
if intersection.is_empty:
    print("两个多边形没有相交。")
else:
    print("相交多边形的顶点:", list(intersection.exterior.coords))


plt.figure()

# 绘制多边形1
x1, y1 = polygon1.exterior.xy
plt.fill(x1, y1, alpha=0.5, fc="r", ec="none")

# 绘制多边形2

x2, y2 = polygon2.exterior.xy
plt.fill(x2, y2, alpha=0.5, fc="b", ec="none")

# 绘制相交多边形

if not intersection.is_empty:
    x, y = intersection.exterior.xy
    plt.fill(x, y, alpha=0.5, fc="black", ec="none")

plt.show()


print(list(intersection.exterior.coords))
