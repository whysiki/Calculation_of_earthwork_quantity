import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

def is_point_on_edge(point, polygon):
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        
        # 向量 (x2-x1, y2-y1)
        vec1 = (p2[0] - p1[0], p2[1] - p1[1])
        # 向量 (x-x1, y-y1)
        vec2 = (point[0] - p1[0], point[1] - p1[1])
        
        # 计算叉积
        cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        
        # 点是否在边上
        if cross_product == 0:
            min_x, max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
            min_y, max_y = min(p1[1], p2[1]), max(p1[1], p2[1])
            if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
                return True, (p1, p2)
    
    return False, None

def is_point_in_polygon_or_on_edge(x, y, polygon_coords):
    point = Point(x, y)
    polygon = Polygon(polygon_coords)
    
    if polygon.contains(point):
        return "Inside", None
    else:
        on_edge, edge = is_point_on_edge((x, y), polygon_coords)
        if on_edge:
            return "On edge", edge
        else:
            return "Outside", None

# 示例使用
polygon_coords = [(1, 1), (1, 4), (4, 4), (4, 1)]
point = (4, 2)

result, edge = is_point_in_polygon_or_on_edge(point[0], point[1], polygon_coords)
print(result)  # 输出："On edge"
if edge:
    print("Edge:", edge)  # 输出："Edge: ((4, 4), (4, 1))"
    print(type(edge))

# 绘图部分
polygon_coords.append(polygon_coords[0])  # 闭合多边形
x_coords, y_coords = zip(*polygon_coords)

plt.figure()
plt.plot(x_coords, y_coords, 'b-')  # 画出多边形
plt.plot(point[0], point[1], 'ro')  # 画出点

# 标记边
if edge:
    edge_coords = zip(*edge)
    plt.plot(*edge_coords, 'g-', linewidth=3)

plt.fill(x_coords, y_coords, alpha=0.2)
plt.text(point[0], point[1], f'({point[0]}, {point[1]})', fontsize=12, ha='right')
plt.title(f'Point is {result}')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
plt.show()
