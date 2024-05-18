from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from rich import print

# 定义多个闭合路径（多边形）
polygons = [
    Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]),
    Polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]),
    Polygon([(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]),
]

print(polygons)

# 使用unary_union合并多边形
merged_polygon = unary_union(polygons)

# 获取新的多边形顶点
if isinstance(merged_polygon, Polygon):
    merged_vertices = list(merged_polygon.exterior.coords)
elif isinstance(merged_polygon, MultiPolygon):
    merged_vertices = []
    for poly in merged_polygon:
        merged_vertices.extend(list(poly.exterior.coords))

# 打印新的多边形顶点
print("新的多边形顶点:")
for vertex in merged_vertices:
    print(vertex)
