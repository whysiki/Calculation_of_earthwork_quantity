import numpy as np
import matplotlib.pyplot as plt
from rich import print
from typing import Iterable, List, Union
from itertools import combinations
from tqdm import tqdm
from collections import defaultdict, deque
from itertools import product
from shapely.geometry import Point, Polygon


def is_point_on_edge(point: tuple[float, float], polygon: Polygon):

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


def is_point_in_polygon_or_on_edge(
    x: float, y: float, polygon_coords: Iterable[tuple[float, float]]
) -> tuple[str, Union[None, tuple[tuple, tuple]]]:
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


class Grid:

    def __init__(
        self,
        left_upper: "GridPoint",
        right_upper: "GridPoint",
        right_lower: "GridPoint",
        left_lower: "GridPoint",
        index: None,
    ) -> None:

        assert all(
            isinstance(p, GridPoint)
            for p in (
                left_upper,
                right_upper,
                right_lower,
                left_lower,
            )
        )

        self.left_upper, self.right_upper, self.right_lower, self.left_lower = (
            left_upper,
            right_upper,
            right_lower,
            left_lower,
        )

        self.path = GridPath(
            [
                self.left_upper,
                self.right_upper,
                self.right_lower,
                self.left_lower,
                self.left_upper,
            ]
        )

        self.index = index

        self.center_coordinate = self.__get_center_coordinate()

    def __get_center_coordinate(self):

        points = (
            self.left_upper,
            self.right_upper,
            self.right_lower,
            self.left_lower,
        )

        x_coords = [point.x for point in points]
        y_coords = [point.y for point in points]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        return (center_x, center_y)

    def __repr__(self) -> str:
        return f"""
                {self.left_upper.coordinate}     {self.right_upper.coordinate}
                {self.left_lower.coordinate}     {self.right_lower.coordinate}
                """

    def __str__(self) -> str:
        return self.__repr__()

    def insertPointInEdgePath(self, point: "GridPoint", index: int):

        edges: list[GridPoint] = self.path.edges

        edges.insert(index, point)

        self.path = GridPath(edges)


class GridNet:

    def __init__(self, verticalPoints=None, horizontalPoints=None) -> None:

        self.edge_length = 20

        self.verticalPoints: list[list[GridPoint]] = (
            verticalPoints if verticalPoints else []
        )  # 表示列视角下每列的的所有顶点

        self.horizontalPoints: list[list[GridPoint]] = (
            horizontalPoints if horizontalPoints else []  # 行视角下每行的顶点
        )

        self.grids: list[list[Grid]] = (
            self.constructGridsNet() if verticalPoints and horizontalPoints else []
        )

    def constructGridsNet(self) -> list[list[Grid]]:

        grids: list[list[Grid]] = []

        grid_index = 0

        # 遍历每一行
        for i in range(len(self.horizontalPoints) - 1):
            row_grids = []

            # 遍历每一列
            for j in range(len(self.verticalPoints) - 1):
                # 获取网格的四个顶点
                left_upper: GridPoint = self.horizontalPoints[i][j]
                right_upper: GridPoint = self.horizontalPoints[i][j + 1]
                right_lower: GridPoint = self.horizontalPoints[i + 1][j + 1]
                left_lower: GridPoint = self.horizontalPoints[i + 1][j]

                # 构建网格
                grid: Grid = Grid(
                    left_upper=left_upper,
                    right_upper=right_upper,
                    right_lower=right_lower,
                    left_lower=left_lower,
                    index=grid_index,
                )
                row_grids.append(grid)
                grid_index += 1

            grids.append(row_grids)

        return grids

    @staticmethod
    def getverticalPoints_and_horizontalPoints(
        vertexs: List["GridPoint"],
    ) -> tuple[List[List["GridPoint"]], List[List["GridPoint"]]]:

        verticalPoints: List[List[GridPoint]] = []

        horizontalPoints: List[List[GridPoint]] = []

        for pv, ph in zip(
            sorted(vertexs, key=lambda p: (p.x, p.y)),
            sorted(vertexs, key=lambda p: (p.y, p.x)),
        ):

            isv: bool = (
                pv.isVertically(verticalPoints[-1][-1])
                if len(verticalPoints) > 0
                else False
            )

            ish: bool = (
                ph.isHorizontally(horizontalPoints[-1][-1])
                if len(horizontalPoints) > 0
                else False
            )

            if not isv:
                verticalPoints.append([pv])
            else:
                verticalPoints[-1].append(pv)

            if not ish:
                horizontalPoints.append([ph])
            else:
                horizontalPoints[-1].append(ph)

        return verticalPoints, horizontalPoints

    @staticmethod
    def drawGridNetLines(
        verticalPoints: List[List["GridPoint"]],
        horizontalPoints: List[List["GridPoint"]],
        plt,
    ):

        # draw vertical line
        for v in verticalPoints:
            xs, ys = zip(*[(p.x, p.y) for p in v])
            plt.plot(xs, ys, color="black")
        # draw horizontal line
        for h in horizontalPoints:
            xs, ys = zip(*[(p.x, p.y) for p in h])
            plt.plot(xs, ys, color="black")


class GridPoint(GridNet):

    def __init__(self, coordinate: tuple[float, float], value: float) -> None:
        super().__init__()
        assert isinstance(coordinate, tuple) and coordinate.__len__() == 2
        assert isinstance(value, (int, float))
        self.x = coordinate[0]
        self.y = coordinate[1]
        self.coordinate = coordinate
        self.value = value

    def isZeroPoint(self) -> bool:

        return (
            (isinstance(self.x, float) or isinstance(self.y, float))
            and (
                abs(self.x) % self.edge_length != 0
                or abs(self.y) % self.edge_length != 0
            )
            and (self.value == 0)
        )

    def isVertically(self, other) -> bool:

        if isinstance(other, GridPoint):

            return self.x == other.x

        return False

    def isHorizontally(self, other) -> bool:

        if isinstance(other, GridPoint):

            return self.y == other.y

        return False

    def __eq__(self, other):
        if isinstance(other, GridPoint):
            return all(
                getattr(self, attr) == getattr(other, attr)
                for attr in ["x", "y", "value"]
            )
        return False

    def __repr__(self) -> str:
        return f"GridPoint(coordinate={self.coordinate}, value={self.value})"

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self):
        return hash((self.coordinate, self.value))


class GridPath(GridNet):

    def __init__(self, edge: Iterable[GridPoint]) -> None:
        super().__init__()
        self.edges: Iterable[GridPoint] = edge
        self.left_vertex: GridPoint = self.edges[0]
        self.right_vertex: GridPoint = self.edges[-1]

    def isAllPositive(self) -> bool:
        return all(p.value >= 0 for p in self.edges)

    def isAllNegative(self) -> bool:
        return all(p.value <= 0 for p in self.edges)

    def isAllSameSign(self) -> bool:
        return self.isAllNegative() or self.isAllPositive()

    def isClosed(self) -> bool:

        return self.left_vertex == self.right_vertex

    def __repr__(self) -> str:
        edge_repr = ", ".join(repr(p) for p in self.edges)
        return f"GridEdge([{edge_repr}])"

    def __str__(self) -> str:
        return self.__repr__()

    def containsAtLeastTwoZeroPointsAndContinuous(self) -> bool:
        count = 0
        prev_index = -1

        for index, p in enumerate(self.edges):
            if p.isZeroPoint():
                count += 1
                if count == 1:
                    prev_index = index
                elif index - prev_index != 1:
                    return False
                else:
                    prev_index = index

        return count >= 2

    def isSameSignPolygon(self):

        return (
            self.isClosed()
            and self.containsAtLeastTwoZeroPointsAndContinuous()
            and self.isAllSameSign()
        )

    def __getitem__(self, index: int) -> GridPoint:
        return self.edges[index]

    def __len__(self) -> int:
        return len(self.edges)


# 每个都是方格网的一个点, 左上角是原点 从左到右为y,从上到下为x轴
data = [
    [-0.44, -0.18, -0.47, 0.07, -0.71],
    [0.56, -0.09, 0.22, 0.14, 0.05],
    [-0.01, 0.12, -0.34, 0.05, -0.11],
    [0.05, 0.21, 0.07, 0.19, -0.23],
]


# (x,y)
zero_points_coordinates = [
    [0, 57.41],
    [0, 61.79],
    [20, 17.23],
    [20, 25.81],
    [40, 1.54],
    [40, 25.22],
    [40, 57.44],
    [40, 66.25],
    [60, 69.05],
    [8.8, 0],
    [39.65, 0],
    [43.33, 0],
    [28.57, 20],
    [13.62, 40],
    [27.86, 40.0],
    [56.59, 40.0],
    [18.68, 80.0],
    [26.25, 80.0],
]


negtive_poins: list[GridPoint] = []

positive_points: list[GridPoint] = []

zero_points: list[GridPoint] = []

for y in range(np.array(data).shape[0]):

    for x in range(np.array(data).shape[1]):

        coordinate = (y * GridNet().edge_length, x * GridNet().edge_length)

        value = data[y][x]

        point = GridPoint(coordinate=coordinate, value=value)

        if point.value >= 0:

            positive_points.append(point)

        else:

            negtive_poins.append(point)

for coordinate in zero_points_coordinates:

    zero_points.append(GridPoint(tuple(coordinate), 0))

vertexs = negtive_poins + positive_points


# for p in vertexs:

#     plt.scatter(p.x, p.y, color="blue" if p.value >= 0 else "red")

# 创建散点图
fig, ax = plt.subplots()


plt.scatter([p.x for p in zero_points], [p.y for p in zero_points], color="green")
# 标注每个点的坐标
for p in zero_points:
    plt.text(p.x, p.y, f'({p.x}, {p.y})')

verticalPoints, horizontalPoints = GridNet.getverticalPoints_and_horizontalPoints(
    vertexs=vertexs
)


GridNet.drawGridNetLines(
    verticalPoints=verticalPoints, horizontalPoints=horizontalPoints, plt=plt
)


# def test_isSameSignPolygon():

#     assert GridPath(zero_points + [zero_points[0]]).isSameSignPolygon() == True

#     assert GridPath(negtive_poins + [negtive_poins[0]]).isSameSignPolygon() == False
#     assert negtive_poins[1].isZeroPoint() == False
#     assert all(not p.isZeroPoint() for p in negtive_poins)

#     assert (
#         GridPath(
#             negtive_poins
#             + [zero_points[1]]
#             + negtive_poins
#             + [zero_points[1]]
#             + [negtive_poins[0]]
#         ).isSameSignPolygon()
#     ) == False

#     assert (
#         GridPath(
#             negtive_poins
#             + [zero_points[1]]
#             + [zero_points[1]]
#             + [zero_points[1]]
#             + [zero_points[1]]
#             + [negtive_poins[0]]
#         ).isSameSignPolygon()
#     ) == True

#     assert (
#         GridPath(
#             negtive_poins + [zero_points[1]] + [zero_points[1]] + [negtive_poins[0]]
#         ).isSameSignPolygon()
#     ) == True


# test_isSameSignPolygon()

gridNet = GridNet(verticalPoints=verticalPoints, horizontalPoints=horizontalPoints)


grids = gridNet.grids


zero_points = sorted(zero_points, key=lambda p: (p.x, p.y))

# 生成随机颜色
# colors = np.random.rand(20, 3)  # 生成10个RGB颜色
insertededgeallsize=0
sizeall = 0
for rs in grids:
    # print(len(rs))
    for grid in rs:  # edge
        # print(grid.path)
        # xs, ys = zip(*[(p.x, p.y) for p in grid.path])
        plt.text(
            grid.center_coordinate[0],
            grid.center_coordinate[1],
            f"{grid.index+1}",
            color="black",
            weight="bold",
        )

        # edges = grid.path  # 闭合边

        # edges_coordinates = [(p.x, p.y) for p in edges]

        # edges_point_list = list(edges)

        # print(edges_coordinates)
        
        edges = grid.path  # 闭合边


        for zero in zero_points:
            
            #edges_point_list = list(edges)
            
            _, insertededge = is_point_in_polygon_or_on_edge(
                zero.x, zero.y, [(p.x, p.y) for p in edges]
            )  #

            if insertededge:
                
                #insertededgeallsize+=1
                
                #print(insertededge)

                # plt.plot(*zip(*insertededge),color="red",linestyle="->")

                # 绘制箭头
                # point_A, point_B = insertededge
                # plt.arrow(point_A[0], point_A[1], point_B[0] - point_A[0], point_B[1] - point_A[1],
                #        head_width=0.3, head_length=1, fc='red', ec='blue')

                # plt.quiver(
                #     point_A[0],
                #     point_A[1],
                #     point_B[0] - point_A[0],
                #     point_B[1] - point_A[1],
                #     angles="xy",
                #     scale_units="xy",
                #     scale=1,
                #     color=np.random.rand(20, 3) 
                # )
                
                #index1 = edges_coordinates.index(insertededge[0])

                #index2 = edges_coordinates.index(insertededge[1])
                
                # 确认点A和点B在路径中的相对位置
                #if (index1 == index2 + 1) or (index1 == len(edges_coordinates) - 1 and index2 == 0):  # A在B之前
                #    insert_position = index1# + 1
                #else:  # B在A之前
                #    insert_position = index2# + 1
                
                #print(edges_coordinates)

                # print(index1,index2)
                # print(insertededge,(zero.x,zero.y))

                grid.insertPointInEdgePath(point=zero,index=len(grid.path))
        
        
        edges_coordinates = [(p.x, p.y) for p in grid.path]
        
        # 定义重排序函数
        
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
        
        print(sort_closepath(edges_coordinates))
        
        # #print([(p.x,p.y) for p in grid.path])
        # order_dict = {value: index for index, value in enumerate(reorder_path(edges_coordinates))}
        
        # print(order_dict)
        # #sorted(sequence, key=lambda x: order_dict.get(x, len(order)))

        # grid.path = GridPath(edge=sorted(grid.path, key=lambda x: order_dict.get(x.coordinate, len(grid.path))))
        
        #print([(p.x,p.y) for p in grid.path])

        # points = sorted([p for p in grid.path] , key= lambda p:(p.x,p.y))

        # print(np.array(points).size)
        # sizeall += np.array(points).size

        # xs, ys = zip(*[(p.x, p.y) for p in points])

        # plt.plot(xs,ys,color="black")

# print(sizeall, len(data) + len(zero_points))
# assert sizeall == (len(vertexs) + len(zero_points))

# print(insertededgeallsize,len(zero_points))
plt.gca().invert_yaxis()
# 将 x 轴放到上面
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
plt.show()
