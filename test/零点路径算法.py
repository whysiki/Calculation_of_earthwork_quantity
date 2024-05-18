import numpy as np
import matplotlib.pyplot as plt
from rich import print
from typing import Iterable, List, Union
import math
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import matplotlib.gridspec as gridspec
from dataProcess import 施工标高


def is_point_on_edge(point: tuple[float, float], polygon: Polygon):

    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]

        vec1 = (p2[0] - p1[0], p2[1] - p1[1])
        vec2 = (point[0] - p1[0], point[1] - p1[1])

        cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]

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

        self.vertexs: tuple[GridPoint, GridPoint, GridPoint, GridPoint] = (
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

        self.insertedPoints: List["GridPoint"] = []

        self.splitZones: list["GridPath"] = []

        self.negtiveZones: list["GridPath"] = []

        self.positiveZones: list["GridPath"] = []

        self.gridPolygon = Polygon(
            [(p.x, p.y) for p in list(self.vertexs) + [self.left_upper]]
        )

        # excavation and filling
        self.excavationAmount = None

        self.fillingAmount = None

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

        self.insertedPoints.append(point)


class GridNet:

    def __init__(
        self,
        verticalPoints=None,
        horizontalPoints=None,
        zeropoints=None,
        positivePoints=None,
        negativePoints=None,
        vertexs=None,
    ) -> None:

        self.edge_length = 20

        self.verticalPoints: list[list[GridPoint]] = (
            verticalPoints if verticalPoints else []
        )  # 表示列视角下每列的的所有顶点

        self.horizontalPoints: list[list[GridPoint]] = (
            horizontalPoints if horizontalPoints else []  # 行视角下每行的顶点
        )

        assert (
            np.array(self.verticalPoints).size == np.array(self.horizontalPoints).size
        )

        self.GridNetPositivePoints: list[GridPoint] = (
            positivePoints if positivePoints else []
        )
        self.GridNetNegativePoints: list[GridPoint] = (
            negativePoints if negativePoints else []
        )

        self.GridNetVertexs: list[GridPoint] = vertexs if vertexs else []

        self.grids: list[list[Grid]] = (
            self.constructGridsNet() if verticalPoints and horizontalPoints else []
        )

        self.gridNetAllzero_points: list[GridPoint] = zeropoints if zeropoints else []

        self.gridNetallpoints = (
            [p for ps in self.verticalPoints for p in ps]
            # + [p for ps in self.horizontalPoints for p in ps]
            + self.gridNetAllzero_points
        )

        if self.gridNetAllzero_points:

            assert all(isinstance(p, GridPoint) for p in self.gridNetallpoints)

        self.gridNetNegativeZone: list[GridPath] = []

        self.gridNetPositiveZone: list[GridPath] = []

        self.fillingAll = 0

        self.excavationAll = 0

    def get_value_by_coordinate(self, coordinate: tuple[float, float]) -> float:

        for p in self.gridNetallpoints:

            if p.coordinate == coordinate:
                return p.value

    def get_point_by_coordinate(self, coordinate: tuple[float, float]) -> "GridPoint":

        for p in self.gridNetallpoints:

            if p.coordinate == coordinate:

                return p

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

    def getGridNetZeroPath(self) -> List["GridPath"]:

        paths: List[GridPath] = []

        for path in self.gridNetNegativeZone:

            # 迭代过程中不能改变正在迭代的集合的大小
            for p in path.edges.copy():

                if not p.isZeroPoint():

                    pass

                    # path.deletePoint(p)

            # 路径删除重复的点
            # path.edges = list(set(path.edges))

            # 排序
            # path.edges = sorted(path.edges, key=lambda p: (p.x, p.y))

            paths.append(path)

        return paths

    # def calculateAmount of excavation and filling
    def calculate_area(self, intersection):
        if not intersection:
            return 0
        if isinstance(intersection, Polygon):
            polygons = [intersection]

        elif isinstance(intersection, MultiPolygon):
            polygons = intersection.geoms
        else:
            return 0

        # print(polygons)

        total_area = 0
        for poly in polygons:
            set_coords = set(i for i in poly.exterior.coords)

            # print(type(poly))
            # print(set_coords)
            # print(poly.area)
            total_area += poly.area * (
                sum(
                    self.get_value_by_coordinate(coordinate=coordinate)
                    for coordinate in set_coords
                )
                / len(set_coords)
            )
        return total_area

    def calculateAmountExcavationAndFilling_for_all_girds(self):

        fillingAll = 0
        excavationAll = 0
        for rs in self.grids:
            for grid in rs:
                gridPolygon: Polygon = grid.gridPolygon

                excavation = sum(
                    self.calculate_area(
                        gridPolygon.intersection(
                            Polygon([(p.x, p.y) for p in negativeZonePath.edges])
                        )
                    )
                    for negativeZonePath in self.gridNetNegativeZone
                )

                filling = sum(
                    self.calculate_area(
                        gridPolygon.intersection(
                            Polygon([(p.x, p.y) for p in positiveZonePath.edges])
                        )
                    )
                    for positiveZonePath in self.gridNetPositiveZone
                )

                filling = round(filling, 3)

                excavation = round(excavation, 3)

                print(
                    f"grid {grid.index+1} excavation: {excavation}, filling: {filling}"
                )

                grid.excavationAmount = excavation

                grid.fillingAmount = filling

                fillingAll += abs(filling)
                excavationAll += abs(excavation)

        self.fillingAll = round(fillingAll, 3)

        self.excavationAll = round(excavationAll, 3)

        print(f"excavationAll: - {self.excavationAll}, fillingAll: + {self.fillingAll}")


class GridPoint:

    def __init__(self, coordinate: tuple[float, float], value: float) -> None:
        assert isinstance(coordinate, tuple) and coordinate.__len__() == 2
        assert isinstance(value, (int, float))
        self.x = coordinate[0]
        self.y = coordinate[1]
        self.coordinate = coordinate
        self.value = value
        self.edge_length = GridNet().edge_length

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

    def getVerticalPointCoordinates(self) -> tuple[tuple, tuple]:

        up = (self.x, math.ceil(self.y / self.edge_length) * self.edge_length)
        down = (self.x, math.floor(self.y / self.edge_length) * self.edge_length)

        return up, down

    def getHorizontalPointCoordinates(self) -> tuple[tuple, tuple]:

        left = (math.floor(self.x / self.edge_length) * self.edge_length, self.y)
        right = (math.ceil(self.x / self.edge_length) * self.edge_length, self.y)

        return left, right

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


class GridPath:

    def __init__(self, edge: List[GridPoint]) -> None:

        self.edges: List[GridPoint] = edge if len(edge) > 0 else []

        self.edge_length = GridNet().edge_length

    def isAllPositive(self) -> bool:

        return all(p.value >= 0 for p in self.edges if p.isZeroPoint() == False)

    def isAllNegative(self) -> bool:

        return all(p.value <= 0 for p in self.edges if p.isZeroPoint() == False)

    def isAllSameSign(self) -> bool:

        return self.isAllNegative() or self.isAllPositive()

    def isClosed(self) -> bool:

        return self.edges[0] == self.edges[-1] if self.edges else False

    def __repr__(self) -> str:
        edge_repr = ",\n ".join(repr(p) for p in self.edges)
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

    def get_index_by_point(self, point: GridPoint):

        try:
            return self.edges.index(point)
        except ValueError:
            return -1

    def insertPoint(self, point: GridPoint, index: int):

        old_len = len(self.edges)
        self.edges.insert(index, point)
        new_len = len(self.edges)

        assert new_len - old_len == 1

    def insertPointStart(self, point: GridPoint):

        old_len = len(self.edges)
        self.edges.insert(0, point)
        new_len = len(self.edges)

        assert new_len - old_len == 1

    def insertPointEnd(self, point: GridPoint):

        old_len = len(self.edges)
        # self.edges.insert(0,point)
        self.edges.append(point)
        new_len = len(self.edges)

        assert new_len - old_len == 1

    def deletePoint(self, point: GridPoint):

        old_len = len(self.edges)

        self.edges.remove(point)

        new_len = len(self.edges)

        assert old_len - new_len == 1

    def __iter__(self):

        return iter(self.edges)


def find_zero_points(
    gridNetValueMatrix, net_length: int = GridNet().edge_length
) -> List[tuple[float, float]]:

    net_length = net_length

    zero_points = []
    rows, cols = len(gridNetValueMatrix), len(gridNetValueMatrix[0])

    # 遍历每一行，寻找水平边的零点
    for i in range(rows):
        for j in range(cols - 1):
            if (gridNetValueMatrix[i][j] > 0 and gridNetValueMatrix[i][j + 1] < 0) or (
                gridNetValueMatrix[i][j] < 0 and gridNetValueMatrix[i][j + 1] > 0
            ):
                # zero_points.append((20 * j + 10, 20 * i))
                point = (
                    net_length * i,
                    net_length * j
                    + net_length
                    * (
                        abs(gridNetValueMatrix[i][j])
                        / (
                            abs(gridNetValueMatrix[i][j])
                            + abs(gridNetValueMatrix[i][j + 1])
                        )
                    ),
                )
                point = round(point[0], 2), round(point[1], 2)
                zero_points.append(point)

    # 遍历每一列，寻找垂直边的零点
    for j in range(cols):
        for i in range(rows - 1):
            if (gridNetValueMatrix[i][j] > 0 and gridNetValueMatrix[i + 1][j] < 0) or (
                gridNetValueMatrix[i][j] < 0 and gridNetValueMatrix[i + 1][j] > 0
            ):
                # zero_points.append((20 * j, 20 * i + 10))
                point = (
                    net_length * i
                    + net_length
                    * (
                        abs(gridNetValueMatrix[i][j])
                        / (
                            abs(gridNetValueMatrix[i][j])
                            + abs(gridNetValueMatrix[i + 1][j])
                        )
                    ),
                    net_length * j,
                )
                point = round(point[0], 2), round(point[1], 2)
                zero_points.append(point)

    return zero_points


if __name__ == "__main__":

    # 每个都是方格网的一个点, 左上角是原点 从左到右为y,从上到下为x轴
    # 施工标高数据
    # np.array(设计标高) - np.array(原场地标高)
    # data = [
    #     [-0.44, -0.18, -0.47, 0.07, -0.71],
    #     [0.56, -0.09, 0.22, 0.14, 0.05],
    #     [-0.01, 0.12, -0.34, 0.05, -0.11],
    #     [0.05, 0.21, 0.07, 0.19, -0.23],
    # ]

    data = 施工标高.tolist()

    # 保留两位小数
    data = [[round(value, 2) for value in row] for row in data]

    # print(data)

    zero_points_coordinates = find_zero_points(data)

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

    fig, ax = plt.subplots()

    # 网格线
    verticalPoints, horizontalPoints = GridNet.getverticalPoints_and_horizontalPoints(
        vertexs=vertexs
    )

    # 网格网络
    gridNet = GridNet(
        verticalPoints=verticalPoints,
        horizontalPoints=horizontalPoints,
        zeropoints=zero_points.copy(),
        positivePoints=positive_points,
        negativePoints=negtive_poins,
        vertexs=vertexs,
    )

    grids = gridNet.grids

    zero_points = sorted(zero_points, key=lambda p: (p.x, p.y))

    for rs in grids:
        for grid in rs:  # edge
            plt.text(
                grid.center_coordinate[0],
                grid.center_coordinate[1],
                f"{grid.index+1}",
                color="black",
                weight="bold",
            )

            edges = grid.path  # 闭合边

            for zero in zero_points:

                _, insertededge = is_point_in_polygon_or_on_edge(
                    zero.x, zero.y, [(p.x, p.y) for p in edges]
                )

                if insertededge:

                    grid.insertPointInEdgePath(point=zero, index=len(grid.path))

            edges_coordinates = [(p.x, p.y) for p in grid.path]

            def sort_closepath(points):
                min_x = min(points, key=lambda p: p[0])[0]
                max_x = max(points, key=lambda p: p[0])[0]
                min_y = min(points, key=lambda p: p[1])[1]
                max_y = max(points, key=lambda p: p[1])[1]

                # 四个角点
                left_upper = (min_x, min_y)
                right_upper = (max_x, min_y)
                right_lower = (max_x, max_y)
                left_lower = (min_x, max_y)

                # 四条边
                edge1 = []
                edge2 = []
                edge3 = []
                edge4 = []

                for point in points:
                    if (
                        point != left_upper
                        and point != right_upper
                        and point != right_lower
                        and point != left_lower
                    ):
                        if point[1] == min_y and min_x < point[0] < max_x:  # Top edge
                            edge1.append(point)
                        elif (
                            point[0] == max_x and min_y < point[1] < max_y
                        ):  # Right edge
                            edge2.append(point)
                        elif (
                            point[1] == max_y and min_x < point[0] < max_x
                        ):  # Bottom edge
                            edge3.append(point)
                        elif (
                            point[0] == min_x and min_y < point[1] < max_y
                        ):  # Left edge
                            edge4.append(point)

                # 对边上的点进行排序

                edge1.sort(key=lambda x: x[0])  # Sort by x-coordinate for top edge
                edge2.sort(key=lambda x: x[1])  # Sort by y-coordinate for right edge
                edge3.sort(
                    key=lambda x: x[0], reverse=True
                )  # Sort by x-coordinate descending for bottom edge
                edge4.sort(
                    key=lambda x: x[1], reverse=True
                )  # Sort by y-coordinate descending for left edge

                # 连接所有点形成闭合路径
                reordered_points = (
                    [left_upper]
                    + edge1
                    + [right_upper]
                    + edge2
                    + [right_lower]
                    + edge3
                    + [left_lower]
                    + edge4
                    + [left_upper]
                )

                assert len(reordered_points) == len(points)

                return reordered_points

            sorted_coordinates = sort_closepath(edges_coordinates)

            def sort_yeild(coordinate, order_index: dict, counter: dict) -> int:

                if coordinate in order_index:
                    indices = order_index[coordinate]
                    index = indices[counter[coordinate] % len(indices)]
                    counter[coordinate] += 1
                    return index
                else:

                    return float("inf")

            order_index = {}
            for index, value in enumerate(sorted_coordinates):
                if value not in order_index:
                    order_index[value] = []
                order_index[value].append(index)
            counter = {key: 0 for key in order_index}

            grid.path = GridPath(
                edge=sorted(
                    grid.path.edges,
                    key=lambda x: sort_yeild(
                        x.coordinate,
                        order_index=order_index,
                        counter=counter,
                    ),
                )
            )

    for rs in grids:

        for grid in rs:  # edge

            split_zones: list[GridPath] = []

            count_zeropoints_time = 0

            last_end_index = 0

            maxzeroindex = max(
                [grid.path.get_index_by_point(p) for p in grid.path if p.isZeroPoint()]
            )

            maxindex = maxzeroindex if maxzeroindex > 0 else len(grid.path)

            minzeroindex = min(
                [grid.path.get_index_by_point(p) for p in grid.path if p.isZeroPoint()]
            )

            minindex = minzeroindex if minzeroindex > 0 else 0

            index = minindex

            begin = True

            while grid.path[index % len(grid.path)] != grid.path[minindex + 1] or begin:
                if len(split_zones) == 0:
                    split_zones.append(GridPath([]))
                if grid.path[index % len(grid.path)].isZeroPoint():
                    count_zeropoints_time += 1
                if count_zeropoints_time == 1:
                    if (
                        split_zones[-1].get_index_by_point(
                            grid.path[index % len(grid.path)]
                        )
                        == -1
                    ):
                        split_zones[-1].insertPointEnd(
                            grid.path[index % len(grid.path)]
                        )
                    index += 1
                    continue
                if count_zeropoints_time == 2:
                    if (
                        split_zones[-1].get_index_by_point(
                            grid.path[index % len(grid.path)]
                        )
                        == -1
                    ):
                        split_zones[-1].insertPointEnd(
                            grid.path[index % len(grid.path)]
                        )
                    split_zones.append(GridPath([]))
                    begin = False
                    count_zeropoints_time = 0

            split_zones = [path for path in split_zones if len(path) > 2]

            split_zones_closed = [
                GridPath(path.edges + [path.edges[0]]) for path in split_zones
            ]

            split_zones_closed = [
                path for path in split_zones_closed if path.isAllSameSign()
            ]

            grid.negtiveZones = [
                path for path in split_zones_closed if path.isAllNegative()
            ]

            grid.positiveZones = [
                path for path in split_zones_closed if path.isAllPositive()
            ]

    def merge_polygons_and_store(
        polygons: list[Polygon], gridNet: GridNet, attribute_name: str
    ):

        merged_polygon = unary_union(polygons)

        merged_vertices = []
        if isinstance(merged_polygon, Polygon):
            merged_vertices = list(merged_polygon.exterior.coords)
        elif isinstance(merged_polygon, MultiPolygon):
            for poly in merged_polygon.geoms:
                merged_vertices.append(list(poly.exterior.coords))

        setattr(
            gridNet,
            attribute_name,
            [
                GridPath(
                    edge=[gridNet.get_point_by_coordinate(vertex) for vertex in vertexs]
                )
                for vertexs in merged_vertices
            ],
        )

    negtive_polygons: list[Polygon] = [
        Polygon([(p.x, p.y) for p in path.edges])
        for rs in grids
        for grid in rs
        for path in grid.negtiveZones
    ]

    positive_polygons: list[Polygon] = [
        Polygon([(p.x, p.y) for p in path.edges])
        for rs in grids
        for grid in rs
        for path in grid.positiveZones
    ]

    merge_polygons_and_store(negtive_polygons, gridNet, "gridNetNegativeZone")

    merge_polygons_and_store(positive_polygons, gridNet, "gridNetPositiveZone")

    zero_paths = gridNet.getGridNetZeroPath()

    plt.scatter(
        [p.x for p in gridNet.gridNetAllzero_points],
        [p.y for p in gridNet.gridNetAllzero_points],
        color="green",
        s=100,
    )

    plt.scatter(
        [p.x for p in gridNet.GridNetNegativePoints],
        [p.y for p in gridNet.GridNetNegativePoints],
        color="red",
        s=100,
    )

    plt.scatter(
        [p.x for p in gridNet.GridNetPositivePoints],
        [p.y for p in gridNet.GridNetPositivePoints],
        color="blue",
        s=100,
    )

    # 标注每个零点的坐标
    for p in gridNet.gridNetAllzero_points:
        plt.text(
            p.x,
            p.y,
            f"({p.x}, {p.y})",
            # color="red" if p.value < 0 else "blue",
            # weight="bold",
        )

    # 标准每个角点的值
    for p in gridNet.GridNetVertexs:
        plt.text(
            p.x,
            p.y,
            f"{p.value}",
            color="red" if p.value < 0 else "blue",
            weight="bold",
        )

    gridNet.drawGridNetLines(gridNet.verticalPoints, gridNet.horizontalPoints, plt)

    def showNegativeAndPositivePath(gridNet: GridNet):

        fig12, ax12 = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        gridNet.drawGridNetLines(
            gridNet.verticalPoints, gridNet.horizontalPoints, ax12[0]
        )
        gridNet.drawGridNetLines(
            gridNet.verticalPoints, gridNet.horizontalPoints, ax12[1]
        )

        ax12[0].scatter(
            [p.x for p in gridNet.gridNetAllzero_points],
            [p.y for p in gridNet.gridNetAllzero_points],
            color="green",
            s=100,
        )

        ax12[1].scatter(
            [p.x for p in gridNet.gridNetAllzero_points],
            [p.y for p in gridNet.gridNetAllzero_points],
            color="green",
            s=100,
        )

        for path in gridNet.gridNetNegativeZone:

            # print(path)

            xs, ys = zip(*[(p.x, p.y) for p in path])

            # fill
            ax.fill(xs, ys, color="red", alpha=0.5, label="negative zone")

            # plt.plot(xs, ys, color="red")
            ax12[0].plot(xs, ys, color="red")

            # 设置图表标题
            ax12[0].set_title("Negative Path")

        for path in gridNet.gridNetPositiveZone:

            # print(path)

            xs, ys = zip(*[(p.x, p.y) for p in path])

            ax.fill(xs, ys, color="blue", alpha=0.5, label="positive zone")

            ax12[1].plot(xs, ys, color="blue")

            # 设置图表标题

            ax12[1].set_title("Positive Path")

        # fig12.gca().invert_yaxis()

        for i in range(2):
            ax12[i].xaxis.set_ticks_position("top")
            ax12[i].xaxis.set_label_position("top")
            # ax12[i].gca().invert_yaxis()
            ax12[i].invert_yaxis()

    gridNet.calculateAmountExcavationAndFilling_for_all_girds()

    def showData(gridNet: GridNet):

        # 创建带有指定网格布局的图形
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # 定义两列，第一列比第二列宽

        # 在第一列创建轴用于绘图
        ax = fig.add_subplot(gs[0])

        # 假设有网格对象和相关方法已经定义
        gridNet.drawGridNetLines(gridNet.verticalPoints, gridNet.horizontalPoints, ax)
        ax.invert_yaxis()
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

        # 数据表格列表
        cell_text = []

        for rs in gridNet.grids:
            for grid in rs:
                textindex = f"{grid.index+1}"
                textexcavation = f"-{abs(grid.excavationAmount)}"
                textfilling = f"+{abs(grid.fillingAmount)}"
                ax.text(
                    grid.center_coordinate[0],
                    grid.center_coordinate[1],
                    textindex,
                    color="black",
                    ha="center",
                    va="center",
                    weight="bold",
                )
                ax.text(
                    grid.center_coordinate[0],
                    grid.center_coordinate[1] + GridNet().edge_length / 6,
                    textexcavation,
                    color="red",
                    ha="center",
                    va="center",
                    weight="bold",
                )
                ax.text(
                    grid.center_coordinate[0],
                    grid.center_coordinate[1] + GridNet().edge_length / 4,
                    textfilling,
                    color="blue",
                    ha="center",
                    va="center",
                    weight="bold",
                )
                # 收集数据到表格数据列表
                cell_text.append([textindex, textexcavation, textfilling])

        cell_text.append(
            ["Total", f"-{gridNet.excavationAll}", f"+{gridNet.fillingAll}"]
        )

        # 添加表头
        cell_text.insert(0, ["Index", "Excavation", "Filling"])

        # 在第二列创建轴用于显示表格
        ax2 = fig.add_subplot(gs[1])
        ax2.axis("off")  # 隐藏坐标轴
        table = ax2.table(
            cellText=cell_text,
            colLabels=cell_text.pop(0),
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)  # 调整表格行高

        ax.set_title("Grid Excavation and Filling Data Table")

        # 调整整个图形布局
        plt.tight_layout()

    showNegativeAndPositivePath(gridNet)

    showData(gridNet)

    ax.set_title("Design elevation minus original site elevation")

    ax.legend(
        loc="center left", bbox_to_anchor=(1, 0.5)
    )  # 表示图例框的中心应位于图表框（axes bounding box）的右侧中心（1 指的是完全在右边界外，0.5 表示垂直居中）。
    fig.subplots_adjust(right=0.75)
    fig.gca().invert_yaxis()
    # 将 x 轴放到上面
    ax.xaxis.set_ticks_position("top")

    ax.xaxis.set_label_position("top")

    plt.show()
