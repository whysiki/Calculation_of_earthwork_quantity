import numpy as np
from rich import print
from typing import Iterable, List, Union
import math
from shapely.geometry import Point, Polygon, MultiPolygon
from dataProcess import lenth_net
from scipy.spatial import ConvexHull
from shapely.ops import unary_union


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


def sort_yeild(
    coordinate: tuple[float, float], order_index: dict, counter: dict
) -> int:

    if coordinate in order_index:
        indices = order_index[coordinate]
        index = indices[counter[coordinate] % len(indices)]
        counter[coordinate] += 1
        return index
    else:

        return float("inf")


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

    @staticmethod
    def sort_closepath(points: list[tuple[float, float]]) -> list[tuple[float, float]]:

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
                elif point[0] == max_x and min_y < point[1] < max_y:  # Right edge
                    edge2.append(point)
                elif point[1] == max_y and min_x < point[0] < max_x:  # Bottom edge
                    edge3.append(point)
                elif point[0] == min_x and min_y < point[1] < max_y:  # Left edge
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

        self.edge_length = lenth_net

        self.verticalPoints: list[list[GridPoint]] = (
            verticalPoints if verticalPoints else []
        )  # 表示列视角下每列的的所有顶点

        self.horizontalPoints: list[list[GridPoint]] = (
            horizontalPoints if horizontalPoints else []  # 行视角下每行的顶点
        )

        assert (
            np.array(self.verticalPoints).size == np.array(self.horizontalPoints).size
        ), "The number of vertical points and horizontal points must be the same"

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

        if self.gridNetAllzero_points:

            assert all(
                isinstance(p, GridPoint) and p.isZeroPoint()
                for p in self.gridNetAllzero_points
            ), "gridNetAllzero_points must be a list of GridPoint and isZeroPoint"

        self.gridNetallpoints = (
            [p for ps in self.verticalPoints for p in ps]
            # + [p for ps in self.horizontalPoints for p in ps]
            + self.gridNetAllzero_points
        )

        if self.gridNetAllzero_points:

            assert all(
                isinstance(p, GridPoint) for p in self.gridNetallpoints
            ), "gridNetallpoints must be a list of GridPoint"

        self.gridNetNegativeZone: list[GridPath] = []

        self.gridNetPositiveZone: list[GridPath] = []

        self.fillingAll = 0

        self.excavationAll = 0

    def get_value_by_coordinate(self, coordinate: tuple[float, float]) -> float:

        assert self.gridNetallpoints, "gridNetallpoints is empty"

        for p in self.gridNetallpoints:

            if p.coordinate == coordinate:
                return p.value

        raise ValueError("coordinate not found")

    def get_point_by_coordinate(self, coordinate: tuple[float, float]) -> "GridPoint":

        assert self.gridNetallpoints, "gridNetallpoints is empty"

        for p in self.gridNetallpoints:

            if p.coordinate == coordinate:

                return p

        raise ValueError("coordinate not found")

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

            set_coords = list(
                set(i for i in poly.exterior.coords)
            )  # 闭合图形的坐标点数等于边长数 ，去重

            # print(type(poly))

            # print(set_coords)

            # print(list(poly.exterior.coords))

            # print(poly.area)

            valuelist = [
                self.get_value_by_coordinate(coordinate=coordinate)
                for coordinate in set_coords
            ]

            average_value = sum(valuelist) / len(set_coords)  # 平均高程

            points3d = [
                (p[0], p[1], self.get_value_by_coordinate(p)) for p in set_coords
            ]

            # print(valuelist)

            # print(set_coords)

            # try:
            #     hull = ConvexHull(points3d)
            #     v = hull.volume
            # except:
            #     v = poly.area * average_value

            v = poly.area * average_value

            total_area += v

            # test
            # print(valuelist)
            # fig, ax = plt.subplots()
            # self.drawGridNetLines(self.verticalPoints, self.horizontalPoints, ax)
            # x, y = poly.exterior.xy
            # color = "red" if min(valuelist) < 0 else "blue"
            # ax.fill(x, y, alpha=0.5, fc=color, ec="none")
            # ax.invert_yaxis()
            # # ax.set_aspect("equal", "box")
            # ax.xaxis.set_ticks_position("top")

            # ax.xaxis.set_label_position("top")
            # plt.show()

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

    def calculateNegtiveZonesAndPositiveZones_geoCenterpoint_and_v(
        self,
    ) -> tuple[
        dict["GridPath", tuple[Point, float]], dict["GridPath", tuple[Point, float]]
    ]:
        """

        gridNetNegativeZone_c_v = {} # key: GridPath, value: (geoCenterPoint, v)

        gridNetPositiveZone_c_v = {} # key: GridPath, value: (geoCenterPoint, v)

        return gridNetNegativeZone_c_v, gridNetPositiveZone_c_v

        """
        assert self.gridNetNegativeZone and self.gridNetPositiveZone, "empty zones"

        gridNetNegativeZone_c_v = {}
        gridNetPositiveZone_c_v = {}

        allvNegative = 0
        allvPositive = 0

        for path in self.gridNetNegativeZone:
            polygon = Polygon([(p.x, p.y) for p in path.edges])
            geoCenterPoint = polygon.centroid
            v = self.calculate_area(polygon)
            allvNegative += v
            gridNetNegativeZone_c_v[path] = (geoCenterPoint, v)

        for path in self.gridNetPositiveZone:
            polygon = Polygon([(p.x, p.y) for p in path.edges])
            geoCenterPoint = polygon.centroid
            v = self.calculate_area(polygon)
            allvPositive += v
            gridNetPositiveZone_c_v[path] = (geoCenterPoint, v)

        assert len(self.gridNetNegativeZone) == len(
            gridNetNegativeZone_c_v
        ), "negative zones not calculated"

        assert len(self.gridNetPositiveZone) == len(
            gridNetPositiveZone_c_v
        ), "positive zones not calculated"

        return gridNetNegativeZone_c_v, gridNetPositiveZone_c_v

    @staticmethod
    def find_zero_points(
        gridNetValueMatrix: list[list[float]], net_length: int
    ) -> list["GridPoint"]:

        net_length = net_length

        zero_points_coordinates = []
        rows, cols = len(gridNetValueMatrix), len(gridNetValueMatrix[0])

        # 遍历每一行，寻找水平边的零点
        for i in range(rows):
            for j in range(cols - 1):
                if (
                    gridNetValueMatrix[i][j] > 0 and gridNetValueMatrix[i][j + 1] < 0
                ) or (
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
                    zero_points_coordinates.append(point)

        # 遍历每一列，寻找垂直边的零点
        for j in range(cols):
            for i in range(rows - 1):
                if (
                    gridNetValueMatrix[i][j] > 0 and gridNetValueMatrix[i + 1][j] < 0
                ) or (
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
                    zero_points_coordinates.append(point)

        zero_points: list[GridPoint] = []

        for coordinate in zero_points_coordinates:

            zero_points.append(GridPoint(tuple(coordinate), 0))

        return zero_points

    @staticmethod
    def findAllNegtive_and_Positive_and_Vertexs_Points(
        gridNetValueMatrix: list[list[float]], net_length: int
    ) -> tuple[list["GridPoint"], list["GridPoint"], list["GridPoint"]]:

        negtive_points: list[GridPoint] = []

        positive_points: list[GridPoint] = []

        for y in range(np.array(gridNetValueMatrix).shape[0]):

            for x in range(np.array(gridNetValueMatrix).shape[1]):

                coordinate = (y * net_length, x * net_length)

                value = gridNetValueMatrix[y][x]

                point = GridPoint(coordinate=coordinate, value=value)

                if point.value >= 0:

                    positive_points.append(point)

                else:

                    negtive_points.append(point)

        vertexs = negtive_points + positive_points

        return negtive_points, positive_points, vertexs

    def construct_grids_and_insert_zero_points(self):

        for rs in self.grids:

            for grid in rs:  # edge

                edges = grid.path  # 闭合边

                for zero in self.gridNetAllzero_points:

                    _, insertededge = is_point_in_polygon_or_on_edge(
                        zero.x, zero.y, [(p.x, p.y) for p in edges]
                    )

                    if insertededge:

                        grid.insertPointInEdgePath(point=zero, index=len(grid.path))

                edges_coordinates = [(p.x, p.y) for p in grid.path]

                sorted_coordinates = Grid.sort_closepath(edges_coordinates)

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

                assert grid.path.isClosed(), "path is not closed"

    def caculate_for_each_grid_negtive_positive_zone(self):

        for rs in self.grids:

            for grid in rs:  # edge

                split_zones: list[GridPath] = []

                count_zeropoints_time = 0

                last_end_index = 0

                maxzeroindex = max(
                    [
                        grid.path.get_index_by_point(p)
                        for p in grid.path
                        if p.isZeroPoint()
                    ]
                )

                maxindex = maxzeroindex if maxzeroindex > 0 else len(grid.path)

                minzeroindex = min(
                    [
                        grid.path.get_index_by_point(p)
                        for p in grid.path
                        if p.isZeroPoint()
                    ]
                )

                minindex = minzeroindex if minzeroindex > 0 else 0

                index = minindex

                begin = True

                while (
                    grid.path[index % len(grid.path)] != grid.path[minindex + 1]
                    or begin
                ):
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

                assert all(
                    path.isClosed() for path in split_zones_closed
                ), "split_zones_closed is not closed"

                split_zones_closed = [
                    path for path in split_zones_closed if path.isAllSameSign()
                ]

                grid.negtiveZones = [
                    path for path in split_zones_closed if path.isAllNegative()
                ]

                grid.positiveZones = [
                    path for path in split_zones_closed if path.isAllPositive()
                ]

    def caculate_gridNetNegativeZone_gridNetPositiveZone(self):

        def get_merged_polygon_Zones(polygons: list[Polygon]):

            merged_polygon = unary_union(polygons)

            merged_vertices = []
            if isinstance(merged_polygon, Polygon):
                merged_vertices = list(merged_polygon.exterior.coords)
            elif isinstance(merged_polygon, MultiPolygon):
                for poly in merged_polygon.geoms:
                    merged_vertices.append(list(poly.exterior.coords))

            return [
                GridPath(
                    edge=[self.get_point_by_coordinate(vertex) for vertex in vertexs]
                )
                for vertexs in merged_vertices
            ]

        assert self.grids, "grids is empty"

        negtive_polygons: list[Polygon] = [
            Polygon([(p.x, p.y) for p in path.edges])
            for rs in self.grids
            for grid in rs
            for path in grid.negtiveZones
        ]

        positive_polygons: list[Polygon] = [
            Polygon([(p.x, p.y) for p in path.edges])
            for rs in self.grids
            for grid in rs
            for path in grid.positiveZones
        ]

        assert negtive_polygons and positive_polygons, "empty polygons"

        self.gridNetNegativeZone = get_merged_polygon_Zones(negtive_polygons)

        self.gridNetPositiveZone = get_merged_polygon_Zones(positive_polygons)


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
