import numpy as np
import matplotlib.pyplot as plt
from rich import print
from typing import Iterable, List, Union
import math
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

#
#
from gridClass import *
from dataProcess import 施工标高


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
        s=50,
    )

    plt.scatter(
        [p.x for p in gridNet.GridNetNegativePoints],
        [p.y for p in gridNet.GridNetNegativePoints],
        color="red",
        s=50,
    )

    plt.scatter(
        [p.x for p in gridNet.GridNetPositivePoints],
        [p.y for p in gridNet.GridNetPositivePoints],
        color="blue",
        s=50,
    )

    # 标注每个零点的坐标
    for p in gridNet.gridNetAllzero_points:
        plt.text(
            p.x,
            p.y,
            f"({p.x}, {p.y})",
        )

    # 标准每个角点的值
    for p in gridNet.GridNetVertexs:
        plt.text(
            p.x - 0.05 * GridNet().edge_length,
            p.y - 0.05 * GridNet().edge_length,
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

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Negative Point",
            markersize=10,
            markerfacecolor="red",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Positive Point",
            markersize=10,
            markerfacecolor="blue",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Zero Point",
            markersize=10,
            markerfacecolor="green",
        ),
    ]

    legend1 = ax.legend(
        handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.2)
    )

    ax.add_artist(legend1)

    legend2 = ax.legend(
        loc="center left", bbox_to_anchor=(1, 0.5)
    )  # 表示图例框的中心应位于图表框（axes bounding box）的右侧中心（1 指的是完全在右边界外，0.5 表示垂直居中）。

    fig.subplots_adjust(right=0.75)
    fig.gca().invert_yaxis()
    # 将 x 轴放到上面
    ax.xaxis.set_ticks_position("top")

    ax.xaxis.set_label_position("top")

    plt.show()
