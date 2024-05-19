import numpy as np
import matplotlib.pyplot as plt
from rich import print
from typing import Iterable, List, Union
import math
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from gridClass import *
from adjustText import adjust_text


def showExcavationAndFillingZone(
    gridNet: GridNet, savePath: str = None, dpi: int = 300
):

    fig, ax = plt.subplots()

    # fig.set_size_inches(8, 8)

    # 画网格线
    gridNet.drawGridNetLines(gridNet.verticalPoints, gridNet.horizontalPoints, ax)

    # 标注每个网格的索引
    for rs in gridNet.grids:

        for grid in rs:

            plt.text(
                grid.center_coordinate[0],
                grid.center_coordinate[1],
                f"{grid.index+1}",
                color="black",
                weight="bold",
            )

    # 画零点分布
    ax.scatter(
        [p.x for p in gridNet.gridNetAllzero_points],
        [p.y for p in gridNet.gridNetAllzero_points],
        color="green",
        s=100,
    )
    # 画正负点分布
    ax.scatter(
        [p.x for p in gridNet.GridNetNegativePoints],
        [p.y for p in gridNet.GridNetNegativePoints],
        color="red",
        s=50,
    )

    ax.scatter(
        [p.x for p in gridNet.GridNetPositivePoints],
        [p.y for p in gridNet.GridNetPositivePoints],
        color="blue",
        s=50,
    )

    # 标注每个零点的坐标
    texts = []
    for p in gridNet.gridNetAllzero_points:
        texts.append(
            ax.text(
                p.x,
                p.y,
                f"({p.x}, {p.y})",
                size=8,
                # color="green",
            )
        )

    # 标准每个角点的值
    for p in gridNet.GridNetVertexs:
        texts.append(
            ax.text(
                p.x,
                p.y,
                f"{p.value}",
                color="red" if p.value < 0 else "blue",
                weight="bold",
            )
        )

    # 处理标注重叠问题
    adjust_text(texts)

    # 填充挖方区域和填方区域

    for path in gridNet.gridNetNegativeZone:

        xs, ys = zip(*[(p.x, p.y) for p in path])

        ax.fill(xs, ys, color="red", alpha=0.5, label="negative zone")

    for path in gridNet.gridNetPositiveZone:

        xs, ys = zip(*[(p.x, p.y) for p in path])

        ax.fill(xs, ys, color="blue", alpha=0.5, label="positive zone")

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

    legend2 = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    ax.add_artist(legend2)

    fig.subplots_adjust(right=0.75)
    fig.gca().invert_yaxis()
    ax.xaxis.set_ticks_position("top")

    ax.xaxis.set_label_position("top")

    ax.set_title("Design elevation minus original site elevation")

    if savePath and isinstance(savePath, str):

        fig.savefig(savePath, dpi=dpi)


def showNegativeAndPositivePath(gridNet: GridNet, savePath: str = None, dpi: int = 300):

    fig12, ax12 = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    gridNet.drawGridNetLines(gridNet.verticalPoints, gridNet.horizontalPoints, ax12[0])
    gridNet.drawGridNetLines(gridNet.verticalPoints, gridNet.horizontalPoints, ax12[1])

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

        ax12[0].plot(xs, ys, color="red")

        # 设置图表标题
        ax12[0].set_title("Negative Path")

    for path in gridNet.gridNetPositiveZone:

        # print(path)

        xs, ys = zip(*[(p.x, p.y) for p in path])

        ax12[1].plot(xs, ys, color="blue")

        # 设置图表标题

        ax12[1].set_title("Positive Path")

    for i in range(2):
        ax12[i].xaxis.set_ticks_position("top")
        ax12[i].xaxis.set_label_position("top")
        ax12[i].invert_yaxis()

    legend_elements12 = [
        Line2D(
            [0],
            [0],
            marker="_",
            color="red",
            label="Negative Path",
            markersize=10,
            markerfacecolor="red",
        ),
        Line2D(
            [0],
            [0],
            marker="_",
            color="blue",
            label="Positive Path",
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

    fig12.subplots_adjust(right=0.9)

    legend12 = ax12[-1].legend(
        handles=legend_elements12, loc="center left", bbox_to_anchor=(1, 0.5)
    )

    fig12.tight_layout()

    fig12.add_artist(legend12)

    if savePath and isinstance(savePath, str):

        fig12.savefig(savePath, dpi=dpi)


def showData(gridNet: GridNet, savePath: str = None, dpi: int = 300):

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # 定义两列，第一列比第二列宽
    ax = fig.add_subplot(gs[0])

    gridNet.drawGridNetLines(gridNet.verticalPoints, gridNet.horizontalPoints, ax)
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

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
            cell_text.append([textindex, textexcavation, textfilling])

    cell_text.append(["Total", f"-{gridNet.excavationAll}", f"+{gridNet.fillingAll}"])

    cell_text.insert(0, ["Index", "Excavation", "Filling"])

    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    table = ax2.table(
        cellText=cell_text,
        colLabels=cell_text.pop(0),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.set_title("Grid Excavation and Filling Data Table")
    fig.tight_layout()

    if savePath and isinstance(savePath, str):

        fig.savefig(savePath, dpi=dpi)
