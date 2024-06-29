import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from gridClass import GridNet
from adjustText import adjust_text
from rich import print
import pandas as pd

# 图表大小

chart_size = (20, 12)


# 全局设置字体大小
plt.rcParams.update(
    {
        "font.size": 8,  # 全局字体大小
        "axes.titlesize": 12,  # 图表标题字体大小
        "axes.labelsize": 8,  # 坐标轴标签字体大小
        "xtick.labelsize": 8,  # x 轴刻度字体大小
        "ytick.labelsize": 8,  # y 轴刻度字体大小
        "legend.fontsize": 8,  # 图例字体大小
        "figure.titlesize": 12,  # 图表整体标题字体大小
    }
)


def showExcavationAndFillingZone(
    gridNet: GridNet, savePath: str = None, dpi: int = 300
):

    fig, ax = plt.subplots()

    # 大小全屏
    fig.set_size_inches(*chart_size)

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
        s=40,
    )
    # 画正负点分布
    ax.scatter(
        [p.x for p in gridNet.GridNetNegativePoints],
        [p.y for p in gridNet.GridNetNegativePoints],
        color="red",
        s=20,
    )

    ax.scatter(
        [p.x for p in gridNet.GridNetPositivePoints],
        [p.y for p in gridNet.GridNetPositivePoints],
        color="blue",
        s=20,
    )

    # 标注每个零点的坐标
    texts = []
    for p in gridNet.gridNetAllzero_points:
        texts.append(
            ax.text(
                p.x + grid.edge_length / 10,
                p.y + grid.edge_length / 10,
                f"({p.x}, {p.y})",
                size=8,
                # color="green",
            )
        )

    # 标准每个角点的值
    for p in gridNet.GridNetVertexs:
        texts.append(
            ax.text(
                p.x + grid.edge_length / 10,
                p.y + grid.edge_length / 10,
                f"{p.value}",
                color="red" if p.value < 0 else "blue",
                weight="bold",
            )
        )

    # 处理标注重叠问题
    adjust_text(texts)

    # 填充挖方区域和填方区域

    for index, path in enumerate(gridNet.gridNetNegativeZone):

        xs, ys = zip(*[(p.x, p.y) for p in path])

        ax.fill(xs, ys, color="red", alpha=0.5, label=f"negative zone {index+1}")

    for index, path in enumerate(gridNet.gridNetPositiveZone):

        xs, ys = zip(*[(p.x, p.y) for p in path])

        ax.fill(xs, ys, color="blue", alpha=0.5, label=f"positive zone {index+1}")

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

    # fig.tight_layout()

    ax.xaxis.set_ticks_position("top")

    ax.xaxis.set_label_position("top")

    ax.set_title("Design elevation minus original site elevation")

    if savePath and isinstance(savePath, str):

        fig.savefig(savePath, dpi=dpi)


def showNegativeAndPositivePath(gridNet: GridNet, savePath: str = None, dpi: int = 300):

    fig12, ax12 = plt.subplots(nrows=1, ncols=2, figsize=chart_size)

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

    gridNetNegativeZone_c_v, gridNetPositiveZone_c_v = (
        gridNet.calculateNegtiveZonesAndPositiveZones_geoCenterpoint_and_v()
    )

    for path in gridNet.gridNetNegativeZone:

        # print(path)
        geoCenterPoint = gridNetNegativeZone_c_v[path][0]
        v = gridNetNegativeZone_c_v[path][1]

        ax12[0].scatter(geoCenterPoint.x, geoCenterPoint.y, color="red", s=50)
        ax12[0].text(
            geoCenterPoint.x,
            geoCenterPoint.y + GridNet().edge_length / 6,
            f"({round(geoCenterPoint.x,2)} , {round(geoCenterPoint.y,2)})",
            color="red",
            size=8,
        )

        # ax12[0].text(
        #     geoCenterPoint.x,
        #     geoCenterPoint.y + GridNet().edge_length / 2,
        #     f"-{round(v,2)}",
        #     color="red",
        #     weight="bold",
        #     size=8,
        # )

        xs, ys = zip(*[(p.x, p.y) for p in path])

        ax12[0].plot(xs, ys, color="red")

        # 设置图表标题
        ax12[0].set_title("Negative Path")

    for path in gridNet.gridNetPositiveZone:

        # print(path)
        geoCenterPoint = gridNetPositiveZone_c_v[path][0]
        v = gridNetPositiveZone_c_v[path][1]

        ax12[1].scatter(geoCenterPoint.x, geoCenterPoint.y, color="blue", s=50)
        ax12[1].text(
            geoCenterPoint.x,
            geoCenterPoint.y + GridNet().edge_length / 6,
            f"({round(geoCenterPoint.x,2)} , {round(geoCenterPoint.y,2)})",
            color="blue",
            # weight="bold",
            size=8,
        )
        # ax12[1].text(
        #     geoCenterPoint.x,
        #     geoCenterPoint.y + GridNet().edge_length / 2,
        #     f"+{round(v,2)}",
        #     color="blue",
        #     weight="bold",
        #     size=8,
        # )

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

    # fig12.tight_layout()

    fig12.add_artist(legend12)

    if savePath and isinstance(savePath, str):

        fig12.savefig(savePath, dpi=dpi)


def showData(gridNet: GridNet, savePath: str = None, dpi: int = 300):

    # fig = plt.figure(figsize=chart_size)
    # gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # 定义两列，第一列比第二列宽
    # ax = fig.add_subplot(gs[0])
    fig, ax = plt.subplots()

    # 图大小全屏
    #
    # fig.set_size_inches(10, 6)

    # 大小全屏
    fig.set_size_inches(*chart_size)

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
                grid.center_coordinate[1] + GridNet().edge_length / 7,
                textexcavation,
                color="red",
                ha="center",
                va="center",
                weight="bold",
            )
            ax.text(
                grid.center_coordinate[0],
                grid.center_coordinate[1] + GridNet().edge_length / 3,
                textfilling,
                color="blue",
                ha="center",
                va="center",
                weight="bold",
            )
            cell_text.append([textindex, textexcavation, textfilling])

    cell_text.append(["Total", f"-{gridNet.excavationAll}", f"+{gridNet.fillingAll}"])

    cell_text.insert(0, ["Index", "Excavation", "Filling"])

    print(cell_text)

    # 导出为表格

    pd.DataFrame(cell_text).to_excel("DataTable.xlsx", index=False)

    # ax2 = fig.add_subplot(gs[1])
    # ax2.axis("off")
    # table = ax2.table(
    #     cellText=cell_text,
    #     colLabels=cell_text.pop(0),
    #     cellLoc="center",
    #     loc="center",
    # )
    # table.auto_set_font_size(False)
    # table.set_fontsize(6)
    # table.scale(1, 2)
    ax.set_title("Grid Excavation and Filling Data Table")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="excavation quantity",
            markersize=10,
            markerfacecolor="red",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="filling quantity",
            markersize=10,
            markerfacecolor="blue",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="grid index",
            markersize=10,
            markerfacecolor="black",
        ),
    ]

    legend = ax.legend(
        handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5)
    )

    # fig.tight_layout()

    if savePath and isinstance(savePath, str):

        fig.savefig(savePath, dpi=dpi)
