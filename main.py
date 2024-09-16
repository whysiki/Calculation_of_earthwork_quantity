import matplotlib.pyplot as plt
from gridClass import GridNet
from dataProcess import 施工标高
from visualizeData import (
    showNegativeAndPositivePath,
    showData,
    showExcavationAndFillingZone,
)
from rich import print
import os

if __name__ == "__main__":

    # 施工标高数据 = np.array(设计标高) - np.array(原场地标高)

    data = 施工标高.tolist()

    # print(data)

    # 保留两位小数
    data = [[round(value, 2) for value in row] for row in data]

    print(data)

    # 构造网格网络
    gridNet = GridNet(
        gridNetValueMatrix=data,
    )

    # 构造插入零点的网格网络
    gridNet.construct_grids_and_insert_zero_points()

    # 计算每个网格的负值和正值区域

    gridNet.caculate_for_each_grid_negtive_positive_zone()

    # 计算整个网格网络的负值和正值区域

    gridNet.caculate_gridNetNegativeZone_gridNetPositiveZone()

    # 计算每个网格的挖方和填方量和整个网格网络的挖方和填方量

    gridNet.calculateAmountExcavationAndFilling_for_all_girds()

    if not os.path.exists("data"):
        os.mkdir("data")

    showExcavationAndFillingZone(
        gridNet, savePath="data/ExcavationAndFillingZone.png", dpi=150
    )

    showNegativeAndPositivePath(
        gridNet, savePath="data/NegativeAndPositivePath.png", dpi=150
    )

    showData(gridNet, savePath="data/DataTable.png", dpi=150)

    plt.show()
