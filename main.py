import matplotlib.pyplot as plt
from gridClass import GridNet
from dataProcess import 施工标高
from visualizeData import (
    showNegativeAndPositivePath,
    showData,
    showExcavationAndFillingZone,
)

if __name__ == "__main__":

    # 每个都是方格网的一个点, 坐标系： 左上角是原点 从左到右为y,从上到下为x轴
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

    showExcavationAndFillingZone(
        gridNet, savePath="ExcavationAndFillingZone.png", dpi=150
    )

    showNegativeAndPositivePath(
        gridNet, savePath="NegativeAndPositivePath.png", dpi=150
    )

    showData(gridNet, savePath="DataTable.png", dpi=150)

    plt.show()
