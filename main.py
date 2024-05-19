import matplotlib.pyplot as plt
from gridClass import GridNet
from dataProcess import 施工标高
from visualizeData import (
    showNegativeAndPositivePath,
    showData,
    showExcavationAndFillingZone,
)

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

    negtive_poins, positive_points, vertexs = (
        GridNet.findAllNegtive_and_Positive_and_Vertexs_Points(
            gridNetValueMatrix=data, net_length=GridNet().edge_length
        )
    )

    zero_points = GridNet.find_zero_points(
        gridNetValueMatrix=data, net_length=GridNet().edge_length
    )

    # 列视角点分布，行视角点分布
    verticalPoints, horizontalPoints = GridNet.getverticalPoints_and_horizontalPoints(
        vertexs=vertexs
    )

    # 构造网格网络
    gridNet = GridNet(
        verticalPoints=verticalPoints,
        horizontalPoints=horizontalPoints,
        zeropoints=zero_points.copy(),
        positivePoints=positive_points,
        negativePoints=negtive_poins,
        vertexs=vertexs,
    )

    gridNet.construct_grids_and_insert_zero_points()
    
    gridNet.caculate_for_each_grid_negtive_positive_zone()
    
    gridNet.caculate_gridNetNegativeZone_gridNetPositiveZone()

    gridNet.calculateAmountExcavationAndFilling_for_all_girds()

    showExcavationAndFillingZone(
        gridNet, savePath="ExcavationAndFillingZone.png", dpi=150
    )

    showNegativeAndPositivePath(
        gridNet, savePath="NegativeAndPositivePath.png", dpi=150
    )

    showData(gridNet, savePath="DataTable.png", dpi=150)

    plt.show()
