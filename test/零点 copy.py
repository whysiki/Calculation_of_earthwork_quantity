import numpy as np
import matplotlib.pyplot as plt
from rich import print

# 测试方格网数据
# 每个都是方格网的一个点, 左上角是原点
data = np.array(
    [
        [-0.44, -0.18, -0.47, 0.07, -0.71],
        [0.56, -0.09, 0.22, 0.14, 0.05],
        [-0.01, 0.12, -0.34, 0.05, -0.11],
        [0.05, 0.21, 0.07, 0.19, -0.23],
    ]
)

# Calculate coordinates
coordinates = (
    np.array(np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))).T.reshape(
        -1, 2
    )
    * 20
)

zero_points = np.array(
    [
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
)

# Flatten the data array
flattened_data = data.flatten()



# Reshape coordinates and flattened_data back to 2D
reshaped_coordinates = coordinates.reshape(data.shape[0], data.shape[1], 2)
reshaped_data = flattened_data.reshape(data.shape)

plt.figure(figsize=(8,8))

for i in range(reshaped_coordinates.shape[0]):
    for j in range(reshaped_coordinates.shape[1]):
        coord = reshaped_coordinates[i, j]
        value = reshaped_data[i, j]
        plt.scatter(coord[0], coord[1], color="blue" if value >= 0 else "red")
        plt.text(coord[0], coord[1], str(value), fontsize=12)

        # Connect points horizontally
        if j != reshaped_coordinates.shape[1] - 1:
            next_coord = reshaped_coordinates[i, j + 1]
            plt.plot(
                [coord[0], next_coord[0]], [coord[1], next_coord[1]], color="black"
            )

        # Connect points vertically
        if i != reshaped_coordinates.shape[0] - 1:
            next_coord = reshaped_coordinates[i + 1, j]
            plt.plot(
                [coord[0], next_coord[0]], [coord[1], next_coord[1]], color="black"
            )


# 画出零点, 并设置大小和颜色
plt.scatter(zero_points[:, 0], zero_points[:, 1], color="green", s=100, marker="x")


plt.show()
