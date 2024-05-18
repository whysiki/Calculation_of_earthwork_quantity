import numpy as np
import matplotlib.pyplot as plt
from rich import print


def find_zero_points(grid):
    net_length = 20
    zero_points = []
    rows, cols = len(grid), len(grid[0])

    # Check horizontal edges
    for i in range(rows):
        for j in range(cols - 1):
            if (grid[i][j] > 0 and grid[i][j + 1] < 0) or (
                grid[i][j] < 0 and grid[i][j + 1] > 0
            ):
                # zero_points.append((20 * j + 10, 20 * i))
                point = (
                    net_length * i,
                    net_length * j
                    + net_length
                    * (abs(grid[i][j]) / (abs(grid[i][j]) + abs(grid[i][j + 1]))),
                )
                point = round(point[0], 2), round(point[1], 2)
                zero_points.append(point)

    # Check vertical edges
    for j in range(cols):
        for i in range(rows - 1):
            if (grid[i][j] > 0 and grid[i + 1][j] < 0) or (
                grid[i][j] < 0 and grid[i + 1][j] > 0
            ):
                # zero_points.append((20 * j, 20 * i + 10))
                point = (
                    net_length * i
                    + net_length
                    * (abs(grid[i][j]) / (abs(grid[i][j]) + abs(grid[i + 1][j]))),
                    net_length * j,
                )
                point = round(point[0], 2), round(point[1], 2)
                zero_points.append(point)

    return zero_points


# 测试方格网数据
# 每个都是方格网的一个点, 左上角是原点
data = np.array(
    [
        [-0.44, -0.18, -0.47, 0.07, -0.71],
        [0.56, -0.09, 0.22, 0.14, 0.05],
        [-0.01, 0.12, -0.34, 0.05, -0.11],
        [0.05, 0.21, 0.07, 0.19, -0.23],
    ]
).tolist()

# 画出方格网的所有角点及在点的左上角标注这个点的值


# 找到零点
zero_points = find_zero_points(data)

print(zero_points, len(zero_points))

# # 画出零点

# # Plot the grid points
# plt.figure(figsize=(10, 10))


# # Grid edge length
# edge_length = 20

# # Calculate coordinates
# coordinates = (
#     np.array(np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))).T.reshape(
#         -1, 2
#     )
#     * edge_length
# )

# # # Flatten the data array
# flattened_data = data.flatten()


# # Reshape coordinates and flattened_data back to 2D
# reshaped_coordinates = coordinates.reshape(data.shape[0], data.shape[1], 2)
# reshaped_data = flattened_data.reshape(data.shape)

# for i in range(reshaped_coordinates.shape[0]):
#     for j in range(reshaped_coordinates.shape[1]):
#         coord = reshaped_coordinates[i, j]
#         value = reshaped_data[i, j]
#         plt.scatter(coord[0], coord[1], color="blue" if value >= 0 else "red")
#         plt.text(coord[0], coord[1], str(value), fontsize=12)

#         # Connect points horizontally
#         if j != reshaped_coordinates.shape[1] - 1:
#             next_coord = reshaped_coordinates[i, j + 1]
#             plt.plot(
#                 [coord[0], next_coord[0]], [coord[1], next_coord[1]], color="black"
#             )

#         # Connect points vertically
#         if i != reshaped_coordinates.shape[0] - 1:
#             next_coord = reshaped_coordinates[i + 1, j]
#             plt.plot(
#                 [coord[0], next_coord[0]], [coord[1], next_coord[1]], color="black"
#             )


# # 画出零点, 并设置大小和颜色
# plt.scatter(zero_points[:, 0], zero_points[:, 1], color="green", s=100, marker="x")


# plt.show()

# 把每个方格内的所有零点互相连接起来, 并计算方格被零点分割的各个面积
# # Function to calculate area given three points
# def calculate_area(point1, point2, point3):
#     return 0.5 * abs(
#         (point1[0] - point3[0]) * (point2[1] - point1[1])
#         - (point1[0] - point2[0]) * (point3[1] - point1[1])
#     )


# # Loop through each grid cell
# for i in range(data.shape[0] - 1):
#     for j in range(data.shape[1] - 1):
#         cell_points = [
#             (i * edge_length, j * edge_length),
#             ((i + 1) * edge_length, j * edge_length),
#             ((i + 1) * edge_length, (j + 1) * edge_length),
#             (i * edge_length, (j + 1) * edge_length),
#         ]
#         cell_area = 0

#         # Loop through zero points
#         for zero_point in zero_points:
#             # Check if zero point lies within the current cell
#             if (
#                 cell_points[0][0] <= zero_point[0] <= cell_points[1][0]
#                 or cell_points[1][0] <= zero_point[0] <= cell_points[2][0]
#             ) and (
#                 cell_points[0][1] <= zero_point[1] <= cell_points[3][1]
#                 or cell_points[3][1] <= zero_point[1] <= cell_points[2][1]
#             ):

#                 # Calculate area formed by the zero point and two vertices of the cell
#                 cell_area += calculate_area(cell_points[0], cell_points[1], zero_point)

#         print(f"Area of cell ({i}, {j}): {cell_area:.2f}")
# Connect zero points within each grid cell and calculate the areas of the grid cells divided by zero points


# def connect_zero_points(zero_points, edge_length):
#     # Dictionary to store segments and their intersection points
#     segment_points = {}

#     # Connect zero points within each grid cell
#     for i in range(len(zero_points)):
#         for j in range(i + 1, len(zero_points)):
#             p1 = tuple(zero_points[i])  # Convert to tuple
#             p2 = tuple(zero_points[j])  # Convert to tuple

#             # Calculate grid cell indices for both points
#             grid_indices_p1 = (int(p1[0] / edge_length), int(p1[1] / edge_length))
#             grid_indices_p2 = (int(p2[0] / edge_length), int(p2[1] / edge_length))

#             if grid_indices_p1 != grid_indices_p2:
#                 # Ensure p1 is to the left of p2
#                 if grid_indices_p1[1] > grid_indices_p2[1]:
#                     p1, p2 = p2, p1

#                 # Add the segment and its intersection point with grid cell boundary
#                 segment = (p1, p2)
#                 intersection_point = (
#                     grid_indices_p1[0] * edge_length,
#                     grid_indices_p1[1] * edge_length,
#                 )
#                 if segment in segment_points:
#                     segment_points[segment].append(intersection_point)
#                 else:
#                     segment_points[segment] = [intersection_point]

#     # Calculate areas of grid cells divided by zero points
#     grid_areas = {}
#     # Plot grid cells and annotate areas divided by zero points

#     for segment, intersection_points in segment_points.items():
#         # Sort intersection points
#         intersection_points.sort(key=lambda x: x[1])

#         # Initialize area with the area of the triangle formed by the first and last points
#         area = 0.5 * abs(
#             (segment[0][0] - segment[1][0])
#             * (intersection_points[0][1] - segment[1][1])
#         )
#         area += 0.5 * abs(
#             (intersection_points[-1][0] - segment[0][0])
#             * (segment[1][1] - intersection_points[-1][1])
#         )

#         # Add area of trapezoids formed by the remaining points
#         for i in range(len(intersection_points) - 1):
#             area += 0.5 * abs(
#                 (intersection_points[i][0] - intersection_points[i + 1][0])
#                 * (intersection_points[i][1] - intersection_points[i + 1][1])
#             )

#         grid_areas[segment] = area

#     return grid_areas


# # Connect zero points and calculate areas
# grid_areas = connect_zero_points(zero_points, edge_length)


# print(grid_areas)


# for coord, value in zip(coordinates, flattened_data):
#     plt.scatter(coord[0], coord[1], color="blue" if value >= 0 else "red")
#     plt.text(coord[0], coord[1], str(value), fontsize=12)
# Rotate the plot clockwise by 90 degrees
# plt.gca().invert_xaxis()
# plt.gca().invert_yaxis()
# plt.grid(True)
# Plot the zero points
# plt.scatter(zero_points[:, 0], zero_points[:, 1], color="red")
# Rotate the plot 90 degrees clockwise
# Rotate the plot clockwise by 90 degrees
# plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
# ax.invert_xaxis()
# Plot the grid points with their values
# fig, ax = plt.subplots(figsize=(10, 10))
# for coord, value in zip(coordinates, flattened_data):
#     ax.scatter(coord[0], coord[1], color="blue" if value >= 0 else "red")
#     ax.text(coord[0], coord[1], str(value), fontsize=12)

# Rotate the plot 90 degrees clockwise
# ax.invert_xaxis()
# ax.invert_xaxis()
# ax.invert_xaxis()
0

# [[ 0.   57.41]
#  [ 0.   61.79]
#  [20.   17.23]
#  [20.   25.81]
#  [40.    1.54]
#  [40.   25.22]
#  [40.   57.44]
#  [40.   66.25]
#  [60.   69.05]
#  [ 8.8   0.  ]
#  [39.65  0.  ]
#  [43.33  0.  ]
#  [28.57 20.  ]
#  [13.62 40.  ]
#  [27.86 40.  ]
#  [56.59 40.  ]
#  [18.68 80.  ]
#  [26.25 80.  ]]

# 18
