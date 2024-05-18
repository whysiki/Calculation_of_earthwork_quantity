from point import *

# from shapely.geometry import Polygon

# from matplotlib.patches import Polygon

# Calculate the area of each grid cell divided by zero points
# areas = np.zeros_like(data)
# for i in range(reshaped_coordinates.shape[0] - 1):
#     for j in range(reshaped_coordinates.shape[1] - 1):
#         # Get the corners of the grid cell
#         corners = [
#             reshaped_coordinates[i, j],
#             reshaped_coordinates[i, j + 1],
#             reshaped_coordinates[i + 1, j + 1],
#             reshaped_coordinates[i + 1, j],
#         ]

#         # Get the zero points in the grid cell
#         cell_zero_points = zero_points[
#             (zero_points[:, 0] >= corners[0][0])
#             & (zero_points[:, 0] <= corners[2][0])
#             & (zero_points[:, 1] >= corners[0][1])
#             & (zero_points[:, 1] <= corners[2][1])
#         ]

#         # If there are two or more zero points, calculate the area
#         if len(cell_zero_points) >= 2:
#             # Sort the zero points by their angle relative to the center of the grid cell
#             center = np.mean(corners, axis=0)
#             angles = np.arctan2(
#                 cell_zero_points[:, 1] - center[1], cell_zero_points[:, 0] - center[0]
#             )
#             cell_zero_points = cell_zero_points[np.argsort(angles)]

#             # Create a polygon and calculate its area
#             polygon = Polygon(np.concatenate([corners, cell_zero_points]))
#             areas[i, j] = polygon.area

# print(areas)
