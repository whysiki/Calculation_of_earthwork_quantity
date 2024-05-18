from shapely.geometry import Polygon

# Define the points of the polygon
points = [(0.0, 0.0), (0.0, 20.0), (20.0, 20.0), (20.0, 17.23), (8.8, 0.0), (0.0, 0.0)]

# Create a polygon from these points
polygon = Polygon(points)

# Calculate the area of the polygon
area = polygon.area
print(area)
