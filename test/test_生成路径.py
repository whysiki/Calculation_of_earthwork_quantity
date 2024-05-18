# # def is_valid(grid, x, y):
# #     return 0 <= x < len(grid) and 0 <= y < len(grid[0])


# # def dfs(grid, x, y, visited, path, zero_paths):
# #     visited[x][y] = True
# #     path.append((x, y))
# #     directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
# #     for dx, dy in directions:
# #         nx, ny = x + dx, y + dy
# #         if is_valid(grid, nx, ny) and not visited[nx][ny]:
# #             if grid[nx][ny] == 0:
# #                 zero_paths.append(list(path))
# #             else:
# #                 dfs(grid, nx, ny, visited, path, zero_paths)
# #     path.pop()
# #     visited[x][y] = False


# # def find_zero_paths(grid):
# #     zero_paths = []
# #     visited = [[False for _ in range(len(grid[0]))] for _ in range(len(grid))]
# #     for i in range(len(grid)):
# #         for j in range(len(grid[0])):
# #             if grid[i][j] == 0 and not visited[i][j]:
# #                 dfs(grid, i, j, visited, [], zero_paths)
# #     return zero_paths


# # # 示例用法
# # grid = [[1, 0, -1], [-1, 0, 1], [1, 1, -1]]
# # zero_paths = find_zero_paths(grid)
# # for path in zero_paths:
# #     print(path)
# def find_zero_paths(grid):
#     zero_paths = []

#     def dfs(x, y, path, positives, negatives):
#         if grid[x][y] == 0:
#             if len(positives) == len(negatives):
#                 zero_paths.append(path)
#             return

#         path.append((x, y))
#         if grid[x][y] > 0:
#             positives.add((x, y))
#         else:
#             negatives.add((x, y))

#         directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
#         for dx, dy in directions:
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and (nx, ny) not in path:
#                 dfs(nx, ny, path[:], set(positives), set(negatives))

#     for i in range(len(grid)):
#         for j in range(len(grid[0])):
#             if grid[i][j] == 0:
#                 dfs(i, j, [], set(), set())

#     return zero_paths


# # 示例用法
# grid = [[1, -1, 0], [0, 0, 0], [-1, 1, 0]]
# print(find_zero_paths(grid))
def is_valid(x, y, grid):
    n = len(grid)
    m = len(grid[0])
    return 0 <= x < n and 0 <= y < m


def dfs(x, y, grid, visited, path):
    visited[x][y] = True
    path.append((x, y))

    # 定义四个方向的移动
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if is_valid(nx, ny, grid) and not visited[nx][ny]:
            dfs(nx, ny, grid, visited, path)


def find_zero_paths(grid):
    n = len(grid)
    m = len(grid[0])
    zero_paths = []

    for i in range(n):
        for j in range(m):
            if grid[i][j] == 0:
                visited = [[False] * m for _ in range(n)]
                path = []
                dfs(i, j, grid, visited, path)
                zero_paths.append(path)

    return zero_paths


import matplotlib.pyplot as plt


def plot_zero_paths(grid, zero_paths):
    fig, ax = plt.subplots()

    # 绘制网格
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                ax.plot(j, i, "bo", markersize=10)  # 蓝色正点
            elif grid[i][j] == -1:
                ax.plot(j, i, "ro", markersize=10)  # 红色负点

    # 绘制零点路径
    for path in zero_paths:
        if len(path) > 1:
            xs, ys = zip(*path)
            ax.plot(ys, xs, "k-", linewidth=2)  # 黑色路径线

    ax.grid(True)
    ax.set_aspect("equal")
    plt.show()


# 示例
grid = [[1, -1, 1, -1, 1], [-1, 0, 1, -1, 1], [1, -1, 1, 0, -1], [1, -1, 0, -1, 0]]

zero_paths = find_zero_paths(grid)
for path in zero_paths:
    print(path)
plot_zero_paths(grid, zero_paths)
