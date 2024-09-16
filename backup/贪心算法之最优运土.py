# 问题描述
#
# 现在我有五个挖土区域，分别知道各自能提供的土方量和中心坐标。
# 然后，我还有六个填土区域，他们各自可容纳的土方量和中心坐标也知道。
# 现在需要从挖土区域运土方到填土区域，直到挖土区域的所有土方用尽或者填土区域可容纳土方量装满。
# 当各个挖土区域运向各个填土区域的运土量乘以距离的总和最小时，运土方案达到最优。
# 挖土区域到运土区域的距离是两个地方的中心坐标距离。
# 请设计算法计算出最优方案，得到各个挖土区域分别向各个运土区域的运土量和距离。


# 备注, 贪心算法不能得出全局最优, 但是能从局部最优得到一个相对较优的方案


from rich import print


def calculate_distance(point1, point2, ndigits: int = 2):
    # 计算两点之间的距离
    return round(
        ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5, ndigits=2
    )


def find_optimal_transport(digging_sites: dict, filling_sites: dict):
    # 使用副本防止对原字典修改
    digging_sites = digging_sites.copy()
    filling_sites = filling_sites.copy()
    # 初始化每个填土区域的剩余容量
    remaining_capacity = {
        site: capacity[-1] for site, capacity in filling_sites.items()
    }

    # 初始化结果字典
    transport_plan = {digging_site: {} for digging_site in digging_sites.keys()}

    # 计算每个挖土区域到每个填土区域的距离，并按距离排序
    distances = {}
    for digging_site, digging_info in digging_sites.items():
        distances[digging_site] = sorted(
            [
                (
                    calculate_distance(digging_info[0], filling_info[0]),
                    filling_site,
                    filling_info[1],
                )
                for filling_site, filling_info in filling_sites.items()
            ],
            key=lambda x: x[0],
        )

    print("距离字典: ")
    print(distances)

    # 贪心算法：依次处理每个挖土区域
    for digging_site, digging_info in digging_sites.items():

        # 初始化挖土区域剩余土方量
        remain_digg = digging_info[1]
        for distance, filling_site, capacity in distances[digging_site]:
            # 如果挖土区域的土方用尽或者填土区域的容量用完，则跳过该填土区域

            if remain_digg <= 0 or remaining_capacity[filling_site] <= 0:
                continue

            # 计算可运输的土方量 (> 0)
            transport_amount = max(
                min(remain_digg, remaining_capacity[filling_site]), 0
            )

            # 更新结果字典
            transport_plan[digging_site][filling_site] = (transport_amount, distance)

            # 更新挖土区域的剩余土方量和填土区域的剩余容量 (> 0)
            remain_digg -= max(transport_amount, 0)
            # (> 0)
            remain_digg = max(remain_digg, 0)
            digging_sites[digging_site] = (
                digging_info[0],
                max(remain_digg - transport_amount, 0),
            )
            remaining_capacity[filling_site] -= max(transport_amount, 0)

    return transport_plan


# 示例输入数据
digging_sites = {
    "Digging Site 1": ((9.26, 24.1), 115.28),
    "Digging Site 2": ((4.41, 75.1), 40.25),
    "Digging Site 3": ((40.09, 1.2), 0.01),
    "Digging Site 4": ((41.20, 42.6), 52.46),
    "Digging Site 5": ((49.20, 74.6), 24.46),
}

filling_sites = {
    "Filling Site 1": ((30, 10), 61.97),
    "Filling Site 2": ((30, 30), 23.77),
    "Filling Site 3": ((10, 60), 36.16),
    "Filling Site 4": ((30, 60), 38.78),
    "Filling Site 5": ((50, 60), 25.01),
    "Filling Site 6": ((50, 20), 52.4),
}


# 调用函数计算最优方案
optimal_transport_plan: dict[str, dict[str, tuple[float, float]]] = (
    find_optimal_transport(digging_sites, filling_sites)
)

print("方案字典:")
print(optimal_transport_plan)


# 验证结果部分


digg_all_original = sum([soil for point, soil in digging_sites.values()])

fill_all_original = sum([soil for point, soil in filling_sites.values()])

# 挖方总和和填方总和

print(f"初始挖方总和和填方总和 {digg_all_original}, {fill_all_original}")

digg_all_transport = sum(
    [
        sum([soil for soil, distance in v.values()])
        for k, v in optimal_transport_plan.items()
    ]
)
fill_all_got = digg_all_transport


diff_where: str = "缺土" if fill_all_got < fill_all_original else "弃土"

print(
    f"挖方总运输量和填方总接受量: {digg_all_transport}, {fill_all_got}, {diff_where}: {round(abs(max(digg_all_original- digg_all_transport, fill_all_original - digg_all_original)),2)}"
)


all_limited_digg = all(
    [
        soil  # 单个挖土区域原土方量
        >= sum(  # 运向各个填土区域的土方量总和
            [
                soil  # 运向单个填土区域的土方量
                for fill_site, (soil, distance) in optimal_transport_plan[
                    digg_site
                ].items()
            ]
        )
        for digg_site, (point, soil) in digging_sites.items()
    ]
)
print("运向各个填土区域的土方量情况:")
print(
    [
        sum(  # 运向各个填土区域的土方量总和
            [
                soil  # 运向单个填土区域的土方量
                for fill_site, (soil, distance) in optimal_transport_plan[
                    digg_site
                ].items()
            ]
        )
        for digg_site, (point, soil) in digging_sites.items()
    ]
)

all_limited_fill = all(
    [
        soil  # 单个填土区域原可容纳土方量
        >= sum(  # 运向各个填土区域的土方量总和
            [
                soil
                for d in optimal_transport_plan.values()
                for fill_sitex, (soil, distance) in d.items()
                if fill_sitex == fill_site
            ]
        )
        for fill_site, (point, soil) in filling_sites.items()
    ]
)

print("各个填土区域容纳土方量情况:")
print(
    [
        sum(  # 运向各个填土区域的土方量总和
            [
                soil
                for d in optimal_transport_plan.values()
                for fill_sitex, (soil, distance) in d.items()
                if fill_sitex == fill_site
            ]
        )
        for fill_site, (point, soil) in filling_sites.items()
    ]
)
assert fill_all_got == fill_all_original or digg_all_transport == digg_all_original
assert all_limited_digg and all_limited_fill
print("结果验证完成")
