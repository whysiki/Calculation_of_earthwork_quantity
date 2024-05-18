from rich import print

v11 = (1 / 6) * 1.54 * 3.33 * 0.01
v12 = (400 - 0.5 * 1.54 * 3.33) * (1 / 5) * (0.12 + 0.21 + 0.05)

print(v11, v12)


v21 = (1 / 6) * 14.78 * 16.59 * 0.34
v22 = (400 - 0.5 * 14.78 * 16.59) * (1 / 5) * (0.12 + 0.21 + 0.07)

print(v21, v22)

v31 = (1 / 6) * 17.44 * 16.59 * 0.34
v32 = (400 - 0.5 * 17.44 * 16.59) * (1 / 5) * (0.19 + 0.07 + 0.05)

print(v31, v32)

v41 = 10 * (13.75 + 10.95) * 0.25 * (0.11 + 0.23)
v42 = 0.125 * 20 * (6.25 + 9.05) * (0.05 + 0.19)

print(v41, v42)

v51 = (1 / 6) * 0.35 * 1.54 * 0.01 + (1 / 6) * 8.57 * 2.77 * 0.09
v52 = (400 - 0.5 * 0.35 * 1.54 - 0.5 * 8.57 * 2.77) * (1 / 6) * (0.56 + 0.12)

print(" ------------")
print(v51, v52)


v61 = (1 / 6) * (5.81 * 8.57 * 0.09) + (1 / 6) * (14.78 * 12.14 * 0.34)
v62 = (400 - 0.5 * (5.81 * 8.57 + 14.78 * 12.14)) * (1 / 6) * (0.09 + 0.34)

print(v61, v62)

v71 = (1 / 6) * 12.14 * 17.44 * 0.34
v72 = (400 - 0.5 * 12.14 * 17.44) * (1 / 5) * (0.22 + 0.14 + 0.05)

print(v71, v72)


v81 = (1 / 6) * 13.75 * 13.75 * 0.11
v82 = (400 - 0.5 * 13.75 * 13.75) * (1 / 5) * (0.05 + 0.05 + 0.14)

print(v81, v82)

print(" ------------")
v91 = (400 - 0.5 * 11.2 * 17.23) * 0.2 * (0.44 + 0.18 + 0.09)
v92 = (1 / 6) * 11.2 * 17.23 * 0.56

print(v91, v92)


v101 = (400 - 0.5 * 14.19 * 6.38) * 0.2 * (0.47 + 0.18 + 0.09)
v102 = (1 / 6) * 14.19 * 6.38 * 0.22


print(v101, v102)
v111 = (1 / 6) * 13.62 * 17.41 * 0.47

v112 = (400 - 0.5 * 13.62 * 17.41) * 0.2 * (0.14 + 0.07 + 0.22)


print(v111, v112)
v121 = (1 / 6) * 18.21 * 18.68 * 0.71
v122 = (400 - 0.5 * 18.21 * 18.68) * 0.2 * (0.07 + 0.14 + 0.05)
print(v121, v122)


print(" ------------")

subma = sum([v11, v21, v31, v41, v51, v61, v71, v81, v91, v101, v111, v121])
add = sum([v12, v22, v32, v42, v52, v62, v72, v82, v92, v102, v112, v122])
print(subma, add)

diff = subma - add
# print(diff)

print("挖方和填方的差值为: ", diff)
print(diff / subma * 100, diff / add * 100)

print((subma + add) * 0.5 * 0.01)

## 挖方区块
ss1 = (
    (1 / 6) * 8.57 * 2.77 * 0.09 + (1 / 6) * (5.81 * 8.57 * 0.09) + v91 + v101 + v111
)  # (1 / 6) * 0.35 * 1.54 * 0.01 + (1 / 6) * 8.57 * 2.77 * 0.09
# v61 = (1 / 6) * (5.81 * 8.57 * 0.09) + (1 / 6) * (14.78 * 12.14 * 0.34)
ss2 = v121
ss3 = v11 + (1 / 6) * 0.35 * 1.54 * 0.01
ss4 = v21 + v31 + (1 / 6) * (14.78 * 12.14 * 0.34) + v71
ss5 = v41 + v81

print("挖方区块")
print(ss1, ss2, ss3, ss4, ss5)
print("挖方区块总和")
print(sum([ss1, ss2, ss3, ss4, ss5]))


# 填方区块
a1 = v52 + v92
a2 = v62 + v102
a3 = v112 + v122
a4 = v72 + v82
a5 = v32 + v42
a6 = v12 + v22
a1 = round(a1, 2)
a2 = round(a2, 2)
a3 = round(a3, 2)
a4 = round(a4, 2)
a5 = round(a5, 2)
a6 = round(a6, 2)
print("填方区块")
print(a1, a2, a3, a4, a5, a6)
print("填方区块总和")
print(sum([a1, a2, a3, a4, a5, a6]))
