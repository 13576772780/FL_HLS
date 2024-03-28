#库导入
import matplotlib.pyplot as plt

# 准备数据
x = ["p=0", "p=0.4", "p=0.6", "p=0.8"]

# fedavg_medal = [60.85,	64.16,	68.92,	71.71,	76.08]
# fedrep_medal = [76.17,	79.62,	83.59,	85.23,	85.74]
# scaffold_medal = [62.18,	69.57,	73.1,	75.85,	77.12]
# ditto_medal= [66.01,	71.19,	75.51,	78.11,	78.4]
# fedpls_medal = [77.45,	81.23,	84.85,	86.36,	86.47]

fedavg_medal = [74.81,		72.2,		71.84,	65.09]
fedaprox_medal = [67.17,		63.46,		61.78,		57.68]
co_teaching = [76.56,		74.14,		72.6,		70.32]
rfl_medal= [70.9,		65.4,		63.77,		60.34]
fedcspl_medal = [77.9,		76.44,		75.35,		71.49]


# 创建折线图
# plt.plot(x, fedavg_medal, marker='o', label="Fedavg")
# plt.plot(x, fedrep_medal, marker='o', label="FedRep")
# plt.plot(x, ditto_medal, marker='o', label="Ditto")
# plt.plot(x, scaffold_medal, marker='o', label="Scaffold")
# plt.plot(x, fedpls_medal, marker='o', label="FedPLS")

plt.plot(x, fedavg_medal, marker='o', label="Fedavg-FT")
plt.plot(x, fedaprox_medal, marker='o', label="FedPorx")
plt.plot(x, co_teaching, marker='o', label="Co-teaching")
plt.plot(x, rfl_medal, marker='o', label="RFL")
plt.plot(x, fedcspl_medal, marker='o', label="FedPLSN")
# 设置标题和坐标轴标签

plt.title('CIFAR-10')
plt.ylabel('Test Accuracy(%)', fontsize=12)
plt.xlabel('Noise level', fontsize=12)
# 显示图表
plt.grid(True, color='gray', linestyle='--')
plt.legend(loc="upper right")
plt.show()



