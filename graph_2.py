#库导入
import matplotlib.pyplot as plt

# 准备数据
x = ["p=0", "p=0.4", "p=0.6", "p=0.8"]

# fedavg_medal = [60.85,	64.16,	68.92,	71.71,	76.08]
# fedrep_medal = [76.17,	79.62,	83.59,	85.23,	85.74]
# scaffold_medal = [62.18,	69.57,	73.1,	75.85,	77.12]
# ditto_medal= [66.01,	71.19,	75.51,	78.11,	78.4]
# fedpls_medal = [77.45,	81.23,	84.85,	86.36,	86.47]

fedavg_medal = [77.4,	72.8, 69.12, 62.62]
fedaprox_medal = [77.16,	66.88, 59.38, 54.82]
co_teaching = [75.86,	71.88,		68.98,		63.24]
rfl_medal= [77.12,	74.74,		69.54, 66.56]
fedcspl_medal = [78.02,	75.72, 72.1,	69.4]


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



