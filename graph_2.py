#库导入
import matplotlib.pyplot as plt

# 准备数据
x = [300, 600, 900, 1200, 1500, 1800]

fedavg_ft_medal = [16, 12, 9, 8, 8, 0]
fedrep_medal = [8, 10, 4, 10, 5, 6]
ditto_medal = [13, 5, 2, 7, 5, 9]
pfedme_medal = [13, 5, 2, 7, 5, 6]
fedpls_medal = [13, 5, 2, 7, 5, 11]

# 创建折线图
plt.plot(x, fedavg_ft_medal, marker='o', label="Fedavg_FT")
plt.plot(x, fedrep_medal, marker='o', label="FedRep")
plt.plot(x, ditto_medal, marker='o', label="Ditto")
plt.plot(x, pfedme_medal, marker='o', label="pFedMe")
plt.plot(x, fedpls_medal, marker='o', label="FedPLS")
# 设置标题和坐标轴标签

plt.title('CIFAR-10')
plt.ylabel('Test Accuracy(%)', fontsize=12)
plt.xlabel('Local Date Size', fontsize=12)
# 显示图表
plt.grid(True, color='gray', linestyle='--')
plt.legend(loc="upper right")
plt.show()



