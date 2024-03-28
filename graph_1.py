#库导入
from matplotlib import pyplot as plt
import numpy as np

#参数设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (6, 4)

#国家和奖牌数据导入
countries = ['c=2', 'c=4', 'c=6', 'c=8', 'c=10']
fedavg_medal = [47.92,	63,	65.44,	69.07,	73.63]
fedrep_medal = [84.05,	80.16,	74.88,	72.31,	70.47]
ditto_medal = [53.17,	65.28,	68.74,	72.59,	72.66]
scaffold_medal = [79.22,	68.9,	59.61,	55.84,	53.41]
fedpls_medal = [84.4,	80.99,	76.72,	75.04,	74.55]
x = np.array([0, 1.5, 3, 4.5, 6])



# countries = ['s=0', 's=0.2', 's=0.4', 's=0.6', 's=0.8', 's=1']
# fedavg_medal = [74.36,	69.19,	56.26,	42.8,	30.77,	16.88]
# fedrep_medal = [70.11,	71.06,	71.11,	71.03,	71,	71.34]
# ditto_medal = [74.81,	70.84,	56.51,	46.65,	31.61,	16.13]
# scaffold_medal = [65.9,	65.61,	64.72,	64.34,	63.65, 63.5]
# fedpls_medal = [74.95,	73.75,	73.83,	74.25,	73.42,	73.37]
# x = np.array([0, 1.5, 3, 4.5, 6, 7.5])
#将横坐标国家转换为数值
# x = np.arange(len(countries))

width = 0.2
#计算每一块的起始坐标
fedavg_x = x
fedrep_x = x + width
ditto_x = x + 2 * width
scaffold_x = x + 3 * width
fedpls_x = x + 4 * width

#绘图
plt.bar(fedavg_x,fedavg_medal,width=width,color="r",label="Fedavg")
plt.bar(fedrep_x,fedrep_medal,width=width,color="g",label="FedRep")
plt.bar(ditto_x,ditto_medal,width=width, color="b",label="Ditto")
plt.bar(scaffold_x,scaffold_medal,width=width,color="c",label="Scaffold")
plt.bar(fedpls_x,fedpls_medal,width=width, color="m",label="FedPLS")

#将横坐标数值转换为国家
plt.xticks(x + width,labels=countries)
plt.title('CIFAR-10')
plt.ylabel('Test Accuracy(%)', fontsize=12)
# plt.xlabel('Concept Shift Rate', fontsize=12)
plt.xlabel('Degree of heterogeneity', fontsize=12)

# #显示柱状图的高度文本
# for i in range(len(countries)):
#     plt.text(gold_x[i],gold_medal[i], gold_medal[i],va="bottom",ha="center",fontsize=8)
#     plt.text(silver_x[i],silver_medal[i], gold_medal[i],va="bottom",ha="center",fontsize=8)
#     plt.text(bronze_x[i],bronze_medal[i], gold_medal[i],va"bottom",ha="center",fontsize=8)

#显示图例
plt.legend(loc="lower right", borderaxespad=0)
plt.show()
