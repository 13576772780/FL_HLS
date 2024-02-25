#库导入
from matplotlib import pyplot as plt
import numpy as np

#参数设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (6, 4)

#国家和奖牌数据导入
countries = ['s=0%', 's=20%', 's=40%', 's=60%', 's=80%']
fedavg_ft_medal = [16, 12, 9, 8, 8]
fedrep_medal = [8, 10, 4, 10, 5]
ditto_medal = [13, 5, 2, 7, 5]
pfedme_medal = [13, 5, 2, 7, 5]
fedpls_medal = [13, 5, 2, 7, 5]


#将横坐标国家转换为数值
# x = np.arange(len(countries))
x = np.array([0, 1.5, 3, 4.5, 6])
width = 0.2

#计算每一块的起始坐标
fedavg_ft_x = x
fedrep_x = x + width
ditto_x = x + 2 * width
pfedme_x = x + 3 * width
fedpls_x = x + 4 * width

#绘图
plt.bar(fedavg_ft_x,fedavg_ft_medal,width=width,color="r",label="Fedavg_FT")
plt.bar(fedrep_x,fedrep_medal,width=width,color="g",label="FedRep")
plt.bar(ditto_x,ditto_medal,width=width, color="b",label="Ditto")
plt.bar(pfedme_x,pfedme_medal,width=width,color="c",label="pFedMe")
plt.bar(fedpls_x,fedpls_medal,width=width, color="m",label="FedPLS")

#将横坐标数值转换为国家
plt.xticks(x + width,labels=countries)
plt.title('CIFAR-10')
plt.ylabel('Test Accuracy(%)', fontsize=12)
plt.xlabel('Concept Shift Rate', fontsize=12)

# #显示柱状图的高度文本
# for i in range(len(countries)):
#     plt.text(gold_x[i],gold_medal[i], gold_medal[i],va="bottom",ha="center",fontsize=8)
#     plt.text(silver_x[i],silver_medal[i], gold_medal[i],va="bottom",ha="center",fontsize=8)
#     plt.text(bronze_x[i],bronze_medal[i], gold_medal[i],va"bottom",ha="center",fontsize=8)

#显示图例
plt.legend(loc="upper right")
plt.show()
