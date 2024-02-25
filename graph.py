#库导入
from matplotlib import pyplot as plt
import numpy as np

#参数设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (5,3)

#国家和奖牌数据导入
countries = ['挪威', '德国', '中国', '美国', '瑞典']
gold_medal = [16, 12, 9, 8, 8]
silver_medal = [8, 10, 4, 10, 5]
bronze_medal = [13, 5, 2, 7, 5]

#将横坐标国家转换为数值
x = np.arange(len(countries))
width = 0.2

#计算每一块的起始坐标
gold_x = x
silver_x = x + width
bronze_x = x + 2 * width

#绘图
plt.bar(gold_x,gold_medal,width=width,color="gold",label="金牌")
plt.bar(silver_x,silver_medal,width=width,color="silver",label="银牌")
plt.bar(bronze_x,bronze_medal,width=width, color="saddlebrown",label="铜牌")

#将横坐标数值转换为国家
plt.xticks(x + width,labels=countries)

# #显示柱状图的高度文本
# for i in range(len(countries)):
#     plt.text(gold_x[i],gold_medal[i], gold_medal[i],va="bottom",ha="center",fontsize=8)
#     plt.text(silver_x[i],silver_medal[i], gold_medal[i],va="bottom",ha="center",fontsize=8)
#     plt.text(bronze_x[i],bronze_medal[i], gold_medal[i],va"bottom",ha="center",fontsize=8)

#显示图例
plt.legend(loc="upper right")
plt.show()
