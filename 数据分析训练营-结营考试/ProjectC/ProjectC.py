from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, ward

#Reading car data
data = pd.read_csv('CarPrice_Assignment.csv', encoding='gbk')
print(data)
#保留数据中所需列. enginelocation全为front,删去
data_1 = data.drop(['car_ID', 'symboling', 'CarName', 'enginelocation'], 1)
print(data_1)

#train_x = data[["fueltype", "aspiration", "doornumber", "carbody", "drivewheel", "enginetype", "cylindernumber", "fuelsystem"]]
ss = ["fueltype", "aspiration", "doornumber", "carbody", "drivewheel", "enginetype", "cylindernumber", "fuelsystem"]

# LabelEncoder
le = LabelEncoder()
for k in ss:
    data_1[k] = le.fit_transform(data_1[k])
#print(data_1)

#规范化
data_1 = preprocessing.MinMaxScaler().fit_transform(data_1)

#找出Kmeans合适的分类个数
#K-Means number of clusters
sse = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k).fit(data_1)
    sse.append(kmeans.inertia_)
x = range(1, 20)
plt.xlabel('Number of clusters')
plt.ylabel('Within Sum of Square')
plt.plot(x, sse, 'x-')
plt.show()

#选择分六或七类较为合理
#KMeans聚类，kmeans为训练好的模型
kmeans = KMeans(n_clusters=7).fit(data_1)
print(kmeans)
print(kmeans.n_iter_)

#合并聚类结果，插入到原数据集
result = pd.concat((data, pd.DataFrame(kmeans.labels_)), axis=1)
#给单元格附名
result.rename({0:u'classification'}, axis=1, inplace=True)
#print(result)
result.to_csv("Poject_c.csv",index=True)
#VW车型出现在五个分类中，其他品牌分类相同的即为竞品


