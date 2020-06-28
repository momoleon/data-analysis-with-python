from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, ward

#Reading car data
data = pd.read_csv('car_data.csv', encoding='gbk')
train_x = data[["人均GDP", "城镇人口比重", "交通工具消费价格指数", "百户拥有汽车量"]]
#print(train_x)

#找出Kmeans合适的分类个数
"""
#K-Means number of clusters
sse = []
for k in range(1, 7):
    kmeans = KMeans(n_clusters=k).fit(train_x)
    sse.append(kmeans.inertia_)
x = range(1, 7)
plt.xlabel('Number of clusters')
plt.ylabel('Within Sum of Square')
plt.plot(x, sse, 'x-')
plt.show()
"""

#规范化
train_x = preprocessing.MinMaxScaler().fit_transform(train_x)
#print(train_x)
#pd.DataFrame(train_x).to_csv('car_result.csv', index=False)

#KMeans聚类，kmeans为训练好的模型
kmeans = KMeans(n_clusters=3).fit(train_x)
print(kmeans)
print(kmeans.n_iter_)
#Center of cluster
#print(kmeans.cluster_centers_)
#print(kmeans.labels_)
#针对测试集可以用训练好的模型进行进一步分类
#predict_y = kmeans.predict(train_x)
#print(predict_y)

"""
#KMeans可视化 ##多维问题，可视化不方便
color = ['r', 'g', 'b']
k = 3
centroids = kmeans.cluster_centers_
print(centroids)
predict_label = kmeans.predict(train_x)
for i in range(k):
    plt.scatter(centroids[i][2], centroids[i][3], c=color[i], marker='X', s=60)
print(data)

for (_data, _label) in zip(train_x, predict_label):
    plt.scatter(_data[2], _data[3], color=color[_label],alpha=0.3)
plt.show()
"""


#合并聚类结果，插入到原数据集
result = pd.concat((data, pd.DataFrame(kmeans.labels_)), axis=1)
#给单元格附名
result.rename({0:u'聚类'}, axis=1, inplace=True)
#print(result)
result.to_csv("car_cluster_result.csv",index=True)

"""
#层次聚类建模方法
from sklearn.cluster import KMeans, AgglomerativeClustering
model = AgglomerativeClustering(linkage='ward', n_clusters=3)
y = model.fit_predict(train_x)
print(y)
"""
#可以用层次聚类的方式对KMeans方法进行可视化，结果大概率一致
#print(train_x)
linkage_matrix = ward(train_x)
dendrogram(linkage_matrix)
plt.show()
