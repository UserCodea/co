from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

dataset, classes = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=0.5, random_state=0)

# make as panda dataframe for easy understanding
df = pd.DataFrame(dataset, columns=['var1', 'var2'])
print(df.head(2))
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12)).fit(df)
visualizer.show()
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(df)

print(kmeans.labels_)
print('=' * 30)
print(kmeans.n_iter_)
print('=' * 30)
print(kmeans.cluster_centers_)
print(Counter(kmeans.labels_))
print('=' * 30)

sns.scatterplot(data=df, x="var1", y="var2", hue=kmeans.labels_)
plt.show()

sns.scatterplot(data=df, x="var1", y="var2", hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
marker="X", c="r", s=80, label="centroids")

plt.legend()
plt.show()