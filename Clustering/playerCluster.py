from IPython.utils.syspathcontext import prepended_to_syspath
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
import pyspark.sql.functions as F
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SQLContext

from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
import os

from pyspark.sql.functions import stddev, mean, min, max, col

# Configure the python
os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3"
# ReadFile
# All the features
FEATURES_COL = ['fg', 'fga', 'fg3', 'fg3a', 'fg2', 'fg2a', 'ft', 'fta', 'orb', 'drb',
                'trb',
                'ast', 'stl', 'blk', 'tov', 'pts']
path = 'data/allPlayers.csv'
spark = SparkSession.builder.appName('NBA-Analysis').getOrCreate()
data = spark.read.csv(path, header=True, inferSchema=True)
data.printSchema()

data = data.where((col('mp') / col('g') > 20) & (
    (col("yr") == 2016)))
data = data.na.fill(0)
'''
Normalizations part
'''
newFEATURES_COL = []


def normalize(data, name):
    min_age, max_age = data.select(min(name), max(name)).first()
    newCol = "normalized_" + name
    newFEATURES_COL.append(newCol)
    data = data.withColumn(newCol, ((col(name) / col('mp') - min_age) / (
            max_age - min_age)))
    return data


for c in FEATURES_COL:
    data = normalize(data, c)

data.select(newFEATURES_COL).show()
'''
Normalizations part
'''

vecAssembler = VectorAssembler(inputCols=newFEATURES_COL, outputCol="Features")
df_kmeans = vecAssembler.transform(data)  # .select('player', 'Features')

pca = PCA(k=3, inputCol="Features", outputCol="features")
model = pca.fit(df_kmeans)
df_kmeans = model.transform(df_kmeans)  # select('player', "features")
features = df_kmeans.select('features').rdd.map(lambda x: np.array(x))
for it in features.collect():
    print(it)
print(type(features))
# result.show(truncate=False)

cost = np.zeros(20)
for k in range(2, 20):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(df_kmeans.sample(False, 0.1, seed=42))
    cost[k] = model.computeCost(df_kmeans)

plt.interactive(True)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(range(2, 20), cost[2:20])
ax.set_xlabel('k')
ax.set_ylabel('cost')
plt.ioff()
fig.show()
plt.savefig('K_Selection.png')

k = 10
kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(df_kmeans)
centers = model.clusterCenters()

print("Cluster Centers: ")
for center in centers:
    print(center)

transformed = model.transform(df_kmeans).select('player', 'features', 'prediction')
rows = transformed.collect()
print(rows[:3])
print(type(transformed))
transformed.show()

df_pred = data.join(transformed, 'player')
df_pred.printSchema()
# print(type(df_pred["features"].fieldIndex[]))
arr = df_pred.select('features').collect()
feature1 = []
feature2 = []
feature3 = []
for it in arr:
    feature1.append(it[0][0])
    feature2.append(it[0][1])
    feature3.append(it[0][2])
# df_pred = df_pred.withColumn("feature1", df_pred["features"][0]).withColumn("feature2", df_pred["features"].getItem(1)).withColumn("feature3", df_pred["features"].getItem(2))
# df_pred.show()
ans = df_pred.select('player', 'team_id', 'yr', 'prediction').sort('prediction')
ans.show(ans.count(), False)
pddf_pred = df_pred.toPandas().set_index('player')

threedee = plt.figure(figsize=(12, 10)).gca(projection='3d')
threedee.scatter(feature1, feature2, feature3, s=30,
                 c=pddf_pred.prediction)
threedee.set_xlabel('f1')
threedee.set_ylabel('f2')
threedee.set_zlabel('f3')
plt.interactive(True)
plt.ioff()
plt.show()
plt.savefig('KMeans.png')
ans = ans.toPandas().set_index('player')
ans.to_csv('Clustering/playersClusters.csv', sep=' ')
