# Import all the packages
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

# ReadFile
# All the features
FEATURES_COL = ['fg', 'fga', 'fg3', 'fg3a', 'fg2', 'fg2a', 'ft', 'fta', 'orb', 'drb', 'trb',
                'ast', 'stl', 'blk', 'tov', 'pts', 'fg_pct', 'fg2_pct', 'fg3_pct', 'efg_pct']

path = 'data/allPlayers.csv'
spark = SparkSession.builder.appName('Player-Classifier').getOrCreate()
data = spark.read.csv(path, header=True, inferSchema=True)
data.printSchema()

data = data.where((col('mp') / col('g') > 20) & (
        (col("yr") == 2016) | (col("yr") == 2015) | (col("yr") == 2014) | (col("yr") == 2013) | (col("yr") == 2012) | (
        col("yr") == 2011) | (col("yr") == 2010))).filter(col('g') > 50)
data = data.na.fill(0)
data.show(data.count(), False)
'''
Normalizations part
'''
newFEATURES_COL = []


def normalize(data, name):
    newCol = "normalized_" + name
    # newFEATURES_COL.append(newCol)
    data = data.withColumn(newCol, ((col(name) / col('g'))))

    min_age, max_age = data.select(min(newCol), max(newCol)).first()
    newCol2 = "normalized_" + newCol
    newFEATURES_COL.append(newCol2)
    data = data.withColumn(newCol2, ((col(newCol) - min_age) / (
            max_age - min_age)))
    return data


for c in FEATURES_COL:
    data = normalize(data, c)
data.select(newFEATURES_COL).sort('player').show()
'''
Normalizations part
'''

vecAssembler = VectorAssembler(inputCols=newFEATURES_COL, outputCol="Features")
df_kmeans = vecAssembler.transform(data)  # .select('player', 'Features')
costPCA = np.zeros(10)
for pcak in range(1, 10):
    pca = PCA(k=pcak, inputCol="Features", outputCol="features")
    model = pca.fit(df_kmeans)
    costPCA[pcak] = sum(model.explainedVariance)


plt.interactive(True)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(range(1, 10), costPCA[1:10])
ax.set_xlabel('k')
ax.set_ylabel('Explained Variance Ratio')
plt.ioff()
fig.show()
plt.savefig('ExplainedRatio.png')

pca = PCA(k=3, inputCol="Features", outputCol="features")
model = pca.fit(df_kmeans)
df_kmeans = model.transform(df_kmeans)# select('player', "features")
print(model.explainedVariance)
features = df_kmeans.select('features').rdd.map(lambda x: np.array(x))
for it in features.collect():
    print(it)
print(type(features))
# result.show(truncate=False)

cost = np.zeros(20)
for k in range(2, 20):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(df_kmeans.sample(False, 0.1, seed=99))
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

transformed = model.transform(df_kmeans).select('player', 'features', 'prediction', 'yr', 'team_id').sort('player')
print("this is the transformaed")
transformed.show(transformed.count(), False)
# rows = transformed.collect()
# print(rows[:3])
# print(type(transformed))
# transformed.show()
#
# df_pred = df_kmeans.join(transformed, 'player')
# df_pred.printSchema()
# # print(type(df_pred["features"].fieldIndex[]))
# arr = df_pred.select('features').collect()
df_pred = transformed
arr = df_pred.select('features').collect()
feature1 = []
feature2 = []
feature3 = []
for it in arr:
    feature1.append(it[0][0])
    feature2.append(it[0][1])
    feature3.append(it[0][2])
ans = df_pred.select('player', 'team_id', 'yr', 'prediction').sort('player').distinct()
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
ans.to_csv('Clustering/playersClusters.csv', sep=',')
