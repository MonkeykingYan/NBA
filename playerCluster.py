from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SQLContext

# ReadFile
FEATURES_COL = ['fg3a', 'trb', 'ast', 'blk', 'tov']
path = 'data/allPlayers.csv'
spark = SparkSession.builder.appName('NBA-Analysis').getOrCreate()
data = spark.read.csv(path, header=True, inferSchema=True)
data.printSchema()

data = data.where((col('mp') > 1000) & ((col("yr") == 2015) | (col("yr") == 2016)))
data = data.na.fill(0)

vecAssembler = VectorAssembler(inputCols=FEATURES_COL, outputCol="features")
df_kmeans = vecAssembler.transform(data).select('player', 'features')
print("!!!!!!!")
print(type(df_kmeans))
df_kmeans.show()

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

k = 3
kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(df_kmeans)
centers = model.clusterCenters()

print("Cluster Centers: ")
for center in centers:
    print(center)

transformed = model.transform(df_kmeans).select('player', 'prediction')
rows = transformed.collect()
print(rows[:3])

df_pred = data.join(transformed, 'player')
pddf_pred = df_pred.toPandas().set_index('player')

threedee = plt.figure(figsize=(12, 10)).gca(projection='3d')
threedee.scatter(pddf_pred['fg3a'], pddf_pred['trb'], pddf_pred['ast'], s=20,
                 c=pddf_pred.prediction)
threedee.set_xlabel('fg3a')
threedee.set_ylabel('trb')
threedee.set_zlabel('ast')
plt.interactive(True)
plt.ioff()
plt.show()
plt.savefig('KMeans.png')
