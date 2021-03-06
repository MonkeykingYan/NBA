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
FEATURES_COL = ['Pace', 'Reb Rate', 'Pts', 'Opp Pts']
path = 'data/3years.csv'
spark = SparkSession.builder.appName('NBA-Analysis').getOrCreate()
data = spark.read.csv(path, header=True, inferSchema=True)
data.printSchema()

vecAssembler = VectorAssembler(inputCols=FEATURES_COL, outputCol="features")
df_kmeans = vecAssembler.transform(data).select('Team', 'features')
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

k = 5
kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(df_kmeans)
centers = model.clusterCenters()

print("Cluster Centers: ")
for center in centers:
    print(center)

transformed = model.transform(df_kmeans).select('Team', 'prediction')
rows = transformed.collect()
print(rows[:3])

df_pred = data.join(transformed, 'Team')
pddf_pred = df_pred.toPandas().set_index('Team')

threedee = plt.figure(figsize=(12, 10)).gca(projection='3d')
threedee.scatter(pddf_pred['Pts'], pddf_pred['Opp Pts'], pddf_pred['Reb Rate'], s=20,
                 c=pddf_pred.prediction)
threedee.set_xlabel('Pts')
threedee.set_ylabel('Opp Pts')
threedee.set_zlabel('Reb Rate')
plt.interactive(True)
plt.ioff()
plt.show()
plt.savefig('KMeans.png')
