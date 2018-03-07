from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

spark = SparkSession.builder.appName('NBA-Analysis').getOrCreate()
data = spark.read.csv('data/17_18_advanced.csv', header=True, inferSchema=True)

data.show()
cols = ['Pts', 'Avg Mrg','EFg%','Ts%']
assembler = VectorAssembler(inputCols=cols, outputCol='features')
assembled_data = assembler.transform(data)

scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')
scaler_model = scaler.fit(assembled_data)
scaled_data = scaler_model.transform(assembled_data)
scaled_data.printSchema()
scaled_data.select('scaledFeatures').show()

k_means_2 = KMeans(featuresCol='scaledFeatures', k=2)
k_means_3 = KMeans(featuresCol='scaledFeatures', k=3)
model_k2 = k_means_2.fit(scaled_data)
model_k3 = k_means_3.fit(scaled_data)

model_k3_data = model_k3.transform(scaled_data)
model_k3_data.groupBy('prediction').count()
model_k3_data.sort(col("prediction")).show()