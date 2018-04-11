import numpy as np
from sklearn.decomposition import PCA
from pyspark.sql.types import *
import numpy as np
import sklearn.cluster
from nltk.metrics import distance
import Pycluster as PC
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.fpm import FPGrowth
import ast

# os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3"
# path = 'teamClusters2011.csv'
# spark = SparkSession.builder.appName('NBA-Analysis').getOrCreate()
# data = spark.read.csv(path, header=True, inferSchema=True)
# data.printSchema()
# data.show()

path = 'ClustersTeams.csv'

spark = SparkSession.builder.appName('NBA-Analysis').getOrCreate()

data = spark.read.csv(path, header=True, inferSchema=True)
data.printSchema()
data.show()

data_rdd = data.rdd
converted_data_rdd = data_rdd.map(lambda row: (row[0], row[1], ast.literal_eval(row[2]), row[3], row[4], row[5]))
data = converted_data_rdd
data = data.map(lambda x: (x[0], x[1], x[2], x[3], list(set(x[2])), x[5]))
data = spark.createDataFrame(data, ["ID", "Team", "Features", 'Season', 'Items', 'label'])
data.show()

d0 = data.where(col("label") == 1).select("Team", 'Season','Items', 'label')
d0.show(d0.count(), False)

fpGrowth = FPGrowth(itemsCol="Items", minSupport=0.5, minConfidence=0.5)
model = fpGrowth.fit(d0)

# Display frequent itemsets.
model.freqItemsets.show(8)

# Display generated association rules.
model.associationRules.show(8)

# transform examines the input items against all the association rules and summarize the
# consequents as prediction
model.transform(d0).show(8)


