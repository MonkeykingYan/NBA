import numpy as np
from sklearn.decomposition import PCA
from pyspark.sql.types import *
import numpy as np
import sklearn.cluster
from nltk.metrics import distance
import Pycluster as PC
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import udf
import ast

# os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3"
path = 'teamClusters.csv'
spark = SparkSession.builder.appName('NBA-Analysis').getOrCreate()

data = spark.read.csv(path, header=True, inferSchema=True)
data.printSchema()
data.show()

data_rdd = data.rdd
converted_data_rdd = data_rdd.map(lambda row: (row[0], row[1], ast.literal_eval(row[2])))
data = converted_data_rdd
data = data.map(lambda x: (x[0], x[1], x[2], list(set(x[2]))))
print(data.collect())
data = spark.createDataFrame(data, ["ID", "Team", "Features", 'Items'])
data.show()
data.printSchema()

# converted_data = spark.createDataFrame(converted_data_rdd, ["ID", "Team", "Features"])
# converted_data.show()
# converted_data.printSchema()
# def map_convert_string_to_array(data):
#     return set(ast.literal_eval(data))
#
# udf_map_convert_string_to_array = udf(map_convert_string_to_array)
#
# data = data.withColumn('Items', set(col('Features')))
# data.show()
# data.printSchema()

fpGrowth = FPGrowth(itemsCol="Items", minSupport=0.5, minConfidence=0.6)
model = fpGrowth.fit(data)

# Display frequent itemsets.
model.freqItemsets.show()

# Display generated association rules.
model.associationRules.show()

# transform examines the input items against all the association rules and summarize the
# consequents as prediction
model.transform(data).show()

# AllTeams = ['OKC', 'GSW', 'SAS', 'CLE', 'LAC',
#             'TOR', 'CHO', 'DET', 'POR', 'ATL',
#             'DAL', 'BOS', 'HOU', 'IND', 'UTA',
#             'WAS', 'MEM', 'MIA', 'PHI', 'NYK',
#             'NOP', 'LAL', 'BRK', 'ORL', 'SAC',
#             'MIL', 'PHO', 'CHI', 'MIN', 'DEN']
#
# TeamDic = {}
# words = []
# for team in AllTeams:
#     lis = data.where(col("Team") == team).select('Features').collect()
#     print(lis[0][0])
#     TeamDic[team] = lis[0][0]
#     words.append(lis[0][0])
