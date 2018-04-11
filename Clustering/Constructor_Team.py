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
paths = ['teamClusters2011.csv', 'teamClusters2012.csv', 'teamClusters2013.csv', 'teamClusters2014.csv',
         'teamClusters2015.csv', 'teamClusters2016.csv']
spark = SparkSession.builder.appName('NBA-Analysis').getOrCreate()

allDataFrames = []
for p in paths:
    data = spark.read.csv(p, header=True, inferSchema=True)
    data.printSchema()
    data.show()

    data_rdd = data.rdd
    converted_data_rdd = data_rdd.map(lambda row: (row[0], row[1], ast.literal_eval(row[2]), row[3]))
    data = converted_data_rdd
    data = data.map(lambda x: (x[0], x[1], x[2], x[3], list(set(x[2]))))
    data = spark.createDataFrame(data, ["ID", "Team", "Features", 'Season', 'Items'])
    allDataFrames.append(data)

d0 = allDataFrames[0]
for index in range(1, len(allDataFrames)):
    d0 = d0.union(allDataFrames[index])
d0.show(d0.count(), False)
# data = spark.read.csv(path, header=True, inferSchema=True)
# data.printSchema()
# data.show()
#
# data_rdd = data.rdd
# converted_data_rdd = data_rdd.map(lambda row: (row[0], row[1], ast.literal_eval(row[2])))
# data = converted_data_rdd
# data = data.map(lambda x: (x[0], x[1], x[2], list(set(x[2]))))
# print(data.collect())
# data = spark.createDataFrame(data, ["ID", "Team", "Features", 'Items'])
# data.show()
# data.printSchema()

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

AllTeams = ['OKC', 'GSW', 'SAS', 'CLE', 'LAC',
            'TOR', 'CHO', 'DET', 'POR', 'ATL',
            'DAL', 'BOS', 'HOU', 'IND', 'UTA',
            'WAS', 'MEM', 'MIA', 'PHI', 'NYK',
            'NOP', 'LAL', 'BRK', 'ORL', 'SAC',
            'MIL', 'PHO', 'CHI', 'MIN', 'DEN']

AllYears = [2011, 2012, 2013,2014, 2015, 2016]
TeamDic = {}
words = []
for team in AllTeams:
    lis = d0.where(col("Team") == team).where(col('Season') == 2016).select('Features').collect()
    TeamDic[team] = str(lis[0][0])
    words.append(str(lis[0][0]))
# print(TeamDic)

# words = ['apple', 'Doppler', 'applaud', 'append', 'barker',
#          'baker', 'bismark', 'park', 'stake', 'steak', 'teak', 'sleek']

dist = [distance.edit_distance(words[i], words[j])
        for i in range(1, len(words))
        for j in range(0, i)]

labels, error, nfound = PC.kmedoids(dist, nclusters=3)  # kmedoids(dist, nclusters=3)
cluster = dict()
for word, label in zip(words, labels):
    cluster.setdefault(label, []).append(word)


# for label, grp in cluster.items():
#     print(grp)

def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0, size):
        list_of_objects.append(list())  # different object reference each time
    return list_of_objects


clusters = init_list_of_objects(int(3))
print(cluster.keys())
for key in cluster.keys():
    feature = cluster[key]
    c = []
    for f in feature:
        team = [k for k, v in TeamDic.items() if v == f]
        c.append(team)
    if (len(c) != 0):
        clusters.append(c)

index = 0
for list in clusters:
    if (len(list) != 0):
        print("* Class {}".format(index))
        print(sorted(list))
        index += 1

fpGrowth = FPGrowth(itemsCol="Items", minSupport=0.3, minConfidence=0.3)
model = fpGrowth.fit(d0)

# Display frequent itemsets.
model.freqItemsets.show()

# Display generated association rules.
model.associationRules.show()

# transform examines the input items against all the association rules and summarize the
# consequents as prediction
model.transform(d0).show()

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
