import numpy as np
from sklearn.decomposition import PCA
from pyspark.sql.types import *
import numpy as np
import sklearn.cluster
from nltk.metrics import distance
import Pycluster as PC
from pyspark.sql import SparkSession

path = 'teamClusters.csv'
spark = SparkSession.builder.appName('NBA-Analysis').getOrCreate()
data = spark.read.csv(path, header=True, inferSchema=True)
data.printSchema()
data.show()

AllTeams = ['OKC', 'GSW', 'SAS', 'CLE', 'LAC',
            'TOR', 'CHO', 'DET', 'POR', 'ATL',
            'DAL', 'BOS', 'HOU', 'IND', 'UTA',
            'WAS', 'MEM', 'MIA', 'PHI', 'NYK',
            'NOP', 'LAL', 'BRK', 'ORL', 'SAC',
            'MIL', 'PHO', 'CHI', 'MIN', 'DEN']

TeamDic = {}
words = []
for team in AllTeams:
    lis = data.select(team).collect()
    TeamDic[team] = lis[0][0]
    words.append(lis[0][0])
# print(TeamDic)

# words = ['apple', 'Doppler', 'applaud', 'append', 'barker',
#          'baker', 'bismark', 'park', 'stake', 'steak', 'teak', 'sleek']

dist = [distance.edit_distance(words[i], words[j])
        for i in range(1, len(words))
        for j in range(0, i)]

labels, error, nfound = PC.kmedoids(dist, nclusters=3)#kmedoids(dist, nclusters=3)
cluster = dict()
for word, label in zip(words, labels):
    cluster.setdefault(label, []).append(word)
for label, grp in cluster.items():
    print(grp)

print(cluster.keys())
for key in cluster.keys():
    feature = cluster[key]
    print(feature)
    for f in feature:
        team = [k for k, v in TeamDic.items() if v == f]
        print(team)
# data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
# data = np.array(data)
# np.savetxt("foo.csv", data, delimiter=",")
#
#
# schema = StructType([StructField('Name', StringType(), True),
#                      StructField('DateTime', TimestampType(), True),
#                      StructField('Age', IntegerType(), True)])
# print(data)
# pca = PCA(n_components=3)
# pca.fit(data)
# X = pca.transform(data)
# print(X)