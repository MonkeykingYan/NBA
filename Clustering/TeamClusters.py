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

# os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3"
path = 'teamClusters2011.csv'
spark = SparkSession.builder.appName('NBA-Analysis').getOrCreate()
data = spark.read.csv(path, header=True, inferSchema=True)
data.printSchema()
data.show()

# fpGrowth = FPGrowth(itemsCol="Features", minSupport=0.5, minConfidence=0.6)
# model = fpGrowth.fit(data)
#
# # Display frequent itemsets.
# model.freqItemsets.show()
#
# # Display generated association rules.
# model.associationRules.show()
#
# # transform examines the input items against all the association rules and summarize the
# # consequents as prediction
# model.transform(data).show()


AllTeams = ['OKC', 'GSW', 'SAS', 'CLE', 'LAC',
            'TOR', 'CHO', 'DET', 'POR', 'ATL',
            'DAL', 'BOS', 'HOU', 'IND', 'UTA',
            'WAS', 'MEM', 'MIA', 'PHI', 'NYK',
            'NOP', 'LAL', 'BRK', 'ORL', 'SAC',
            'MIL', 'PHO', 'CHI', 'MIN', 'DEN']

TeamDic = {}
words = []
for team in AllTeams:
    lis = data.where(col("Team") == team).select('Features').collect()
    TeamDic[team] = lis[0][0]
    words.append(lis[0][0])
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
    if(len(list)!=0):
        print("* Class {}".format(index))
        print(sorted(list))
        index += 1
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


# final_dis = []
# for d in data.collect():  # all the data set
#     for index_c in range(0, len(c)):  # all the center
#         final_dis.append(distance(c[index_c][1], d[1]))# distance
#     indx_min = final_dis.index(min(final_dis))
#     final_dis = []
#     clusters[indx_min].append(d[0])

# clusters = sorted(clusters, key=itemgetter(0))
#
# index = 0
# for list in clusters:
#     print("* Class {}".format(index))
#     print(" ".join(sorted(list))+" ")
#     index+=1
