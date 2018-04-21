import json
from pprint import pprint
import pandas as pd
import os
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext
from pyspark import sql
from pyspark.sql.functions import col
from pyspark.sql.functions import udf

conf = SparkConf()
conf.setMaster('local')
conf.setAppName('PassingNetwork')
sc = SparkContext(conf=conf)

os.chdir("../PageRank/GSW2016")
# os.environ["PYSPARK_SUBMIT_ARGS"] = (
#     "--packages graphframes:graphframes:0.3.0-spark2.0-s_2.11 pyspark-shell"
# )

from graphframes import *

spark = SparkSession.builder.getOrCreate()
raw = pd.DataFrame()

# GSW
playerids = [2738, 202691, 101106, 2760, 2571, 203949, 203546,
             203110, 201939, 203105, 2733, 1626172, 203084]

# MIA
# playerids = [951, 1740, 2203, 2365, 2544, 2547, 2548, 2617, 201202, 201563, 201596, 201962, 202708]
# Calling API and store the results as JSON


# for playerid in playerids:
#     os.system('curl "http://stats.nba.com/stats/playerdashptpass?'
#         'DateFrom=&'
#         'DateTo=&'
#         'GameSegment=&'
#         'LastNGames=0&'
#         'LeagueID=00&'
#         'Location=&'
#         'Month=0&'
#         'OpponentTeamID=0&'
#         'Outcome=&'
#         'PerMode=Totals&'
#         'Period=0&'
#         'PlayerID={playerid}&'
#         'Season=2015-16&'
#         'SeasonSegment=&'
#         'SeasonType=Regular+Season&'
#         'TeamID=0&'
#         'VsConference=&'
#         'VsDivision=" > {playerid}.json'.format(playerid=playerid))


for player in playerids:
    with open(str(player) + '.json') as json_data:
        d = json.load(json_data)['resultSets'][0]
        raw = raw.append(
            pd.DataFrame(d['rowSet'], columns=d['headers']))
        print(d)

raw = raw.rename(columns={'PLAYER_NAME_LAST_FIRST': 'PLAYER'})

# Save passes.csv for plotting
raw[raw['PASS_TO']
    .isin(raw['PLAYER'])][['PLAYER', 'PASS_TO', 'PASS']].to_csv(
    'passes.csv', index=False)

raw['id'] = raw['PLAYER'].str.replace(', ', '')

# Make raw vertices
pandas_vertices = raw[['PLAYER', 'id']].drop_duplicates()
pandas_vertices.columns = ['name', 'id']

# Make raw edges
pandas_edges = pd.DataFrame()
for passer in raw['id'].drop_duplicates():
    for receiver in raw[(raw['PASS_TO'].isin(raw['PLAYER'])) &
                        (raw['id'] == passer)]['PASS_TO'].drop_duplicates():
        pandas_edges = pandas_edges.append(pd.DataFrame(
            {'passer': passer, 'receiver': receiver
                .replace(', ', '')},
            index=range(int(raw[(raw['id'] == passer) &
                                (raw['PASS_TO'] == receiver)]['PASS'].values))))

pandas_edges.columns = ['src', 'dst']
sqlContext = sql.SQLContext(sc)
# Bring the local vertices and edges to Spark
# vertices = pandas_vertices
vertices = sqlContext.createDataFrame(pandas_vertices)
# edges = pandas_edges
edges = sqlContext.createDataFrame(pandas_edges)
# Analysis part
print(type(vertices))
print(type(edges))

g = GraphFrame(vertices, edges)
print("vertices")
g.vertices.show()
print("edges")
g.edges.show()
print("inDegrees")
g.inDegrees.sort('inDegree', ascending=False).show()
print("outDegrees")
g.outDegrees.sort('outDegree', ascending=False).show()
print("degrees")
g.degrees.sort('degree', ascending=False).show()
print("labelPropagation")
g.labelPropagation(maxIter=5).show()
g.labelPropagation(maxIter=5).toPandas().to_csv("groups.csv", index=False)
print("pageRank")
d0 = g.pageRank(resetProbability=0.15, tol=0.01).vertices.sort(
    'pagerank', ascending=False)
g.pageRank(resetProbability=0.15, tol=0.01).vertices.sort(
    'pagerank', ascending=False).show()
g.pageRank(resetProbability=0.15, tol=0.01).vertices.toPandas().to_csv(
    "size.csv", index=False)

# df_players = pd.read_csv('/Users/mayan/Desktop/BigData/project/Project_BigData/Clustering/playersClusters.csv')
# # df_p = spark.createDataFrame(df_players, schema=playersSchema)
# df_p = spark.createDataFrame(df_players)
# df_p.show()
# label = df_p.where(col("player") == 'Allen,Ray').select('prediction').distinct().collect()
# print(label[0][0])
# def setPlayerLabel(playerName):
#     pn = playerName.strip()
#     print(pn)
#     label = df_p.where(col("player")==pn).select('prediction').distinct().collect()
#     if (len(label) == 0):
#         return -1
#     return label[0][0]
#
#
# udf_yan = udf(setPlayerLabel)
#
# d0 = d0.withColumn('player Label', udf_yan(d0['name'])).show()