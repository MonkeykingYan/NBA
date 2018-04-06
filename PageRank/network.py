import json
from pprint import pprint
import pandas as pd
import os
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext
from pyspark import sql


conf = SparkConf()
conf.setMaster('local')
conf.setAppName('PassingNetwork')
sc = SparkContext(conf=conf)

os.chdir("/Users/mayan/Desktop/BigData/data")
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages graphframes:graphframes:0.3.0-spark2.0-s_2.11"
)

spark = SparkSession.builder.getOrCreate()
raw = pd.DataFrame()
playerids = [2738, 202691, 101106, 2760, 2571, 203949, 203546,
             203110, 201939, 203105, 2733, 1626172, 203084]
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

from graphframes import *

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
g.pageRank(resetProbability=0.15, tol=0.01).vertices.sort(
    'pagerank', ascending=False).show()
g.pageRank(resetProbability=0.15, tol=0.01).vertices.toPandas().to_csv(
    "size.csv", index=False)
