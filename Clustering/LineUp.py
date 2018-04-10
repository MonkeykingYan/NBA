import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col
import math
import numpy as np
from itertools import *
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Configure the python
# os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3"
spark = SparkSession.builder.appName('NBA-Analysis').getOrCreate()
df = pd.read_csv('/Users/mayan/Desktop/BigData/project/Project_BigData/data/lineups_1516.csv')
df_players = pd.read_csv('playersClusters.csv')

# df_p = spark.createDataFrame(df_players, schema=playersSchema)
df_p = spark.createDataFrame(df_players)
df_p.show(df_p.count(), False)

# df = spark.createDataFrame(df, schema=mySchema)
df_lineup = spark.createDataFrame(df)
df_lineup.show()


def getPlayers(df, teamName, k):
    players = {}
    weight = []
    i = 0
    for it in df.select('Lineup', 'MP').where(col('Tm') == teamName).collect():
        print(it[0])
        print(it[1])
        weight.append(it[1])
        it = it[0].split('|')
        players[i] = it
        if (i == k):
            break
        i += 1
    return (players, weight)


# res = getPlayers(df, 'GSW', 3)
# print(res)

def getTeamFeaturs(teamName, rank):
    matrix = []
    (res, weight) = getPlayers(df_lineup, teamName, rank)
    print(res)
    for i in res.keys():
        feature = []
        players = res[i]
        for p in players:
            p = p.strip()
            print('%' + p[3:len(p)] + '%')
            print('%,' + p[0] + '%')
            f = df_p.filter(df_p.player.like('%' + p[3:len(p)] + '%')).filter(
                df_p.player.like('%,' + p[0] + '%')).distinct()
            feac = f.select('prediction').collect()
            if (len(feac) != 0):
                feature.append(feac[0][0])
        matrix = constructFeatureMatrix(feature, math.ceil(weight[i] * 10 / sum(weight)), matrix)
    if (len(matrix) > 10):
        matrix = matrix[:10]
    else:
        while (len(matrix) < 10):
            matrix.append(matrix[0])
    return matrix


def constructFeatureMatrix(lineup, weight, matrix):
    for w in range(1, weight):
        matrix.append(lineup)
    return matrix


AllTeams = ['OKC', 'GSW', 'SAS', 'CLE', 'LAC',
            'TOR', 'CHO', 'DET', 'POR', 'ATL',
            'DAL', 'BOS', 'HOU', 'IND', 'UTA',
            'WAS', 'MEM', 'MIA', 'PHI', 'NYK',
            'NOP', 'LAL', 'BRK', 'ORL', 'SAC',
            'MIL', 'PHO', 'CHI', 'MIN', 'DEN']

ans = {}
for team in AllTeams:
    # print(team)
    ans[team] = getTeamFeaturs(team, 2)

TeamDic = {}
for a in ans:
    A = list(chain.from_iterable(ans[a]))
    TeamDic[a] = A

print(TeamDic)
s = pd.Series(TeamDic, name='features')
s.index.name = 'Team'
s.reset_index()
print(type(s))
print(s)

df = pd.DataFrame([s])
df.to_csv('teamClusters.csv')
