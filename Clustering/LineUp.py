import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col
import math
import numpy as np
from itertools import *
from pyspark.sql.functions import lit
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Configure the python
# os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3"
spark = SparkSession.builder.appName('NBA-Analysis').getOrCreate()
df = pd.read_csv('../data/lineups_11To16.csv')
df_players = pd.read_csv('playersClusters.csv')

# df_p = spark.createDataFrame(df_players, schema=playersSchema)
df_p = spark.createDataFrame(df_players)
df_p.show(df_p.count(), False)

# df = spark.createDataFrame(df, schema=mySchema)
df_lineup = spark.createDataFrame(df)
df_lineup.show(df_lineup.count(), False)


def getPlayersFromLineups(df, teamName, k, Season):
    players = {}
    weight = []
    i = 0
    for it in df.select('Lineup', 'MP', 'Season').where(
            col('Tm') == teamName).where(
        col('Season') == Season).collect():
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
# Return linups with labels
def getLinupsByplayersLabel(linups, linups_Weight, Season):
    label_lineup = []
    for i in linups.keys():
        players = linups[i]
        for p in players:
            p = p.strip()
            f = df_p.where((df_p.yr == int(Season[:4]))|(df_p.yr == int(Season[:4])+1)).filter(
                 df_p.player.like('%' + p[3:len(p)] + '%'))#.filter(
                # df_p.player.like('%,' + p[0] + '%'))
            feac = f.select('prediction').collect()
            if (len(feac) != 0):
                label_lineup.append(feac[0][0])
    return (label_lineup, linups_Weight)


def getTeamFeaturs(teamName, rank, Season):
    matrix = []
    (res, weight) = getPlayersFromLineups(df_lineup, teamName, rank, Season)
    (lable_lineup, linups_Weight) = getLinupsByplayersLabel(res, weight, Season)
    for i in range(0, len(linups_Weight)):
        matrix = constructFeatureMatrix(lable_lineup, math.ceil(linups_Weight[i] * 10 / sum(linups_Weight)), matrix)
    if (len(matrix) == 0):
        return (matrix, False)
    if (len(matrix) > 10):
        matrix = matrix[:10]
    else:
        while (len(matrix) < 10):
            matrix.append(matrix[0])
    return (matrix, True)


# def getTeamFeaturs(teamName, rank, Season):
#     matrix = []
#     (res, weight) = getPlayersFromLineups(df_lineup, teamName, rank, Season)
#     print(res)
#     for i in res.keys():
#         feature = []
#         players = res[i]
#         for p in players:
#             p = p.strip()
#             print('%' + p[3:len(p)] + '%')
#             print('%,' + p[0] + '%')
#             print(int(Season[:4]))
#             f = df_p.filter(df_p.player.like('%' + p[3:len(p)] + '%')).filter(
#                 df_p.player.like('%,' + p[0] + '%')).distinct()
#             feac = f.select('prediction').collect()
#             if (len(feac) != 0):
#                 feature.append(feac[0][0])
#         matrix = constructFeatureMatrix(feature, math.ceil(weight[i] * 10 / sum(weight)), matrix)
#     if (len(matrix) > 10):
#         matrix = matrix[:10]
#     else:
#         while (len(matrix) < 10):
#             matrix.append(matrix[0])
#     return matrix


def constructFeatureMatrix(lineup, weight, matrix):
    for w in range(1, weight):
        matrix.append(lineup)
    return matrix


AllTeams = ['OKC', 'GSW', 'SAS', 'CLE', 'LAC',
            'TOR', 'CHO', 'DET', 'POR', 'ATL',
            'DAL', 'BOS', 'HOU', 'IND', 'UTA',
            'WAS', 'MEM', 'MIA', 'PHI', 'NYK',
            'NOP', 'LAL', 'BRK', 'ORL', 'SAC',
            'MIL', 'PHO', 'CHI', 'MIN', 'DEN',
            'NOH', 'NJN', 'CHA']

# def LineUp(k):
#     AllYears = [2011,2012,2013,2014,2015,2016]
#     AllTeams = ['OKC', 'GSW', 'SAS', 'CLE', 'LAC',
#                 'TOR', 'CHO', 'DET', 'POR', 'ATL',
#                 'DAL', 'BOS', 'HOU', 'IND', 'UTA',
#                 'WAS', 'MEM', 'MIA', 'PHI', 'NYK',
#                 'NOP', 'LAL', 'BRK', 'ORL', 'SAC',
#                 'MIL', 'PHO', 'CHI', 'MIN', 'DEN']
#     ans = {}
#     season_team = {}
#     res = []
#     for team in AllTeams:
#         for year in AllYears:
#             ans[team] = getTeamFeaturs(team, k, year)
#         season_team[year] = ans[team]
#     res.append(season_team)
#     return res
# res = LineUp(2)
# for i in res:
#     print(i)
AllYears = [2011, 2012, 2013, 2014, 2015, 2016]
AllSeasons = ['2011-12', '2012-13', '2013-14', '2014-15', '2015-16', '2016-17']
for season in AllSeasons:
    TeamDic = {}
    ans = {}
    for team in AllTeams:
        print(team)
        (Features, Empty) = getTeamFeaturs(team, 2, season)
        if (Empty):
            ans[team] = Features
        else:
            continue

    for a in ans:
        A = list(chain.from_iterable(ans[a]))
        TeamDic[a] = A[:50]

    print(TeamDic)
    if (len(TeamDic.keys()) != 0):
        constructor = []
        for teamName in TeamDic.keys():
            print(set(TeamDic[teamName]))
            constructor.append((teamName, TeamDic[teamName]))
    df = spark.createDataFrame(constructor, ["Team", "Features"]).withColumn('Season', lit(int(season[:4])))
    df.show()
    df.toPandas().to_csv('teamClusters{}.csv'.format(int(season[:4])))
