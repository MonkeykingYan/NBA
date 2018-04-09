import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col

# Configure the python
# os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3"
spark = SparkSession.builder.appName('NBA-Analysis').getOrCreate()
df = pd.read_csv('/Users/mayan/Desktop/BigData/project/Project_BigData/data/lineups_1516.csv')
df_players = pd.read_csv('playersClusters.csv')

# mySchema = StructType([StructField("RK", IntegerType(), True) \
#                           , StructField("Lineup", StringType(), True) \
#                           , StructField("TM", StringType(), True) \
#                           , StructField("Season", StringType(), True) \
#                           , StructField("G", StringType(), True) \
#                           , StructField("MP", StringType(), True) \
#                           , StructField("PTS", DoubleType(), True) \
#                           , StructField("PG", DoubleType(), True) \
#                           , StructField("PGA", DoubleType(), True) \
#                           , StructField("PG%", DoubleType(), True) \
#                           , StructField("3P", DoubleType(), True) \
#                           , StructField("3PA", DoubleType(), True) \
#                           , StructField("3P%", DoubleType(), True) \
#                           , StructField("eFG%", DoubleType(), True) \
#                           , StructField("FT", DoubleType(), True) \
#                           , StructField("FTA", DoubleType(), True) \
#                           , StructField("FT%", DoubleType(), True) \
#                           , StructField("ORB", DoubleType(), True) \
#                           , StructField("ORB%", DoubleType(), True) \
#                           , StructField("DRB", DoubleType(), True) \
#                           , StructField("DRB%", DoubleType(), True) \
#                           , StructField("TRB", DoubleType(), True) \
#                           , StructField("TRB%", DoubleType(), True) \
#                           , StructField("AST", DoubleType(), True) \
#                           , StructField("STL", DoubleType(), True) \
#                           , StructField("BLK", DoubleType(), True) \
#                           , StructField("TOV", DoubleType(), True) \
#                           , StructField("PF", DoubleType(), True)])

# df_p = spark.createDataFrame(df_players, schema=playersSchema)
df_p = spark.createDataFrame(df_players)
df_p.show(df_p.count(), False)

# df = spark.createDataFrame(df, schema=mySchema)
df_lineup = spark.createDataFrame(df)
df_lineup.show()


def getPlayers(df, teamName, k):
    players = {}
    i = 0
    for it in df.select('Lineup','MP').where(col('Tm') == teamName).collect():
        print(it[0])
        print(it[1])
        it = it[0].split('|')
        players[i] = it
        if (i == k):
            break
        i += 1
    return players


# res = getPlayers(df, 'GSW', 3)
# print(res)

def getTeamFeaturs(teamName, rank):
    feature = []
    res = getPlayers(df_lineup, teamName, rank)
    for i in res.keys():
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
    return feature


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

for p in ans:
    print(p)
    print(ans[p])
