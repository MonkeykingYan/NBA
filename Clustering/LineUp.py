import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import udf

spark = SparkSession.builder.appName('NBA-Analysis').getOrCreate()
df = pd.read_csv('../data/cavs.csv')
df_players = pd.read_csv('playersClusters.csv')

playersSchema = StructType([StructField("player", StringType(), True) \
                               , StructField("team_id", StringType(), True) \
                               , StructField("yr", IntegerType(), True) \
                               , StructField("prediction", IntegerType(), True)])

mySchema = StructType([StructField("RK", IntegerType(), True) \
                          , StructField("Lineup", StringType(), True) \
                          , StructField("MP", StringType(), True) \
                          , StructField("PTS", DoubleType(), True) \
                          , StructField("PG", DoubleType(), True) \
                          , StructField("PGA", DoubleType(), True) \
                          , StructField("PG%", DoubleType(), True) \
                          , StructField("3P", DoubleType(), True) \
                          , StructField("3PA", DoubleType(), True) \
                          , StructField("3P%", DoubleType(), True) \
                          , StructField("eFG%", DoubleType(), True) \
                          , StructField("FT", DoubleType(), True) \
                          , StructField("FTA", DoubleType(), True) \
                          , StructField("FT%", DoubleType(), True) \
                          , StructField("ORB", DoubleType(), True) \
                          , StructField("ORB%", DoubleType(), True) \
                          , StructField("DRB", DoubleType(), True) \
                          , StructField("DRB%", DoubleType(), True) \
                          , StructField("TRB", DoubleType(), True) \
                          , StructField("TRB%", DoubleType(), True) \
                          , StructField("AST", DoubleType(), True) \
                          , StructField("STL", DoubleType(), True) \
                          , StructField("BLK", DoubleType(), True) \
                          , StructField("TOV", DoubleType(), True) \
                          , StructField("PF", DoubleType(), True)])

df_p = spark.createDataFrame(df_players, schema=playersSchema)
df_p.show(df_p.count(), False)

df = spark.createDataFrame(df, schema=mySchema)
df.show()


def getPlayers(df, k):
    players = {}
    i = 1
    for it in df.select('Lineup').collect():
        it = it[0].split('|')
        players[i] = it
        if (i == k):
            break
        i += 1
    return players


res = getPlayers(df, 5)

feature = []
for i in res.keys():
    players = res[i]
    for p in players:
        f = df_p.filter(df_p.player.like(p[3:]+'%')).filter(df_p.player.like('%'+p[0]+'%')).distinct()
        feac = f.select('prediction').collect()
        print(p)
        if(len(feac)!=0):
            feature.append(feac[0][0])


print(feature)
