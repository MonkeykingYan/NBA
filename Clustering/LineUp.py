import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import playerClustering.py

spark = SparkSession.builder.appName('NBA-Analysis').getOrCreate()
df = pd.read_csv('../data/cavs.csv')
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

df = spark.createDataFrame(df, schema=mySchema)
df.show()

def getPlayers(df, k):
    players = []
    i = 1
    for it in df.select('Lineup').collect():
        it = it[0].split('|')
        linup = {i:it}
        players.append(linup)
        if(i == k):
            break
        i+=1
    return players

res = getPlayers(df, 5)
print(res[0].get(1)[0])