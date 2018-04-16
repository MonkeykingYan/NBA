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
df = pd.read_csv('/Users/mayan/Desktop/BigData/project/Project_BigData/data/lineups_11To16.csv')
df_players = pd.read_csv('/Users/mayan/Desktop/BigData/project/Project_BigData/Clustering/playersClusters.csv')

# df_p = spark.createDataFrame(df_players, schema=playersSchema)
df_p = spark.createDataFrame(df_players)
df_p.show(df_p.count(), False)

# df = spark.createDataFrame(df, schema=mySchema)
df_lineup = spark.createDataFrame(df)
df_lineup.show(df_lineup.count(), False)

#Test
# f = df_p.filter(df_p.player.like('%' + 'Turner' + '%')).filter(
#     df_p.player.like('%,' + 'E' + '%')).filter(
#     (df_p.yr == 2011)).distinct()
f = df_p.filter(df_p.team_id == 'PHI').filter(
    (df_p.yr == 2011)).distinct()

f.show()