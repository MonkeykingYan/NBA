from pyspark.sql.functions import mean
def Normalize(df, columns):
    selectExpr = []
    for column in columns:
       average = df.agg(mean(df[column]).alias("mean")).collect()[0]["mean"]
       selectExpr.append(df[column] - average)
    return df.select(selectExpr)