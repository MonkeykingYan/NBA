import numpy as np
from sklearn.decomposition import PCA
from pyspark.sql.types import *
import numpy as np
import sklearn.cluster
from nltk.metrics import distance

import Pycluster as PC

words = ['apple', 'Doppler', 'applaud', 'append', 'barker',
         'baker', 'bismark', 'park', 'stake', 'steak', 'teak', 'sleek']

dist = [distance.edit_distance(words[i], words[j])
        for i in range(1, len(words))
        for j in range(0, i)]

labels, error, nfound = PC.kmedoids(dist, nclusters=3)#kmedoids(dist, nclusters=3)
cluster = dict()
for word, label in zip(words, labels):
    cluster.setdefault(label, []).append(word)
for label, grp in cluster.items():
    print(grp)

data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
data = np.array(data)
np.savetxt("foo.csv", data, delimiter=",")


schema = StructType([StructField('Name', StringType(), True),
                     StructField('DateTime', TimestampType(), True),
                     StructField('Age', IntegerType(), True)])
print(data)
pca = PCA(n_components=3)
pca.fit(data)
X = pca.transform(data)
print(X)