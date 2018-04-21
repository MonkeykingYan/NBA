# README

### Team Classification Part:
* Step1: Run playerCluster.py, the output file is playersClusters.csv in workspace.
* Step2: Run Lineup.py, the output file is teamClusters2011.csv (2011 - 2016) in workspace.
* Step3: Run TeamClassifier.py, the output file is teamClusters.csv  in workspace.

### Team Structure Discovery Part:
* Frequent Itemsets: Run FrequentItemSets.py
* PageRank: Run network.py

### Configuration
* In TeamClassifier.py, **Pycluster package** is needed for Damerau–Levenshtein distance calculation
* In network.py, you have to install graphframes package for pyspark, adding <br>
 **os.environ["PYSPARK_SUBMIT_ARGS"] = (
     "--packages graphframes:graphframes:0.3.0-spark2.0-s_2.11 pyspark-shell"
 )**
* You need to change folder and playerids to get MIA2013 or GSW2016 results. 
* You need R language to get the passing network visualization, running plot_network.R need under MIA2013 or GSW2016 folders.
 and **library(d3Network)** and **Library(DT)** are needed
