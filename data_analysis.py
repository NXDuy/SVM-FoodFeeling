import matplotlib.pyplot as plt
from sklearn import feature_selection
from utils.database import read_file
import seaborn as sns
import pandas as pd
import math
from sklearn.feature_selection import SelectKBest, f_classif
# database = read_file()

# fig, axes = plt.subplots(1, 2, figsize=(15, 10))
# sns.histplot(ax=axes[0] ,data=database["viewer feeling of youtuber's style "])
# axes[0].set_title("Start time without normalize")
# # plt.show()
# start_time = database['start time'].to_numpy().reshape(-1)
# start_time = list(math.sqrt(time) for time in start_time)

# sns.histplot(ax=axes[1], data=start_time)
# axes[1].set_title("Start time to normal distibution")
# plt.show()

# print(database.describe())
features, labels = read_file() 
features_names = features.columns

features = features.to_numpy()
labels = labels.to_numpy().reshape(-1)
feature_selection = SelectKBest(score_func=f_classif, k=10)
feature_selection.fit(features, labels)

features_scores = list(zip(features_names, feature_selection.scores_))
features_scores.sort(key=lambda tup: tup[1])
features_pvalue = list(zip(features_names, feature_selection.pvalues_))

print(features_scores)
print(features_pvalue)

features_labels = [tup[0] for tup in features_scores]
scores_list = [tup[1] for tup in features_scores]
plt.barh(features_labels, scores_list)
plt.xlabel('Scores')
plt.ylabel('Feature name')
plt.show()

# print(features.describe())



