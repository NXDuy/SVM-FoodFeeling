from sklearn.feature_selection import SelectKBest, f_classif
from utils.database import read_file

input_data, output_data = read_file()
X_train = input_data.to_numpy()
y_train = output_data.to_numpy().reshape(-1)

select_module = SelectKBest(score_func=f_classif, k='all')
select_module.fit(X_train, y_train)

feature_name = input_data.columns
feature_score = list(zip(feature_name, select_module.scores_))
feature_score.sort(key=lambda tup: tup[1], reverse=True)

print(feature_score)


