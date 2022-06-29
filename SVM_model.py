from utils.database import read_file
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import joblib
from os.path import exists

MODEL_SAVE_PATH = 'checkpoint/svm.pth'

def load_data() -> list[np.ndarray]:
    input_data: np.ndarray  # (n_samples, n_feature)
    output_data: np.ndarray  # (n_samples)
    input_data, output_data = read_file()

    X_train, X_test, y_train, y_test = train_test_split(
        input_data, output_data, test_size=0.2, random_state=72
    )

    return X_train, X_test, y_train, y_test


def train_evalute_model(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
):
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    print(model)
    print(classification_report(y_test, y_predict, zero_division=True))
    print("*" * 100)
    return model

def model_search():

    model = svm.SVC()
    paramaters = {
        'C': [1, 10, 25, 50, 100],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'degree': [3, 4],
        'gamma': ['scale', 'auto']
    }
    X_train, X_test, y_train, y_test = load_data()
    grid_search = GridSearchCV(model, param_grid=paramaters)
    grid_search.fit(X_train, y_train)

    print('Cv result', grid_search.cv_results_)
    print('Best estimator', grid_search.best_estimator_)
    print('Best params', grid_search.best_params_)
    print('Best socre', grid_search.best_score_)

def model_evaluate():
    model = svm.SVC(kernel='poly', C=50, degree=4, gamma='scale')
    
    X_train, X_test, y_train, y_test = load_data()
    # print(X_train, X_test)
    model = train_evalute_model(model, X_train, X_test, y_train, y_test)
    save_model(model)

def save_model(model, file_dir=MODEL_SAVE_PATH):
    joblib.dump(model, file_dir)

def load_model(file_dir=MODEL_SAVE_PATH):
    model = joblib.load(file_dir)
    return model

if __name__ == "__main__":
    # model_evaluation()
    if exists(MODEL_SAVE_PATH):
        model = load_model()
        X_train, X_test, y_train, y_test = load_data()
        y_predict = model.predict(X_test)
        print(model)
        print(classification_report(y_test, y_predict, zero_division=True))
        print("*" * 100)
    else:   
        model_evaluate()
