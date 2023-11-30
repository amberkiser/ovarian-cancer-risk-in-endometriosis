import time
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from base_tuning import HyperparameterTuning


start_time = time.time()

# Load Data
with open('X_columns.pkl', 'rb') as f:
    X_columns = pickle.load(f)
with open('selected_columns.pkl', 'rb') as f:
    selected_columns = pickle.load(f)

with open('train_X.pkl', 'rb') as f:
    train_X = pickle.load(f)
with open('train_y.pkl', 'rb') as f:
    train_y = pickle.load(f)

train_X = pd.DataFrame(train_X, columns=X_columns)
train_X = train_X[selected_columns]
train_X = train_X.values


# Logistic regression tuning
base_clf = LogisticRegression(max_iter=5000, class_weight='balanced')
algorithm = 'lr'
parameters = {
    'penalty': ['l1', 'l2'],
    'solver': ['saga', 'liblinear'],
    'C': [0.01, 0.1, 0.25, 0.50, 1, 2]
}

tune = HyperparameterTuning(train_X, train_y, base_clf, parameters, 10, algorithm)
tune.tune_parameters()
tune.process_and_save_results()

end_time = time.time()
total_time = end_time - start_time
print('Run time: %f seconds' % total_time)
print('Done with LR!')
