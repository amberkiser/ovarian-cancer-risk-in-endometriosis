import time
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from base_tuning import HyperparameterTuning


start_time = time.time()

# Load Data
with open('X_columns.pkl', 'rb') as f:
    X_columns = pickle.load(f)
with open('selected_columns_rf.pkl', 'rb') as f:
    selected_columns = pickle.load(f)

with open('train_X.pkl', 'rb') as f:
    train_X = pickle.load(f)
with open('train_y.pkl', 'rb') as f:
    train_y = pickle.load(f)

train_X = pd.DataFrame(train_X, columns=X_columns)
train_X = train_X[selected_columns]
train_X = train_X.values


# Random forest tuning
base_clf = RandomForestClassifier(class_weight='balanced')
algorithm = 'rf'
parameters = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 2, 10, 20],
    'criterion': ['gini', 'entropy']
}

tune = HyperparameterTuning(train_X, train_y, base_clf, parameters, 10, algorithm)
tune.tune_parameters()
tune.process_and_save_results()

end_time = time.time()
total_time = end_time - start_time
print('Run time: %f seconds' % total_time)
print('Done with RF!')
