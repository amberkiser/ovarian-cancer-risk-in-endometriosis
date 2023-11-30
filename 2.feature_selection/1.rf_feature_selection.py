import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


# Load data
with open('X_columns.pkl', 'rb') as f:
    X_columns = pickle.load(f)
with open('train_X.pkl', 'rb') as f:
    X = pickle.load(f)
with open('train_y.pkl', 'rb') as f:
    y = pickle.load(f)

X = pd.DataFrame(X, columns=X_columns)

cv_num = 10
repeats = 5
k_folds = StratifiedKFold(n_splits=cv_num)
importance_data = pd.DataFrame()

for r in range(repeats):
    fold = 0
    for train_index, val_index in k_folds.split(X, y):
        fold += 1
        X_train = X.iloc[train_index]
        y_train = y[train_index]

        rf_clf = RandomForestClassifier(n_estimators=200,
                                        max_depth=20,
                                        criterion='gini',
                                        class_weight='balanced')
        rf_clf.fit(X_train, y_train)

        importance_data = pd.concat([importance_data,
                                     pd.DataFrame({'feature': X.columns,
                                                   'rf_importance': rf_clf.feature_importances_,
                                                   'fold': fold,
                                                   'repeat': r})])

importance_data.to_csv('rf_feature_importances.csv', index=False)
