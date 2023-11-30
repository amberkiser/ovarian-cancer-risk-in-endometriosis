import pandas as pd
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier


# Load data
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

# Instantiate model
rf_clf = RandomForestClassifier(n_estimators=200,
                                max_depth=20,
                                criterion='gini',
                                class_weight='balanced')

# Fit model
rf_clf.fit(train_X, train_y)

# Save model
joblib.dump(rf_clf, 'rf_model.joblib')
