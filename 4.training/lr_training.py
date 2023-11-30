import pandas as pd
import pickle
import joblib
from sklearn.linear_model import LogisticRegression


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
lr_clf = LogisticRegression(max_iter=50000,
                            solver='liblinear',
                            penalty='l1',
                            class_weight='balanced',
                            C=0.1)

# Fit model
lr_clf.fit(train_X, train_y)

# Save model
joblib.dump(lr_clf, 'lr_model.joblib')
