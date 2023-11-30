import pandas as pd
import pickle
import joblib
from sklearn.naive_bayes import ComplementNB


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
nb_clf = ComplementNB(norm=True)

# Fit model
nb_clf.fit(train_X, train_y)

# Save model
joblib.dump(nb_clf, 'nb_model.joblib')
