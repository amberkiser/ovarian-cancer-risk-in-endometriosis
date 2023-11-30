import pandas as pd
import pickle
import joblib
import shap
import matplotlib.pyplot as plt


plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


# Load data
with open('X_columns.pkl', 'rb') as f:
    X_columns = pickle.load(f)
with open('selected_columns_rf.pkl', 'rb') as f:
    selected_columns = pickle.load(f)
with open('test_X.pkl', 'rb') as f:
    test_X = pickle.load(f)

feature_names = pd.read_csv('feature_names.csv', index_col='column')

test_X = pd.DataFrame(test_X, columns=X_columns)
test_X = test_X[selected_columns]
test_X = test_X.rename(columns=feature_names.to_dict()['name'])


# Load models
rf_clf = joblib.load('rf_model.joblib')


# RF SHAP
explainer_tst = shap.TreeExplainer(rf_clf)
shap_values_test = explainer_tst(test_X)
shap.plots.bar(shap_values_test[:, :, 1], show=False, max_display=30)
plt.savefig('shap_values_test.png', bbox_inches='tight')
plt.close()

shap.plots.beeswarm(shap_values_test[:, :, 1], show=False, max_display=30)
plt.savefig('shap_beeswarm_test.png', bbox_inches='tight')
plt.close()

print('Done with explainability!')
