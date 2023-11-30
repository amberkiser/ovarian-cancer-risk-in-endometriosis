import logging
import pickle
import pandas as pd
import copy

from nn_load_data import LoadNNData
from nn_utils import *
from nn_models import *

# Train model
debug = False
logging.basicConfig(filename='training_NN.log', level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S %Z')
logging.info('SCRIPT STARTED...')
device = find_device()
logging.info('Device: %s' % device)

# Load Data
with open('X_columns.pkl', 'rb') as f:
    X_columns = pickle.load(f)
with open('selected_columns_rf.pkl', 'rb') as f:
    selected_columns = pickle.load(f)

with open('train_X.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('train_y.pkl', 'rb') as f:
    y_train = pickle.load(f)

with open('val_X_NN.pkl', 'rb') as f:
    X_val = pickle.load(f)
with open('val_y_NN.pkl', 'rb') as f:
    y_val = pickle.load(f)

X_train = pd.DataFrame(X_train, columns=X_columns)
X_train = X_train[selected_columns]
X_train = X_train.values

X_val = pd.DataFrame(X_val, columns=X_columns)
X_val = X_val[selected_columns]
X_val = X_val.values

batch_size = 2048
lr = 0.005

number_of_features = len(selected_columns)
input_size = number_of_features
h1_size = 32
h2_size = 16
output_size = 1
epochs = 1000


# load data
train_loader = LoadNNData(X_train, y_train, batch_size)
val_loader = LoadNNData(X_val, y_val, batch_size)

# instantiate the model
model = NeuralNetModule(input_size, h1_size, h2_size, output_size)
model = model.double()
model.to(device)

# set training variables
pos_weight = train_loader.find_pos_weight()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
sigmoid = nn.Sigmoid()
bestValAUC = 0.0
bestValEpoch = 0
patience = 10

# epoch loop
for ep in range(1, epochs + 1):
    logging.info('Epoch %d' % ep)
    start = time.time()

    # batch loop
    n_iter = 0
    for inputs, labels in train_loader.loader:
        n_iter += 1
        logging.info('Batch %d' % n_iter)
        train_batch(model, inputs, labels, device, criterion, optimizer)

    train_time = time_since(start)

    val_start = time.time()
    train_auc, val_auc = validate_nn(model, device, train_loader, val_loader, sigmoid)
    val_time = time_since(val_start)

    logging.info('EPOCH #%d: TRAIN AUC %f, VAL AUC %f, TRAIN TIME %s, VAL TIME %s' % (ep, train_auc, val_auc,
                                                                                      train_time, val_time))

    # early stopping check
    if val_auc > bestValAUC:
        bestValAUC = val_auc
        bestValEpoch = ep
        best_model = copy.deepcopy(model)
    if ep - bestValEpoch > patience:
        logging.info('STOPPING...BEST EPOCH #%d, BEST VAL AUC %f' % (bestValEpoch, bestValAUC))
        break

# save model & parameters
torch.save(best_model.state_dict(), 'NN_model.st')
logging.info('SCRIPT DONE')
