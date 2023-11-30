import logging
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import pickle

from nn_load_data import LoadNNData
from nn_utils import *
from nn_models import *


# hyperparameter tuning for neural network
# hyperparameter(s) include: learning rate, batch size
logging.basicConfig(filename='hyperparameter_tuning_NN.log', level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S %Z')
logging.info('Started script...')

device = find_device()
logging.info('Device: %s' % device)

# Load Data
with open('X_columns_NN.pkl', 'rb') as f:
    X_columns = pickle.load(f)
with open('selected_columns_rf.pkl', 'rb') as f:
    selected_columns = pickle.load(f)

with open('train_X.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('train_y.pkl', 'rb') as f:
    y_train = pickle.load(f)

X_train = pd.DataFrame(X_train, columns=X_columns)
X_train = X_train[selected_columns]
X_train = X_train.values


batch_size_list = [512, 1024, 2048, 4096]
lr_list = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]


number_of_features = len(selected_columns)
input_size = number_of_features
h1_size = 32
h2_size = 16
output_size = 1
epochs = 1000
cv_num = 10


k_folds = StratifiedKFold(n_splits=cv_num)
for batch_size in batch_size_list:
    logging.info('BATCH SIZE: %d' % batch_size)
    for lr in lr_list:
        logging.info('LR: %f' % lr)
        # perform 10-fold cross validation with 5 repeats - 2 for loops
        fold = 0
        for train_index, val_index in k_folds.split(X_train, y_train):
            fold += 1
            logging.info('Fold %d' % fold)
            X_trn, X_val = X_train[train_index], X_train[val_index]
            y_trn, y_val = y_train[train_index], y_train[val_index]
            for rep in range(1, 6):
                logging.info('Fold %d; Rep %d' % (fold, rep))

                # load data
                train_loader = LoadNNData(X_trn, y_trn, batch_size)
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
                for ep in range(1, epochs+1):
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

                    logging.info('EPOCH #%d: TRAIN AUC %f, VAL AUC %f, TRAIN TIME %s, VAL TIME %s' % (ep, train_auc,
                                                                                                      val_auc, train_time,
                                                                                                      val_time))

                    # early stopping check
                    if val_auc > bestValAUC:
                        bestValAUC = val_auc
                        bestValEpoch = ep
                        # best_model = model
                    if ep - bestValEpoch > patience:
                        logging.info('STOPPING...BEST EPOCH #%d' % bestValEpoch)
                        break
