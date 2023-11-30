library(pROC)

# Read in data
hispanic <- read.csv('hispanic.csv')
nonhispanic <- read.csv('nonhispanic.csv')
white <- read.csv('white.csv')
poc <- read.csv('poc.csv')

# Compare AUROC
hispanic_roc <- roc(data = hispanic, response = 'ovarian_ca', predictor = 'y_prob', auc = TRUE, ci = TRUE, plot = TRUE, col = 'blue', print.auc = TRUE)
print(hispanic_roc$auc)
print(hispanic_roc$ci)
nonhispanic_roc <- roc(data = nonhispanic, response = 'ovarian_ca', predictor = 'y_prob', auc = TRUE, ci = TRUE, plot = TRUE, add = TRUE, col = 'red', print.auc = TRUE, print.auc.y = 0.4)
print(nonhispanic_roc$auc)
print(nonhispanic_roc$ci)
roc.test(hispanic_roc, nonhispanic_roc, method='delong', alternative='two.sided', paired=FALSE)

white_roc <- roc(data = white, response = 'ovarian_ca', predictor = 'y_prob', auc = TRUE, ci = TRUE, plot = TRUE, col = 'blue', print.auc = TRUE)
print(white_roc$auc)
print(white_roc$ci)
poc_roc <- roc(data = poc, response = 'ovarian_ca', predictor = 'y_prob', auc = TRUE, ci = TRUE, plot = TRUE, add = TRUE, col = 'red', print.auc = TRUE, print.auc.y = 0.4)
print(poc_roc$auc)
print(poc_roc$ci)
roc.test(white_roc, poc_roc, method='delong', alternative='two.sided', paired=FALSE)
