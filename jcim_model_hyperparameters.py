import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


datos = pd.read_csv('file.csv')

#Selected by using Gini decrease method
X=datos[['ROBB760111','GEOR030106','GEOR030105','GEOR030104','WILM950103','PRAM820101','GEOR030101','TANS770109','RACS770101','TANS770106']]

y = datos['PRED']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix


model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                                bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0,
                                max_samples=None).fit(X_train, y_train)

# Save model
import joblib
joblib.dump(model, 'model.pkl')

#Cross-validation a fold =5
def confusion_matrix_scorer(clf, X_train, y_train):
        y_pred = clf.predict(X_train)
        cm = confusion_matrix(y_train, y_pred)
        return {'tn': cm[0, 0], 'fp': cm[0, 1],
                'fn': cm[1, 0], 'tp': cm[1, 1]}
        
cv_results = cross_validate(model, X_train, y_train, cv=5,
                            scoring=confusion_matrix_scorer)

# Getting the test set true positive scores
TP = cv_results['test_tp'].mean()

# Getting the test set false negative scores
FN = cv_results['test_fn'].mean()

# Getting the test set false positive scores
FP = cv_results['test_fp'].mean()

# Getting the test set true negative scores
TN = cv_results['test_tn'].mean()

####TRAINING###
acurracy = (TP+TN) / (TP+TN+FP+FN)
F1_score = 2*TP / ((2*TP) + (FP + FN))
precision = TP / (TP + FP)
specificity = TN / (FP + TN)
sensitivity_recall = TP / (TP + FN)
import math 
MCC = ((TP*TN) - (FP*FN)) / math.sqrt(((TP+FP)*(TP+FN))*((TN+FP)*(TN+FN)))

print("Accuracy: ", acurracy)
print("F1_score: ", F1_score)
print("Precision: ", precision)
print("Specificity: ", specificity)
print("Sensitivity/Recall: ", sensitivity_recall)
print("MCC: ", MCC)


####TESTING###
from sklearn.metrics import classification_report
pred_test=model.predict(X_test)

conf = confusion_matrix(y_test, pred_test)
TP = conf[1, 1]
FP = conf[0, 1]
TN = conf[0, 0]
FN = conf[1, 0]

acurracy = (TP+TN) / (TP+TN+FP+FN)
F1_score = 2*TP / ((2*TP) + (FP + FN))
precision = TP / (TP + FP)
specificity = TN / (FP + TN)
sensitivity_recall = TP / (TP + FN)

import math 
MCC = ((TP*TN) - (FP*FN)) / math.sqrt(((TP+FP)*(TP+FN))*((TN+FP)*(TN+FN)))

print("Accuracy: ", acurracy)
print("F1_score: ", F1_score)
print("Precision: ", precision)
print("Specificity: ", specificity)
print("Sensitivity/Recall: ", sensitivity_recall)
print("MCC: ", MCC)