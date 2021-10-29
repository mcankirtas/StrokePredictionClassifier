import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA

veriler = pd.read_csv("healthcare-dataset-stroke-data.csv")

veriler.drop(["id"], inplace=True, axis=1)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in veriler.columns[0:] :
    veriler[i] = le.fit_transform(veriler[i])
    

x=veriler.iloc[:,0:-1]
y=veriler.iloc[:,-1:]
"""
corelation_matrix = x.corr()

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)    

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
"""
from sklearn.svm import SVC
"""
svr_reg = SVC(kernel="poly")
svr_reg.fit(X_train, y_train.values.ravel())

predicted_svr=svr_reg.predict(X_test)

cm_svm = confusion_matrix(y_test, predicted_svr)
acc_svm = accuracy_score(y_test, predicted_svr)
"""
#decision tree
from sklearn.tree import DecisionTreeClassifier
"""
r_dt = DecisionTreeClassifier(criterion="gini", splitter= "best", max_features="auto", max_depth=3)

r_dt.fit(x_train,y_train)

r_dt_predicted = r_dt.predict(x_test)

cm_dt = confusion_matrix(y_test, r_dt_predicted)

acc_dt = accuracy_score(y_test, r_dt_predicted)

from sklearn import tree

fig = plt.figure(figsize=(25,20))
tree.plot_tree(r_dt, filled=True)
"""
#random forest
from sklearn.ensemble import RandomForestClassifier
"""
rf_reg = RandomForestClassifier(n_estimators=40, criterion= "entropy", max_depth=3)
                                
rf_reg.fit(X_train,y_train.values.ravel())

predicted_rf = rf_reg.predict(X_test)

cm_rf = confusion_matrix(y_test, predicted_rf)
acc_rf = accuracy_score(y_test, predicted_rf)
"""
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X, y = oversample.fit_resample(x, y)

steps = [('scaler', StandardScaler()), ('Random Forest', RandomForestClassifier(n_estimators=40, criterion= "entropy", max_depth=14))]

from sklearn.pipeline import Pipeline
pipeline = Pipeline(steps)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.33)

model = pipeline.fit(X_train1, y_train1)
"""
print("Train Score Accuracy :", model.score(X_train1, y_train1))
print("Test Score Accuracy :", model.score(X_test1, y_test1))
"""
smote_rf_predict = model.predict(X_test1)
acc_smote_rf = accuracy_score(y_test1, smote_rf_predict)
cm_smote_rf = confusion_matrix(y_test1, smote_rf_predict)

#trying dt
steps_dt = [('scaler', StandardScaler()), ('Decision Tree Classifier', DecisionTreeClassifier(criterion="entropy", splitter= "best", max_features="auto", max_depth=18))]

from sklearn.pipeline import Pipeline
pipeline_dt = Pipeline(steps_dt)

model_dt = pipeline_dt.fit(X_train1, y_train1)

smote_dt_predict = model_dt.predict(X_test1)
acc_smote_dt = accuracy_score(y_test1, smote_dt_predict)
cm_smote_dt = confusion_matrix(y_test1, smote_dt_predict)

print("Train Score Accuracy :", model_dt.score(X_train1, y_train1))
print("Test Score Accuracy :", model_dt.score(X_test1, y_test1))

#trying svc
steps_svc = [('scaler', StandardScaler()), ('Support Vector Classifier', SVC(kernel="rbf"))]

from sklearn.pipeline import Pipeline
pipeline_svc = Pipeline(steps_svc)

model_svc = pipeline_svc.fit(X_train1, y_train1)

smote_svc_predict = model_svc.predict(X_test1)
acc_smote_svc = accuracy_score(y_test1, smote_svc_predict)
cm_smote_svc = confusion_matrix(y_test1, smote_svc_predict)

