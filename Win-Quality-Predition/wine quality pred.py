import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

data = pd.read_csv('winequality-red.csv')
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)

bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names)
label_quality = LabelEncoder()
data['quality'] = label_quality.fit_transform(data['quality'])
data['quality'].value_counts()
sns.countplot(data['quality'])

X = data.drop('quality', axis = 1)
y = data['quality']

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

## Random forest

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

print(classification_report(y_test, pred_rfc))

rfc_eval = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)
rfc_eval.mean()

## Support Vector Classification(SVC)

svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)

print(classification_report(y_test, pred_svc))

svc_eval = cross_val_score(estimator = svc, X = X_train, y = y_train, cv = 10)
svc_eval.mean()

## KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)

print(classification_report(y_test, pred_knn))

knn_eval = cross_val_score(estimator = knn, X = X_train, y = y_train, cv = 10)
knn_eval.mean()

##naive_bayes

from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
pred_naive_bayes = naive_bayes.predict(X_test)

print(classification_report(y_test, pred_naive_bayes))
   

                                  ###### The End ######





