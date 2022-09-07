from Preprocessing import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = Preprocess()

def RandomforestClassifier(Y_test=y_test):
    Train_Time = 7116.800401687622
    try:
      grid_search = pickle.load(open('RandomForestClassifier.sav', "rb"))
    except (OSError, IOError) as e:
      start_training = time.time()
      result = RandomForestClassifier()
      y_train_int = [int(sample) for sample in y_train]
      n_estimators = [10, 100, 700]
      max_features = ['sqrt', 'log2']
      grid = dict(n_estimators=n_estimators,max_features=max_features)
      cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
      grid_search = GridSearchCV(estimator=result, param_grid=grid, cv=cv, scoring='accuracy',verbose=1000)
      x=grid_search.fit(X_train, y_train_int)
      filename = 'RandomForestClassifier.sav'
      pickle.dump(grid_search, open(filename, 'wb'))
      print("Best Parameters %s" % x.best_params_)
      end_training = time.time()
      measure_trainig = end_training - start_training
      print('Trainig Time : ', measure_trainig)
    start_testing = time.time()
    y_pred = grid_search.predict(X_test)
    end_testing = time.time()
    print('Accuracy: ', metrics.accuracy_score(Y_test.astype(int), y_pred.astype(int)))
    print('Mean Squared Error:', metrics.mean_squared_error(Y_test.astype(int), y_pred.astype(int)))

    measure_testing = end_testing - start_testing

    print('Test Time : ',measure_testing)
    return metrics.accuracy_score(Y_test.astype(int), y_pred.astype(int))




def DecisiontreeClassifier(Y_test=y_test):
     Train_Time = 256.3358
     try:
      grid_search = pickle.load(open('DecisiontreeClassifier.sav', "rb"))
     except (OSError, IOError) as e:
       start_training = time.time()
       clf = DecisionTreeClassifier()
       criterion = ['gini','entropy']
       grid = dict(criterion=criterion)
       cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
       grid_search = GridSearchCV(estimator=clf, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0,verbose=1000)
       x = grid_search.fit(X_train, y_train.astype(int))
       filename = 'DecisiontreeClassifier.sav'
       pickle.dump(grid_search, open(filename, 'wb'))
       print("Best Parameters %s" % x.best_params_)
       end_training = time.time()
       measure_trainig = end_training - start_training
       print('Trainig Time : ', measure_trainig)
     start_testing = time.time()
     y_pred = grid_search.predict(X_test)
     end_testing = time.time()
     print('Accuracy:', metrics.accuracy_score(Y_test.astype(int), y_pred.astype(int)))
     print('Mean Squared Error:', metrics.mean_squared_error(Y_test.astype(int), y_pred.astype(int)))
     measure_testing = end_testing-start_testing
     print('Test Time : ', measure_testing)
     return metrics.accuracy_score(Y_test.astype(int), y_pred.astype(int))

def KNN(Y_test=y_test):
     Train_Time = 1088.724
     try:
        grid_search = pickle.load(open('KNeighborsClassifier.sav', "rb"))
     except (OSError, IOError) as e:
        start_training = time.time()
        clf = KNeighborsClassifier()
        n_neighbors = [5,15,21]
        metric = ['euclidean', 'manhattan', 'minkowski']
        grid = dict(n_neighbors=n_neighbors, metric=metric)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        grid_search = GridSearchCV(estimator=clf, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0,verbose=1000)
        x = grid_search.fit(X_train,y_train.astype(int))
        filename = 'KNeighborsClassifier.sav'
        pickle.dump(grid_search, open(filename, 'wb'))
        print("Best Parameters %s" % x.best_params_)
        end_training = time.time()
        measure_trainig = end_training - start_training
        print('Trainig Time : ', measure_trainig)
     start_testing = time.time()
     y_pred = grid_search.predict(X_test)
     end_testing = time.time()
     print('Accuracy:', metrics.accuracy_score(Y_test.astype(int), y_pred.astype(int)))
     print('Mean Squared Error:', metrics.mean_squared_error(Y_test.astype(int), y_pred.astype(int)))

     measure_testing = end_testing - start_testing

     print('Test Time : ', measure_testing)
     return metrics.accuracy_score(Y_test.astype(int), y_pred.astype(int))
def Logistic_Regression():
    # try:
    #     clf = pickle.load(open('LogisticRegression.sav', "rb"))
    #except (OSError, IOError) as e:
    start_training = time.time()
    clf = LogisticRegression()
    clf.fit(X_train,y_train.astype(int))
    end_training = time.time()
    start_testing = time.time()
    y_pred = grid_search.predict(X_test)
    end_testing = time.time()
    print('Accuracy ', metrics.accuracy_score(y_test.astype(int),y_pred.astype(int)))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test.astype(int), y_pred.astype(int)))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test.astype(int), y_pred.astype(int)))
    measure_trainig = end_training - start_training
    measure_testing = end_testing - start_testing
    print('Trainig Time : ', measure_trainig)
    print('Test Time : ', measure_testing)
    #filename = 'LogisticRegression.sav'
    #pickle.dump(result, open(filename, 'wb'))
#model 1



plt.show()
print("Model: RFC")
Class_1_Accuracy= RandomforestClassifier(y_test)

# model 2
print("Model: Decision Tree")
Class_2_Accuracy= DecisiontreeClassifier(y_test)

#model 3
print("Model: Knn")
Class_3_Accuracy = KNN(y_test)
names = ['RFC','DT','KNN']
values = [Class_1_Accuracy,Class_2_Accuracy,Class_3_Accuracy]
plt.subplot(132)
plt.bar(names, values)
plt.suptitle('Accuracies')
plt.show()
print("DONE")