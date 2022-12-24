import numpy as np
import sklearn
import sklearn.ensemble
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sklearn.metrics
import pandas as pd
#from sklearn.inspection import DecisionBoundaryDisplay

def Train_Test(model, X_train, X_test, t_train, t_test): # For Classification
    model.fit(X_train, t_train)
    t_pred = model.predict(X_test)
    print("accuracy_score for "  + ": " + sklearn.metrics.accuracy_score(t_test, t_pred),(np.average(sklearn.model_selection.cross_val_score(model,X_train,t_train,cv=10))))
    print("accuracy_score for "  + ": " + sklearn.metrics.accuracy_score(t_test, t_pred),(np.average(sklearn.model_selection.cross_val_score(model,X_train,t_train,cv=10))))
    print("CM for " + ": " +sklearn.metrics.confusion_matrix(t_test, t_pred))
    return(sklearn.metrics.accuracy_score(t_test, t_pred))

def Train_Test1(model, X_train, X_test, t_train, t_test): #For Regression
    model.fit(X_train, t_train)
    t_pred = model.predict(X_test)
    #print(np.mean((t_pred - t_test) ** 2))
    print("MSE for " +  ": " +sklearn.metrics.mean_squared_error(y_pred=t_pred,y_true=t_test))
    print(model.score(X_test, t_test))
    return model.score(X_test, t_test)

def Test_NN_Regression(data): #Best Score: 0.527485248781123 Best layer_size: [25, 7]
    print("Training the model with MLPRegressor")
    X, t = data[:, :-2], data[:, -2]
    best_score = 0
    best_layer_size = [0,0]
    for i in range(3,4):
        n_train = 1200 + i * 100
        print("training data: " + str(n_train) )
        X_train = X[0:n_train]
        t_train = t[0:n_train]
        X_test = X[n_train:]
        t_test = t[n_train:]
        for j in range (10,35,5):
            for k in range (5, 20, 2):
                print("training the data with NN with hidden-layer-size( " +str(j) +","+str(k) +")" )
                NN_regr = sklearn.neural_network.MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(j, k),random_state=1, max_iter=10000)
                score= Train_Test1(NN_regr, X_train, X_test, t_train, t_test)
                if score > best_score:
                    best_score = score
                    best_layer_size = [j,k]
    print("Best Score: " + str(best_score) + "Best layer_size: " + str(best_layer_size))


def Test_NN_Classification(data): # Best(49,13) accuracy: 0.81  Slightly more time-consuming than RF
    print("Training the model with MLPClassifier")
    X, t = data[:, :-2], data[:, -1]
    best_score = 0
    best_layer_size = [0, 0]
    for i in range(3,4):
        n_train = 1200 + i * 100
        print("training data: " + str(n_train) )
        X_train = X[0:n_train]
        t_train = t[0:n_train]
        X_test = X[n_train:]
        t_test = t[n_train:]
        for j in range (10,50):
            for k in range (5, 20):
                print("training the data with NN with hidden-layer-size( " +str(j) +","+str(k) +")" )
                NN_class = sklearn.neural_network.MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(j, k),random_state=1, max_iter=10000)
                start_time = time.time()
                score = Train_Test(NN_class, X_train, X_test, t_train, t_test)
                end_time = time.time()
                print("Time used " + str(end_time - start_time))
                if score > best_score:
                    best_score = score
                    best_layer_size = [j, k]
    print("Best score "+ str(best_score)+ "Best layer_size: " + str(best_layer_size) )


def Test_RF_Classification(data): # Best(30,11) accuracy: 0.798
    print("Training the model with RandomForest")
    X, t = data[:, :-2], data[:, -1]
    best_score = 0
    best_n_estimator = 0
    best_n_maxDepth = 0
    best_time = 0
    for i in range(3,4):
        n_train = 1200 + i * 100
        print("training data: " + str(n_train) )
        X_train = X[0:n_train]
        t_train = t[0:n_train]
        X_test = X[n_train:]
        t_test = t[n_train:]
        for j in range (10,100,10):
            for k in range (1, 20, 2):
                print("training the data with NN with hidden-layer-size( " +str(j) +","+str(k) +")" )
                model = sklearn.ensemble.RandomForestClassifier(n_estimators=j, max_depth=k, random_state=0)
                start_time = time.time()
                score = Train_Test(model, X_train, X_test, t_train, t_test)
                end_time = time.time()
                print("Time used " + str(end_time - start_time))
                if score > best_score:
                    best_score = score
                    best_n_estimator = j
                    best_n_maxDepth = k
                    best_time = end_time - start_time
    print("Best score " + str(best_score) + "Best n_estimator: " + str(best_n_estimator) + "Best max_Depth " + str(best_n_maxDepth) + "Best Time " + str(best_time))

def loadData(file_name):
    with open(f"../data/{file_name}") as file:
        data=pd.read_csv(file)
        return data.to_numpy()

# data = np.loadtxt('../data/augmented_data.csv', delimiter=',', skiprows=1, usecols = range(5,17))
# X,t = data[:, :-2], data[:, -1]
# X_train = X[0:1500]
# t_train = t[0:1500]
# X_test = X[1500:]
# t_test = t[1500:]

# X1,t1 = data[:, :-2], data[:, -2]
# X1_train = X1[0:1500]
# t1_train = t1[0:1500]
# X1_test = X1[1500:]
# t1_test = t1[1500:]
dataset= loadData("augmented_data.csv")
y=dataset[:,1:3]
X=dataset[:,3:]
#X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,random_state=0)
X_train=X[0:1500,:]
X_test=X[1500:,:]
X1_train=X_train
X1_test=X_test

t_train=y[0:1500,1]
t_test=y[1500:,1]
t1_train=y[0:1500:,0]
t1_test=y[1500:,0]
data=dataset

regr = sklearn.linear_model.LinearRegression()
SGD = sklearn.linear_model.SGDRegressor();
ada = sklearn.ensemble.AdaBoostRegressor(n_estimators = 500)

svm = sklearn.svm.SVC()
RF = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)
NN = sklearn.neural_network.MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(50, 15), random_state=1,max_iter = 10000)

#LR = sklearn.linear_model.LogisticRegression(random_state=0, solver='newton-cholesky', multi_class='ovr')
LR = sklearn.linear_model.LogisticRegression(random_state=0, solver='newton-cg', multi_class='ovr')
NN_regr = sklearn.neural_network.MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 8), random_state=1,max_iter = 10000)
Train_Test(svm,X_train, X_test, t_train, t_test)
Train_Test(RF,X_train, X_test, t_train, t_test)
Train_Test(LR,X_train, X_test, t_train, t_test)
Train_Test(NN,X_train, X_test, t_train, t_test)
Train_Test1(NN_regr,X1_train, X1_test, t1_train, t1_test)
Train_Test1(ada,X1_train, X1_test, t1_train, t1_test)
