import os    
os.environ['MPLCONFIGDIR'] = "matplot-temp"

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def loadAndProcess():
    print('Loading...')
    # load the data...
    iris = load_iris()
    X = iris.data
    y = iris.target
    featureNames = iris.feature_names
    labelNames = iris.target_names
    
    return labelNames, featureNames, X, y
    
def buildTrainAndTest(X, y):
    print('Building train and test sets...')
    # create the train and test sets for X and y
    # traning has 67% of the rows and test has 33% of the rows...
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 12)

    # print shape of data sets...
    print('Entire set shape= %s' % str(X.shape))
    print('Training set shape= %s' % str(X_train.shape))
    print('Test set shape= %s' % str(X_test.shape))
    
    return X_train, X_test, y_train, y_test

def train(X_train, y_train):
    # train the decision tree algorithm...
    print('Training...')
    treeIris = tree.DecisionTreeClassifier(max_depth = 10, random_state = 32)
    treeIris.fit (X_train, y_train)
    return (treeIris, treeIris.score( X_train , y_train))
    
def test(treeIris, X_test , y_test):
    # test the decision tree algorithm...
    print('Testing...')
    return treeIris.score(X_test , y_test)
    
def predict(treeIris, irisMeasurements):
    print('Precicting...')
    y_classification = treeIris.predict(irisMeasurements)
    return (y_classification[0])

def generatePlots(treeIris, featureNames):
    # now create a bar chart of feature importances
    print('Generating plots...')
    plt.barh( range( len(featureNames) ), treeIris.feature_importances_ , align = 'center' )
    plt.yticks( np.arange( len(featureNames) ), featureNames )
    plt.xlabel( "Feature importance" )
    plt.ylabel( "Feature" )
    plt.ylim ( 0 , len(featureNames) )
    plt.subplots_adjust(left=0.28, right=0.9, top=0.9, bottom=0.1)

    # save the plot...
    if os.path.isfile("featureImportances.png"):
        os.remove("featureImportances.png")
    plt.savefig('featureImportances.png')

    # plot the tree to png...
    plt.figure(dpi=1200)
    tree.plot_tree(treeIris)

    # save the plot...
    if os.path.isfile("treeIris.png"):
        os.remove("treeIris.png")    
    plt.savefig(r"treeIris.png",bbox_inches='tight')
    

def main():
    print("Running Main...")
    labelNames, featureNames, X, y = loadAndProcess()
    
    #print features and labels...
    print("Features %s" % featureNames)
    print("Labels %s" % labelNames)
    
    # print shape...
    print('X shape: %s' % str(X.shape))
    print('y shape: %s' % str(y.shape))
    
    # print data...
    print('first five rows of X= \n%s' % X[0:6, :])
    print('first 150 rows of y= \n%s' % y[0:150])

    X_train, X_test, y_train, y_test = buildTrainAndTest(X, y)
    print("X_train = %s\n" % X_train)
    print("X_test = %s\n" % X_test)
    print("y_train = %s\n" % y_train)
    print("y_test = %s\n" % y_test)

    treeIris, score = train(X_train, y_train)
    print("Score on train data %s\n" % score)

    score = test(treeIris, X_test , y_test)
    print("Score on test data %s\n" % score)

    prediction = predict(treeIris, [[5,  3.5, 1.3, 0.3]])
    flowerType = labelNames[prediction];
    print("Prediction: f([5,  3.5, 1.3, 0.3])->%s\n" % flowerType)

    generatePlots(treeIris, featureNames)
