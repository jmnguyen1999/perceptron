#-------------------------------------------------------------------------
# AUTHOR: Josephine Nguyen
# FILENAME: perceptron.py
# SPECIFICATION: This program reads the training data in file optdigits.tra to create a classifer using the neural network method Perceptron and Multi-Level Perceptron, and tests + compares their accuracies using the test data in file optdigits.tes.
# FOR: CS 4210- Assignment #4
# TIME SPENT: 40 min
#-----------------------------------------------------------*/
#
#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier    #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None)     #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64]     #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]      #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None)     #reading the data by using Pandas library
X_test = np.array(df.values)[:,:64]     #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]      #getting the last field to form the class label for test



maxPAccuracy = 0
maxMlpAccuracy = 0

for w in n: #iterates over n
    for b in r: #iterates over r
        for a in range(2): #iterates over the algorithms
            #Create a Neural Network classifier
            if a == 0:
                clf = Perceptron(eta0=w, random_state=b, max_iter=1000) #eta0 = learning rate, random_state = shuffle the training data
            else:
                clf = MLPClassifier(activation='logistic', learning_rate_init=w, hidden_layer_sizes=(25,), random_state=b, max_iter=1000) #learning_rate_init = learning rate, hidden_layer_sizes = number of neurons in the ith hidden layer, random_state = shuffle the training data

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            amountCorrect = 0
            for(x_testSample, y_testSample) in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])
                if prediction == y_testSample:
                    amountCorrect = amountCorrect + 1

            # check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy and print it together with the network hyperparameters
            #perceptron algorithm:
            if a == 0:
                currPAccuracy = amountCorrect / len(X_test)
                if currPAccuracy > maxPAccuracy:
                    maxPAccuracy = currPAccuracy
                    print("Highest Perceptron accuracy so far: " + str(maxPAccuracy) + ", Parameters: learning rate=" + str(w) + ", random_state=" + str(b))
            else:
                currMlpAccuracy = amountCorrect / len(X_test)
                if currMlpAccuracy > maxMlpAccuracy:
                    maxMlpAccuracy = currMlpAccuracy
                    print("Highest MLP accuracy so far: " + str(maxMlpAccuracy) + ", Parameters: learning rate=" + str(w) + ", random_state=" + str(b))

