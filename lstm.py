from numpy import mean
from numpy import std
from numpy import dstack
from numpy import array
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers.embeddings import Embedding
import numpy as np


def evaluate_model():
    verbose, epochs, batch_size = 0, 200, 480
    names = ['Window No', 'Avg of x', 'Avg of y', 'Avg of z', 'Min of x', 'Min of y', 'Max of x', 'Max of y',
             'Max of z',
             'Variance', 'Class']
  #  dataset = read_csv("walking_down_stairs.csv", names=names, encoding='ascii')
    trainData = read_csv("train.csv", names=names, encoding='utf8')
    testData = read_csv("test.csv", names=names, encoding='utf8')
    #print(dataset.head())
  #  x = dataset.drop('Class', axis=1)
  #  y = dataset.Class
    X_train = trainData.drop('Class', axis=1)
    y_train = trainData.Class
    X_test = testData.drop('Class', axis=1)
    y_test = testData.Class

    y_train = np.unique(y_train)
    y_test = np.unique(y_test)



   # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
    #print(y_train.shape)
    #y_test = to_categorical(y_test)
    X_train = X_train.values.reshape((6, 5, 10))
    X_test = X_test.values.reshape((6, 5, 10))

  #  y_train = y_train.values.reshape(6, 5)
   # y_test = y_test.values.reshape(6, 5)
    #print(y_train.shape)
    #one hot encode y
    y_train = to_categorical(y_train, num_classes=None, dtype='float32')

    #y_train = y_train.reshape(6, 6)
    y_test = to_categorical(y_test, num_classes=None, dtype='float32')
    #y_test = y_test.reshape(6, 6)


    print("x train:", X_train)
    print("y train:", y_train)
    print("x test:", X_test)
    print("y test:", y_test)


    model = Sequential()
    #model.add(LSTM(100, input_shape=(5, 10)))
    #model.add(Dropout(0.5))
    #model.add(Dense(100, activation='relu'))
    #model.add(Dense(6, activation='softmax'))
    model.add(LSTM(10, input_shape=(5, 10)))
    model.add(Dropout(0.5))

    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    # fit network
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    pred = model.predict(X_test)
    predict_classes = np.argmax(pred, axis=1)
    print("Predicted Classes: {}", predict_classes)
    print(pred)
  #  print(X_test)
   # print(y_test)
  #  print(y_train)
   # print(X_train)

    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)

    print(accuracy)
    return accuracy

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# run an experiment
def run_experiment2(repeats=4):
    # load data
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model()
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


#run
evaluate_model()