from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import random

(datasetForTrain),(datasetForTest) = mnist.load_data()
x_train, y_train = datasetForTrain[0], datasetForTrain[1]
x_test, y_test = datasetForTest[0], datasetForTest[1]

x_train = x_train / 255
x_test = x_test / 255

def to_categorical(arrayy, lenght):
    final = []
    for ind in arrayy:
        arr=[]
        for i in range(lenght):
            if i != ind:
                arr.append(0)
            else:
                arr.append(1)
        final.append(arr)
    return final

def to_1d_array(array):
    arr = []
    for i in array:
        for j in i:
            arr.append(j)
    return arr

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)



loss_arr = []

# CONSTS
INPUT_DIM = 784
OUTPUT_DIM = 10
SECOND_DIM = 128
EPOCH = 5

ALPHA = 0.001


def relu(t):
    return np.maximum(t, 0)

def softmax(t):
    out = np.exp(t)
    return out/np.sum(out)

def categorical_crossentropy(z, y):

    return -np.sum(y*np.log(z))


def relu_deriv(t):
    return(t >= 0).astype(float)

W1 = np.random.randn(INPUT_DIM, SECOND_DIM)
b1 = np.random.randn(1, SECOND_DIM)
W2 = np.random.randn(SECOND_DIM, OUTPUT_DIM)
b2 = np.random.randn(1, OUTPUT_DIM)

for j in range(EPOCH):
    print("EPOCH NUMBER " + str(j))
    for i in range(len(x_train[:15000])):
        print("itteration number " + str(i))
        x, y = x_train[i], y_train[i]
        # Forward
        t1 = to_1d_array(x) @ W1 + b1
        h1 = relu(t1)
        t2 = h1 @ W2 + b2
        z = softmax(t2)
        E = categorical_crossentropy(z, y_train_cat[i])
        # Backward

        dE_dt2 = z - y_train_cat[i]
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = dE_dt2
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dW1 = np.array([to_1d_array(x)]).T @ dE_dt1
        dE_db1 = dE_dt1

        # Update
        W1 = W1 - ALPHA * dE_dW1
        b1 = b1 - ALPHA * dE_db1
        W2 = W2 - ALPHA * dE_dW2
        b2 = b2 - ALPHA * dE_db2

def predict(x):
    t1 = to_1d_array(x) @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax(t2)
    return z

print(np.argmax(predict(x_test[0])))
# print(np.argmax(predict(x_test[1])))
# print(np.argmax(predict(x_test[2])))
# print(np.argmax(predict(x_test[-1])))
