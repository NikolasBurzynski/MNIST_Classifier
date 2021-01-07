import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from matplotlib import pyplot

def main():
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train(train_data, train_labels)


    
def train(data, labels):
    N = len(data) 
    HL_NODES = 10
    PIXELS = 784
    DIGITS = 10
    REG = 1e-3
    STEP_SIZE = .01
    images = np.asarray(data)
    flat_images = np.empty((N,PIXELS))
    for i, image in enumerate(images):
        flat_images[i] = images[i].flatten()
        flat_images[i].reshape(784,1)
        flat_images[i] = flat_images[i] / 255
    W1 = 0.01 * np.random.randn(HL_NODES,PIXELS)
    B1 = np.zeros(HL_NODES)
    W2 = 0.01 * np.random.randn(DIGITS,HL_NODES)
    B2 = np.zeros(DIGITS)
    correct = 0
    for j in range(4):
        for i in range(N):
            HL = np.maximum(0, W1.dot(flat_images[i])+B1)
            SCORE = W2.dot(HL) + B2
            #Loss Calculation
            exp_score = np.exp(SCORE)
            norm_score = exp_score / np.sum(exp_score)

            correct_digit = labels[i]
            correct_score = norm_score[correct_digit]

            pred_digit = np.argmax(norm_score)

            if(pred_digit == correct_digit):
                correct += 1

            
            data_loss = -np.log(correct_score)
            reg_loss = .5*REG*np.sum(W1*W1) + .5*REG*np.sum(W2*W2)
            total_loss = data_loss + reg_loss
            print("Current iteration: {} Correct Digit: {} Confidence: {}".format(i,correct_digit, correct_score))
            #backPropogation
            dscores = norm_score
            dscores[correct_digit] -= 1

            dW2 = np.outer(dscores, HL) 
    
            dB2 = np.sum(dscores, axis = 0, keepdims=True)
            dHL = W2.T.dot(dscores)
            #relu activation backprop
            dHL[HL <= 0] = 0

            dW1 = np.outer(dHL, flat_images[i])
            dB1 = np.sum(dHL, axis = 0, keepdims=True)

            dW2 += REG * W2
            dW1 += REG * W1

            W1 += -(STEP_SIZE * dW1)
            B1 += -(STEP_SIZE * dB1)
            W2 += -(STEP_SIZE * dW2)
            B2 += -(STEP_SIZE * dB2)

    print("I got {} correct. This is {} accuracy".format(correct, correct/N))
    trained_NN = np.array([W1.flatten(),B1.flatten(),W2.flatten(),B2.flatten()], dtype = object)
    file = open("TNN", "wb")
    np.save(file, trained_NN, allow_pickle=True)
    file.close

def findMax(scores):
    index = 0
    record = 0  
    for i, score in enumerate(scores):
        if score >= record:
            record = score
            index = i
    return index
        

if __name__ == "__main__":
    main()