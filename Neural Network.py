import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from matplotlib import pyplot

def main():
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train(train_data, train_labels)

    
def train(data, labels):
    N = len(data) 
    PIXELS = 784
    DIGITS = 10
    REG = 1e-3
    STEP_SIZE = 1
    images = np.asarray(data)
    flat_images = np.empty((N,PIXELS))
    for i, image in enumerate(images):
        flat_images[i] = images[i].flatten()
    #initial W1 and W2
    W1 = 0.001 * np.random.randn(PIXELS,DIGITS)
    B1 = np.zeros(DIGITS)
    #lets train some shit ey?
    
    for j in range(1):
        for i in range(20):
            print("FLAT:")
            print(flat_images[i])
            print("W1:")
            print(W1)
            # print("Split")
            # print(B1)
            # print("Split")
            decision = np.dot(flat_images[i], W1) + B1
            #print(decision)
            #Calculate the score of the decision via softmax classifier
            #print(decision)
            decision = np.exp(decision) 
            print(decision)
            decision_probs = decision / np.sum(decision)
            loss = -np.log(decision_probs[labels[0]])
            reg_loss = .5*REG*np.sum(W1*W1)
            total_loss = loss + reg_loss
            print("Prediction "+ str(i) + " run " + str(j) + " loss: " + str(total_loss))
            #Now we get to do the back propogation shenanegans
            
            dscores = decision_probs
            dscores = dscores - 1
            dscores /= N
            
            dW1 = np.outer(flat_images[i].T, dscores)
            dB1 = np.sum(dscores, axis=0)
            
            dW1 += W1*REG
            #print(dB1)
            W1 += -STEP_SIZE * dW1
            B1 += -STEP_SIZE * dB1
            decision.fill(0)


    
if __name__ == "__main__":
    main()