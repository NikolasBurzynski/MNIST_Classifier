import numpy as np

def main():
    PIXELS = 5
    EPOCHS = 10
    BATCH_SIZE = 5
    NUM_IMAGES = 20
    HL_NODES=50
    STEP_SIZE = .001
    REG = 1e-3
    DIGITS = 10
    data = np.random.randn(NUM_IMAGES,PIXELS)
    labels = np.random.randint(10, size = (NUM_IMAGES))
    print(data)
    W1 = .01 * np.random.randn(PIXELS,HL_NODES)
    B1 = np.zeros((BATCH_SIZE, HL_NODES))
    W2 = .01 * np.random.randn(HL_NODES,DIGITS)
    B2 = np.zeros((BATCH_SIZE, DIGITS))
    #I need to make a minibatch of the labels as well and pass those through
    for j in range(EPOCHS):
        for i in range(round(NUM_IMAGES / BATCH_SIZE)):
            mini_data = data[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
            mini_labels = labels[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
            # print("MiniBatch {} is {}".format(i+1,mini_data))
            dW1, dB1, dW2, dB2 = learn(mini_data, mini_labels, W1, B1, W2, B2, REG)
            W1 += -(STEP_SIZE * dW1)
            B1 += -(STEP_SIZE * dB1)
            W2 += -(STEP_SIZE * dW2)
            B2 += -(STEP_SIZE * dB2)

def learn(mini_data, mini_labels, W1, B1, W2, B2, REG):  
    batch_size = mini_labels.size
    HL = np.maximum(0, np.dot(mini_data,W1) + B1) #this is (60010X50) basically all of the images and 50 hidden nodes each
    scores = np.dot(HL, W2) + B2

    #loss calc
    sum_exp_scores = np.exp(scores).sum(axis=1, keepdims=True)
    softmax_matrix = np.exp(scores)/sum_exp_scores
    data_loss = np.sum(-np.log(softmax_matrix[np.arange(batch_size), mini_labels]))/batch_size
    reg_loss = .5*REG*np.sum(W1*W1) + .5*REG*np.sum(W2*W2)
    total_loss = data_loss + reg_loss
    print(total_loss)

    # Weight Gradient
    softmax_matrix[np.arange(batch_size),mini_labels] -= 1
    
    # Backprop Baby
    dW2 = np.dot(HL.T, softmax_matrix)
    dB2 = np.sum(softmax_matrix, axis = 0, keepdims=True)
    dHL = np.dot(softmax_matrix, W2.T)

    #RELU activation backprop
    dHL[HL <= 0] = 0

    dW1 = np.dot(mini_data.T, dHL)
    dB1 = np.sum(dHL, axis = 0, keepdims=True)

    dW2 += REG * W2
    dW1 += REG * W1
    return dW1, dB1, dW2, dB2 


if __name__ == "__main__":
    main()