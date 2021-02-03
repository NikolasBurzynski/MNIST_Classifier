import numpy as np
import time
def main():

    start_time = time.time()
    data = np.load('HWDATA', allow_pickle=True)
    labels = np.load('HWLABELS', allow_pickle=True)
    
    #Hyperparams
    PIXELS = len(data[0])
    EPOCHS = 750
    BATCH_SIZE = 2
    NUM_IMAGES = len(data)
    HL_NODES=75
    STEP_SIZE = .001
    REG = 1e-3
    DIGITS = 10
    DROP_P = .5
    DROPOUT = True

    if NUM_IMAGES % BATCH_SIZE != 0:
        print("Incompatible batch size, please change it")
        exit()

    
    W1, B1, W2, B2 = initializeNN(PIXELS, HL_NODES, BATCH_SIZE, DIGITS)
    dataSeen = 0
    totalCorrect = 0
    
    for j in range(EPOCHS):
        for i in range(round(NUM_IMAGES / BATCH_SIZE)):
            mini_data = data[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
            mini_labels = labels[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
            dW1, dB1, dW2, dB2, total_loss, numCorrect = learn(mini_data, mini_labels, W1, B1, W2, B2, REG, DROPOUT, BATCH_SIZE, DROP_P)
            dataSeen += BATCH_SIZE
            totalCorrect += numCorrect
            W1 += -(STEP_SIZE * dW1)
            B1 += -(STEP_SIZE * dB1)
            W2 += -(STEP_SIZE * dW2)
            B2 += -(STEP_SIZE * dB2)
            print("EPOCH: {} BATCH NUM: {} CURRENT LOSS: {} Accuracy: {}%".format(j+1, i+1, total_loss, totalCorrect/dataSeen*100), end='\r')

    trained_NN = np.array([W1.flatten(),B1[0],W2.flatten(),B2[0]], dtype = object)
    file_name = 'vectorizedTNN'
    if(DROPOUT):
        file_name = 'HWONLYvectorizedDTNN'        
    file = open(file_name, "wb")
    np.save(file, trained_NN, allow_pickle=True)
    file.close
    print()
    print("--- %s seconds ---" % (time.time() - start_time))


def initializeNN(PIXELS, HL_NODES, BATCH_SIZE, DIGITS):
    W1 = .01 * np.random.randn(PIXELS,HL_NODES)
    B1 = np.zeros((BATCH_SIZE, HL_NODES))
    W2 = .01 * np.random.randn(HL_NODES,DIGITS)
    B2 = np.zeros((BATCH_SIZE, DIGITS))
    return W1, B1, W2, B2

def learn(mini_data, mini_labels, W1, B1, W2, B2, REG, DROPOUT, batch_size, DROP_P):  
    HL = np.maximum(0, np.dot(mini_data,W1) + B1) 
    #Dropout
    if DROPOUT:
        MHL = (np.random.rand(*HL.shape) < DROP_P) / DROP_P
        HL *= MHL
    scores = np.dot(HL, W2) + B2

    #loss calc
    sum_exp_scores = np.exp(scores).sum(axis=1, keepdims=True)
    norm_matrix = np.exp(scores)/sum_exp_scores
    predicted_digits = norm_matrix.argmax(1)
    numCorrect = np.sum(predicted_digits == mini_labels)
    data_loss = np.sum(-np.log(norm_matrix[np.arange(batch_size), mini_labels]))/batch_size
    reg_loss = .5*REG*np.sum(W1*W1) + .5*REG*np.sum(W2*W2)
    total_loss = data_loss + reg_loss

    # Back prop softmax 
    norm_matrix[np.arange(batch_size),mini_labels] -= 1
    
    
    dW2 = np.dot(HL.T, norm_matrix)
    dB2 = np.sum(norm_matrix, axis = 0, keepdims=True)

    dHL = np.dot(norm_matrix, W2.T)

    #RELU activation backprop
    dHL[HL <= 0] = 0

    dW1 = np.dot(mini_data.T, dHL)
    dB1 = np.sum(dHL, axis = 0, keepdims=True)

    dW2 += REG * W2
    dW1 += REG * W1

    return dW1, dB1, dW2, dB2, total_loss, numCorrect


if __name__ == "__main__":
    main()