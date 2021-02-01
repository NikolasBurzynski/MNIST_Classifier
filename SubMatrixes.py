import numpy as np

def main():
    PIXELS = 5
    BATCH_SIZE = 5
    NUM_IMAGES = 20
    HL_NODES=50
    DIGITS = 10
    data = np.random.randint(5, size = (NUM_IMAGES,PIXELS))
    labels = np.random.randin(10, size = (NUM_IMAGES))
    print(data)
    W1 = np.random.randint(5, size = (PIXELS,HL_NODES))
    B1 = np.random.randint(5, size = (NUM_IMAGES, HL_NODES))
    W2 = np.random.randint(5, size = (HL_NODES,DIGITS))
    B2 = np.random.randint(5, size = (NUM_IMAGES, DIGITS))
    #I need to make a minibatch of the labels as well and pass those through
    for i in range(round(NUM_IMAGES / BATCH_SIZE)):
        mini_data = data[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
        mini_labels = labels[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
        print("MiniBatch {} is {}".format(i+1,mini_data))

def learn(mini_data, mini_labels, W1, B1, W2, B2, REG):  
    HL = np.maximum(0, np.dot(data,W1) + B1) #this is (60010X50) basically all of the images and 50 hidden nodes each
    OUTPUT = np.dot(HL, W2) + B2

    #loss calc
    exp_score = np.exp(OUTPUT)


    return dW1, dB1, dW2, dB2 


if __name__ == "__main__":
    main()