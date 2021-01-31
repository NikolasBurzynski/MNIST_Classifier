import numpy as np
import time

def main():
    start_time = time.time()
    train_data = np.load('DATA', allow_pickle=True)
    train_labels = np.load('LABELS', allow_pickle=True)
    train(train_data, train_labels, EPOCHS=500, DROPOUT=True)
    print("--- %s seconds ---" % (time.time() - start_time))

    
def train(data, labels, EPOCHS, DROPOUT):
    BATCH_SIZE = 15
    HL_NODES = 75
    PIXELS = 784
    DIGITS = 10
    REG = 1e-3
    STEP_SIZE = .001
    N = len(data)
    drop_p = 0.5
    
    W1 = 0.01 * np.random.randn(HL_NODES,PIXELS)
    B1 = np.zeros(HL_NODES)
    W2 = 0.01 * np.random.randn(DIGITS,HL_NODES)
    B2 = np.zeros(DIGITS)
    correct = 0
    for j in range(EPOCHS):
        adW1 = np.zeros((HL_NODES,PIXELS))
        adB1 = np.zeros(HL_NODES)
        adW2 = np.zeros((DIGITS, HL_NODES))
        adB2 = np.zeros(DIGITS)
        for i in range(N):
            HL = np.maximum(0, W1.dot(data[i])+B1)
            if DROPOUT:
                MHL = (np.random.rand(*HL.shape) < drop_p) / drop_p
                HL *= MHL
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
            print("Epoch: {} Current iteration: {} Loss: {} Confidence: {}".format(j+1, i, total_loss, correct_score), end='\r')
            
            #backPropogation
            dscores = norm_score
            dscores[correct_digit] -= 1

            dW2 = np.outer(dscores, HL) 
    
            dB2 = np.sum(dscores, axis = 0, keepdims=True)
            dHL = W2.T.dot(dscores)
            
            #relu activation backprop
            dHL[HL <= 0] = 0

            dW1 = np.outer(dHL, data[i])
            dB1 = np.sum(dHL, axis = 0, keepdims=True)

            dW2 += REG * W2
            dW1 += REG * W1
            
            #accumulate gradients
            adW1 += dW1 
            adB1 += dB1
            adW2 += dW2
            adB2 += dB2

            #once a batch is finished average the gradients, apply the changes and then empty the accumulator 
            if i%BATCH_SIZE == 0 and i != 0:
                # print("Batch size reached at i = {}".format(i))
                adW1 /= BATCH_SIZE 
                adB1 /= BATCH_SIZE
                adW2 /= BATCH_SIZE
                adB2 /= BATCH_SIZE
                W1 += -(STEP_SIZE * adW1)
                B1 += -(STEP_SIZE * adB1)
                W2 += -(STEP_SIZE * adW2)
                B2 += -(STEP_SIZE * adB2)
                adW1.fill(0) 
                adB1.fill(0)
                adW2.fill(0)
                adB2.fill(0)
            # print(i)

    print()
    print("I got {} correct. This is {} accuracy".format(correct, correct/(N*EPOCHS)))
    trained_NN = np.array([W1.flatten(),B1.flatten(),W2.flatten(),B2.flatten()], dtype = object)
    file_name = 'TNN'
    if(DROPOUT):
        file_name = 'DTNN'        
    file = open(file_name, "wb")
    np.save(file, trained_NN, allow_pickle=True)
    file.close
        

if __name__ == "__main__":
    main()