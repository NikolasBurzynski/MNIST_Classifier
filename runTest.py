import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import PIL
from PIL import Image
import decimal
import matplotlib.pyplot as plt

def main():
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    trained_weights = np.load('TNN', allow_pickle=True)

    W1 = trained_weights[0].reshape(10,784)
    B1 = trained_weights[1].reshape(10,)
    W2 = trained_weights[2].reshape(10,10)
    B2 = trained_weights[3].reshape(10,)

    # test(W1, B1, W2, B2, test_data, test_labels)
    myNumTest(W1, B1, W2, B2, "Numbers/4.png", 4)

def myNumTest(W1, B1, W2, B2, image, label):
    an_image = PIL.Image.open(image)

    grayscale_image = an_image.convert("L")
    grayscale_array = np.asarray(grayscale_image)
    # print(grayscale_array)
    plt.imshow(grayscale_array, cmap="gray")
    

    input_array = grayscale_array.flatten()
    
    input_array = input_array / 255
    print(input_array)
     
    HL = np.maximum(0, W1.dot(input_array)+B1)
        
    SCORE = W2.dot(HL) + B2

    exp_score = np.exp(SCORE)
    norm_score = exp_score / np.sum(exp_score)
    # print(norm_score.shape)
    correct_digit = label
    correct_score = norm_score[correct_digit]

    pred_digit = np.argmax(norm_score)
    print(pred_digit)
    plt.show()


def test(W1, B1, W2, B2, test_data, test_labels):
    N = len(test_labels)
    PIXELS = 784
    images = np.asarray(test_data)
    flat_images = np.empty((N,PIXELS))
    correct = 0
    for i, image in enumerate(images):
        flat_images[i] = images[i].flatten()
        flat_images[i].reshape(784,1)
        flat_images[i] = flat_images[i] / 255
        
    

    for i in range(N):
        HL = np.maximum(0, W1.dot(flat_images[i])+B1)
        # print("Size of Flat Image is {}".format(flat_images[i].shape))
        # print(flat_images[i])
        # print("HL size {}".format(HL.shape))
        SCORE = W2.dot(HL) + B2

        exp_score = np.exp(SCORE)
        norm_score = exp_score / np.sum(exp_score)
        # print(norm_score.shape)
        correct_digit = test_labels[i]
        correct_score = norm_score[correct_digit]

        pred_digit = np.argmax(norm_score)
        pred_digit = pred_digit

        # print("Correct Digit: {} Predicted Digit: {}".format(correct_digit, pred_digit))
        if(pred_digit == correct_digit):
            correct += 1
            
    print(correct / N * 100)

if __name__ == "__main__":
    main()