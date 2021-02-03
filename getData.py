from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import os
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import random

def main():
    print("Crafting Data")
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    images = np.asarray(train_data)
    N = len(train_data)
    PIXELS = 784
    flat_images = np.empty((N,PIXELS))
    for i, image in enumerate(images):
        flat_images[i] = images[i].flatten()
        flat_images[i].reshape(784,1)
        flat_images[i] = flat_images[i] / 255
    # flat_images, train_labels = addHandwritten(flat_images, train_labels)
    file = open("DATANOHW", "wb")
    np.save(file, flat_images, allow_pickle = True)
    file.close
    file = open("LABELSNOHW", "wb")
    np.save(file, train_labels, allow_pickle = True)
    file.close
    print("Done")



def addHandwritten(images, labels):
    directory = r'C:\Users\Niko\Desktop\MNIST_Classifier\Numbers\Training'
    for i, filename in enumerate(os.listdir(directory)):
        # print(i)
        # print(filename)
        index = random.randint(0,len(images)-2)#2 just in case
        an_image = PIL.Image.open('Numbers\Training\\' + filename)
        grayscale_image = an_image.convert("L")
        grayscale_array = np.asarray(grayscale_image)
        plt.imshow(grayscale_array, cmap="gray")
        input_array = grayscale_array.flatten().reshape((1,784))
        input_array = input_array / 255
        images[index] = input_array
        labels[index] - i
        print(index)
    print(images.shape)
    return images, labels



if __name__ == "__main__":
    main()