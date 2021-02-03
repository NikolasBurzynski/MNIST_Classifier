from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import os
import PIL
from PIL import Image
import matplotlib.pyplot as plt

def main():
    print("Crafting Test Data")
    PIXELS = 784
    directory = r'C:\Users\Niko\Desktop\MNIST_Classifier\Niko Nums'
    images = np.empty((0,784))
    labels = np.empty((0,1), dtype=int)
    for i, filename in enumerate(os.listdir(directory)):
        # print(i)
        # print(filename)
        an_image = PIL.Image.open('Niko Nums\\' + filename)
        grayscale_image = an_image.convert("L")
        grayscale_array = np.asarray(grayscale_image)
        plt.imshow(grayscale_array, cmap="gray")
        input_array = grayscale_array.flatten().reshape((1,784))
        input_array = input_array / 255
        # print(input_array.shape)
        images = np.append(images, input_array, axis = 0)
        # print(images.shape)
        labels = np.append(labels, i+1)
    directory = r'C:\Users\Niko\Desktop\MNIST_Classifier\Alex Nums'
    for i, filename in enumerate(os.listdir(directory)):
        # print(i)
        # print(filename)
        an_image = PIL.Image.open('Alex Nums\\' + filename)
        grayscale_image = an_image.convert("L")
        grayscale_array = np.asarray(grayscale_image)
        plt.imshow(grayscale_array, cmap="gray")
        input_array = grayscale_array.flatten().reshape((1,784))
        input_array = input_array / 255
        # print(input_array.shape)
        images = np.append(images, input_array, axis = 0)
        # print(images.shape)
        labels = np.append(labels, i+1)
    file = open("HWTestingData", "wb")
    np.save(file, images, allow_pickle = True)
    file.close
    file = open("HWTestingLabels", "wb")
    np.save(file, labels, allow_pickle = True)
    file.close
    print("Done")

if __name__ == "__main__":
    main()