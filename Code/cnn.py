
import numpy as np #imports numpy for matrix
import random #import random
import PIL
import skimage
import conv

from PIL import Image
from skimage import color

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sig_der(x): #plug in y for x
    return x * (1 - x)

def train_and_adjust(training_inputs, training_outputs, training_epochs):

    synaptic_weights = 2 * np.random.random((2500, 1)) - 1

    for it in range(training_iterations):

        final_comp_result = sigmoid(np.dot(inputs, synaptic_weights))
        return final_comp_result

        error = training_outputs - final_comp_result

        adjustments = np.dot(error.T * sig_der(final_comp_result)) # weighted by % error

        synaptic_weights += adjustments

def graph():
    # Graphing results
    fig0, ax0 = matplotlib.pyplot.subplots(nrows=1, ncols=1)
    ax0.imshow(img).set_cmap("gray")
    ax0.set_title("Input Image")
    ax0.get_xaxis().set_ticks([])
    ax0.get_yaxis().set_ticks([])
    matplotlib.pyplot.savefig("in_img.png", bbox_inches="tight")
    matplotlib.pyplot.close(fig0)
    # Layer 1
    fig1, ax1 = matplotlib.pyplot.subplots(nrows=3, ncols=2)
    ax1[0, 0].imshow(l1_feature_map[:, :, 0]).set_cmap("gray")
    ax1[0, 0].get_xaxis().set_ticks([])
    ax1[0, 0].get_yaxis().set_ticks([])
    ax1[0, 0].set_title("L1-Map1")
    ax1[0, 1].imshow(l1_feature_map[:, :, 1]).set_cmap("gray")
    ax1[0, 1].get_xaxis().set_ticks([])
    ax1[0, 1].get_yaxis().set_ticks([])
    ax1[0, 1].set_title("L1-Map2")
    ax1[1, 0].imshow(l1_feature_map_relu[:, :, 0]).set_cmap("gray")
    ax1[1, 0].get_xaxis().set_ticks([])
    ax1[1, 0].get_yaxis().set_ticks([])
    ax1[1, 0].set_title("L1-Map1ReLU")
    ax1[1, 1].imshow(l1_feature_map_relu[:, :, 1]).set_cmap("gray")
    ax1[1, 1].get_xaxis().set_ticks([])
    ax1[1, 1].get_yaxis().set_ticks([])
    ax1[1, 1].set_title("L1-Map2ReLU")
    ax1[2, 0].imshow(l1_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
    ax1[2, 0].get_xaxis().set_ticks([])
    ax1[2, 0].get_yaxis().set_ticks([])
    ax1[2, 0].set_title("L1-Map1ReLUPool")
    ax1[2, 1].imshow(l1_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
    ax1[2, 0].get_xaxis().set_ticks([])
    ax1[2, 0].get_yaxis().set_ticks([])
    ax1[2, 1].set_title("L1-Map2ReLUPool")
    matplotlib.pyplot.savefig("L1.png", bbox_inches="tight")
    matplotlib.pyplot.close(fig1)
    # Layer 2
    fig2, ax2 = matplotlib.pyplot.subplots(nrows=3, ncols=3)
    ax2[0, 0].imshow(l2_feature_map[:, :, 0]).set_cmap("gray")
    ax2[0, 0].get_xaxis().set_ticks([])
    ax2[0, 0].get_yaxis().set_ticks([])
    ax2[0, 0].set_title("L2-Map1")
    ax2[0, 1].imshow(l2_feature_map[:, :, 1]).set_cmap("gray")
    ax2[0, 1].get_xaxis().set_ticks([])
    ax2[0, 1].get_yaxis().set_ticks([])
    ax2[0, 1].set_title("L2-Map2")
    ax2[0, 2].imshow(l2_feature_map[:, :, 2]).set_cmap("gray")
    ax2[0, 2].get_xaxis().set_ticks([])
    ax2[0, 2].get_yaxis().set_ticks([])
    ax2[0, 2].set_title("L2-Map3")
    ax2[1, 0].imshow(l2_feature_map_relu[:, :, 0]).set_cmap("gray")
    ax2[1, 0].get_xaxis().set_ticks([])
    ax2[1, 0].get_yaxis().set_ticks([])
    ax2[1, 0].set_title("L2-Map1ReLU")
    ax2[1, 1].imshow(l2_feature_map_relu[:, :, 1]).set_cmap("gray")
    ax2[1, 1].get_xaxis().set_ticks([])
    ax2[1, 1].get_yaxis().set_ticks([])
    ax2[1, 1].set_title("L2-Map2ReLU")
    ax2[1, 2].imshow(l2_feature_map_relu[:, :, 2]).set_cmap("gray")
    ax2[1, 2].get_xaxis().set_ticks([])
    ax2[1, 2].get_yaxis().set_ticks([])
    ax2[1, 2].set_title("L2-Map3ReLU")
    ax2[2, 0].imshow(l2_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
    ax2[2, 0].get_xaxis().set_ticks([])
    ax2[2, 0].get_yaxis().set_ticks([])
    ax2[2, 0].set_title("L2-Map1ReLUPool")
    ax2[2, 1].imshow(l2_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
    ax2[2, 1].get_xaxis().set_ticks([])
    ax2[2, 1].get_yaxis().set_ticks([])
    ax2[2, 1].set_title("L2-Map2ReLUPool")
    ax2[2, 2].imshow(l2_feature_map_relu_pool[:, :, 2]).set_cmap("gray")
    ax2[2, 2].get_xaxis().set_ticks([])
    ax2[2, 2].get_yaxis().set_ticks([])
    ax2[2, 2].set_title("L2-Map3ReLUPool")
    matplotlib.pyplot.savefig("L2.png", bbox_inches="tight")
    matplotlib.pyplot.close(fig2)
    # Layer 3
    fig3, ax3 = matplotlib.pyplot.subplots(nrows=1, ncols=3)
    ax3[0].imshow(l3_feature_map[:, :, 0]).set_cmap("gray")
    ax3[0].get_xaxis().set_ticks([])
    ax3[0].get_yaxis().set_ticks([])
    ax3[0].set_title("L3-Map1")
    ax3[1].imshow(l3_feature_map_relu[:, :, 0]).set_cmap("gray")
    ax3[1].get_xaxis().set_ticks([])
    ax3[1].get_yaxis().set_ticks([])
    ax3[1].set_title("L3-Map1ReLU")
    ax3[2].imshow(l3_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
    ax3[2].get_xaxis().set_ticks([])
    ax3[2].get_yaxis().set_ticks([])
    ax3[2].set_title("L3-Map1ReLUPool")
    matplotlib.pyplot.savefig("L3.png", bbox_inches="tight")
    matplotlib.pyplot.close(fig3)

"""====================================================================================
sigmoid(summation(weight * input)) <--- forward prop for 'votes'

training data input,


training_inputs = Image.open(count.png) #use os to get cats from folder

training_inputs = skimage.color.rgb2gray(training_inputs) #Changes the array to 1 channel

training_inputs = asarray(training_inputs) #turns the image into an array

training_outputs =

training_epochs = int(input("Enter The Amount of Iterations You Would Like to Train: "))

train_and_adjust(training_inputs, training_outputs, training_epochs)

## THIS IS THE PLACE WHERE THE AI IS TESTED ###########################################
==================================================================================="""

l1_filter = numpy.zeros((2, 3, 3))  # Creates 2 number filters, each with 3 rows and 3 columns

l1_filter[0, :, :] = numpy.array([[[-1, 0, 1, ],  # creates the first array detecting vertical edges
                                   [-1, 0, 1, ],
                                   [-1, 0, 1, ]]])

l1_filter[1, :, :] = numpy.array([[[1, 1, 1],  # creates the second array detecting horizontal edges
                                   [0, 0, 0],
                                   [-1, -1, -1]]])

l1_feature_map = conv(image, l1_filter)  # Convolves image according to conv function and the l1 filter

l1_feature_map_relu = relu(l1_feature_map)

l1_feature_map_relu_pool = pooling(l1_feature_map_relu, 2, 2)

# Second conv layer

l2_filter = numpy.random.rand(3, 5, 5, l1_feature_map_relu_pool.shape[-1])
print("\n**Working with conv layer 2**")

l2_feature_map = conv(l1_feature_map_relu_pool, l2_filter)
print("\n**ReLU**")

l2_feature_map_relu = relu(l2_feature_map)
print("\n**Pooling**")

l2_feature_map_relu_pool = pooling(l2_feature_map_relu, 2, 2)
print("**End of conv layer 2**\n")

# Third conv layer

l3_filter = numpy.random.rand(1, 7, 7, l2_feature_map_relu_pool.shape[-1])
print("\n**Working with conv layer 3**")

l3_feature_map = conv(l2_feature_map_relu_pool, l3_filter)
print("\n**ReLU**")

l3_feature_map_relu = relu(l3_feature_map)
print("\n**Pooling**")

l3_feature_map_relu_pool = pooling(l3_feature_map_relu, 2, 2)
print("**End of conv layer 3**\n")

training_inputs = asarray(training_inputs) #turns the image into an array

# printing initial arrays
print("initial array", str(training_inputs))

# Multiplying arrays
input = training_inputs.flatten()

# printing results
print("New resulting array: ", input)
