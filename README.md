**#MNIST data**

MNIST is a database of images, resolution 28 by 28 pixels, in 8-bit format, representing handwritten digits 0-9 in equal proportions. Many different machine learning algorithms have been tested on their ability to identify each image as the appropriate digit 0-9.

Each algorithm has an accuracy rating, which represents how often it outputs the correct digit when given one image in the MNIST test set.

MNIST data is public data. Accordingly, the source files are not included in the repository but can be easily found online.

**#Using XGBoost on MNIST**

1. Python code is written to use XGBoost, train on a subset of MNIST samples
   
The model is a Random forest, works faster than TensorFlow, does not require one-hot encoding

2. Then the code will test on a subset of MNIST testing data

Around 95% accuracy after using 12,000 training samples, 300 estimators and max depth of 9 levels (Random chance would be 10% accuracy)
[Note: This is comparable to the result obtained with simple Convolutional Neural Networks (CNN).]

# Noisy-MNIST

1. Code reduces value range to small number by multiplying all pixel values by a small constant
   
2. Replaces a rectangular region in each sample with a “patch of noise” – the noise will have the full range of values 0-255 while the useful data will only have small values!
   
3. Code Randomizes both the location of the rectangle and the pixel values within it
   
4. Two types of noisy MNIST:
   
   Type 1 - Noise-free training data, noisy test data
   
   Type 2 - Noisy training data, noisy test data

**#Output**

Code produces a plot of the accuracy as a function of the fraction of the image area that is replaced by noise. Two curves are shown, one for each type of noisy MNIST.

**#Result Conclusion**

1. Adding a “noisy region” to the test data results in progressive degradation of XGBoost-based digit identification in MNIST test data, with larger noise areas resulting in increasingly poor performance
   
2. However, the performance degradation is greatly reduced when the model is trained on data with statistically similar noise
   
3. Demonstrates the value of training models on data with similar noise profile as the data it is used on






