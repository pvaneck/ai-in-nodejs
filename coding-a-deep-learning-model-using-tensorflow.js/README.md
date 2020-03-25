# An introduction to AI in Node.js

This directory contains source code for exercises from the 'Coding a deep learning model using TensorFlow.js'
tutorial.

## Contents

**[build-model.js](./build-model.js)**: Script showing how to build and train a model to classify for a
a subset of the classes in the [Fashion-MNIST](https://developer.ibm.com/exchanges/data/all/fashion-mnist/)
dataset. This uses the TensorFlow.js Layers API.

**[test-model.js](./test-model.js)** - Script showing how to preprocess an image to be fed into
the trained model saved from `build-model.js` for inference. This script will output predictions for
the image.

**[transfer-learn.js](./transfer-learn.js)**: - Script showing how to perform transfer learning using
the model from `build-model.js`. We show how quick and easy it is to train a classifier for the
remaining classes of Fashion-MNIST.

**[sample-images](./sample-images)**: Directory containing some sample images that can be used
with `test-model.js` for testing out the models.
