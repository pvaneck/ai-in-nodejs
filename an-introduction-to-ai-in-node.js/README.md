# An introduction to AI in Node.js

This directory contains source code for exercises from the 'Getting Started' tutorial.

## Contents

**[run-prepacked-model.js](./run-prepacked-model.js)**: Script showing how to use a prepackaged
TensorFlow.js NPM module to quickly make a Node object detection app.

**[run-tfjs-model.js](./run-tfjs-model.js)** - Script showing how to load an object detection
model with TensorFlow.js directly. Here, we show the preprocessing and postprocessing steps
necessary in order to pass in an image for inference and get nicely formated results returned.

**[labels.js](./labels.js)**: - A mapping of the object labels to their index values/IDs
returned by the model in `run-tfjs-model.js`.

**[sample-images](./sample-images)**: Directory containing some sample images that can be used
with the object detection scripts.
