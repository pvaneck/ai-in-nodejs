# Building a machine learning node for Node-RED using TensorFlow.js

This directory contains source code for exercises from the 'Building a machine learning node for Node-RED using TensorFlow.js' tutorial.

## Contents

**[nodered-dev](./nodered-dev)**: Development environment for creating a custom Node-RED node. This contains a
sample [flow](./nodered-dev/flows.json) that uses the custom node.

**[nodered-dev/node-red-contrib-tfjs-tutorial](./nodered-dev/node-red-contrib-tfjs-tutorial)** - The node package
for the custom node that performs object detection. Can be installed into a Node-RED environment.

**[raspberrypi-flow.json](./raspberrypi-flow.json)**: - A sample Node-RED flow meant for running on a Raspberry Pi.
This leverages attached sensors and peripherals. This flow detects motions, and if motion was detected, an image is
captured with a usb camera and sent through some TF custom nodes for object detection. If a class of interest
was detected, a specific audio clip is played through the connected speaker.

**[package-sample.json](./package-sample.json)**: A sample package.json file for deploying Node-RED apps
on the cloud.