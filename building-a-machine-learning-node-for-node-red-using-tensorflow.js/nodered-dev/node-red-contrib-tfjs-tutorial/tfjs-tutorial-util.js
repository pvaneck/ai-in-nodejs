const tf = require('@tensorflow/tfjs-node');
const labels = require('./labels.js');

const maxNumBoxes = 5;

// load COCO-SSD graph model from TensorFlow Hub
const loadModel = async function (modelUrl, fromTFHub) {
  console.log(`loading model from ${modelUrl}`);

  if (fromTFHub) {
    model = await tf.loadGraphModel(modelUrl, {fromTFHub: true});
  } else {
    model = await tf.loadGraphModel(modelUrl);
  }

  return model;
}

// convert image to Tensor
const processInput = function (imageBuffer) {
  console.log(`preprocessing image`);

  const uint8array = new Uint8Array(imageBuffer);

  return tf.node.decodeImage(uint8array, 3).expandDims();
}

// process the model output into a friendly JSON format
const processOutput = function (prediction, height, width) {
  console.log('processOutput');

  const [maxScores, classes] = extractClassesAndMaxScores(prediction[0]);
  const indexes = calculateNMS(prediction[1], maxScores);

  return createJSONresponse(prediction[1].dataSync(), maxScores, indexes, classes, height, width);
}

// determine the classes and max scores from the prediction
const extractClassesAndMaxScores = function (predictionScores) {
  console.log('calculating classes & max scores');

  const scores = predictionScores.dataSync();
  const numBoxesFound = predictionScores.shape[1];
  const numClassesFound = predictionScores.shape[2];

  const maxScores = [];
  const classes = [];

  // for each bounding box returned
  for (let i = 0; i < numBoxesFound; i++) {
    let maxScore = -1;
    let classIndex = -1;

    // find the class with the highest score
    for (let j = 0; j < numClassesFound; j++) {
      if (scores[i * numClassesFound + j] > maxScore) {
        maxScore = scores[i * numClassesFound + j];
        classIndex = j;
      }
    }

    maxScores[i] = maxScore;
    classes[i] = classIndex;
  }

  return [maxScores, classes];
}

// perform non maximum suppression of bounding boxes
const calculateNMS = function (outputBoxes, maxScores) {
  console.log('calculating box indexes');

  const boxes = tf.tensor2d(outputBoxes.dataSync(), [outputBoxes.shape[1], outputBoxes.shape[3]]);
  const indexTensor = tf.image.nonMaxSuppression(boxes, maxScores, maxNumBoxes, 0.5, 0.5);

  return indexTensor.dataSync();
}

// create JSON object with bounding boxes and label
const createJSONresponse = function (boxes, scores, indexes, classes, height, width) {
  console.log('create JSON output');

  const count = indexes.length;
  const objects = [];

  for (let i = 0; i < count; i++) {
    const bbox = [];

    for (let j = 0; j < 4; j++) {
      bbox[j] = boxes[indexes[i] * 4 + j];
    }

    const minY = bbox[0] * height;
    const minX = bbox[1] * width;
    const maxY = bbox[2] * height;
    const maxX = bbox[3] * width;

    objects.push({
      bbox: [minX, minY, maxX, maxY],
      label: labels[classes[indexes[i]]],
      score: scores[indexes[i]]
    });
  }

  return objects;
}

module.exports = {
  loadModel: loadModel,
  processInput: processInput,
  processOutput: processOutput
}
