// TensorFlow.js for Node,js
const tf = require('@tensorflow/tfjs-node');

// Fashion-MNIST training & test data
const trainDataUrl = 'file://./fashion-mnist/fashion-mnist_train.csv';
const testDataUrl = 'file://./fashion-mnist/fashion-mnist_test.csv';

// mapping of Fashion-MNIST labels (i.e., T-shirt=0, Trouser=1, etc.)
const labels = [
  'T-shirt/top',
  'Trouser',
  'Pullover',
  'Dress',
  'Coat',
  'Sandal',
  'Shirt',
  'Sneaker',
  'Bag',
  'Ankle boot'
];


// Build, train a model with a subset of the data
// Use the first n classes
const numOfClasses = 5;

const imageWidth = 28;
const imageHeight = 28;
const imageChannels = 1;

const batchSize = 100;
const epochsValue = 5;

// load and transform data
const loadData = function (dataUrl, batches=batchSize) {
  // normalize data values between 0-1
  const normalize = ({xs, ys}) => {
    return {
        xs: Object.values(xs).map(x => x / 255),
        ys: ys.label
    };
  };

  // transform input array (xs) to 3D tensor
  // binarize output label (ys)
  const transform = ({xs, ys}) => {
    // array of zeros
    const zeros = (new Array(numOfClasses)).fill(0);

    return {
        xs: tf.tensor(xs, [imageWidth, imageHeight, imageChannels]),
        ys: tf.tensor1d(zeros.map((z, i) => {
            return i === (ys - numOfClasses) ? 1 : 0;
        }))
    };
  };

  // load, normalize, transform, batch
  return tf.data
    .csv(dataUrl, {columnConfigs: {label: {isLabel: true}}})
    .map(normalize)
    .filter(f => f.ys >= (labels.length - numOfClasses))
    .map(transform)
    .batch(batchSize);
};

// Define the model architecture
const buildModel = function (baseModel) {

  // Remove the last layer of the base model. This is the softmax
  // classification layer used for classifying the first 5 classes
  // of Fashion-MNIST. This leaves us with the 'Flatten' layer as the
  // new final layer.
  baseModel.layers.pop();

  // Freeze the weights in the base model layers (feature layers) so they
  // don't change when we train the new model.
  for (layer of baseModel.layers) {
    layer.trainable = false;
  }

  // Create a new sequential model starting from the layers of the
  // previous model.
  const model = tf.sequential({
    layers: baseModel.layers
  });

  // Add a new softmax dense layer. This layer will have the trainable
  // parameters for classifying our new classes.
  model.add(tf.layers.dense({
    units: numOfClasses,
    activation: 'softmax',
    name: 'topSoftmax'
  }));

  model.compile({
    optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
  });

  return model;
}

// train the model against the training data
const trainModel = async function (model, trainingData, epochs=epochsValue) {
  const options = {
    epochs: epochs,
    verbose: 0,
    callbacks: {
      onEpochBegin: async (epoch, logs) => {
        console.log(`Epoch ${epoch + 1} of ${epochs} ...`)
      },
      onEpochEnd: async (epoch, logs) => {
        console.log(`  train-set loss: ${logs.loss.toFixed(4)}`)
        console.log(`  train-set accuracy: ${logs.acc.toFixed(4)}`)
      }
    }
  };

  return await model.fitDataset(trainingData, options);
};

// verify the model against the test data
const evaluateModel = async function (model, testingData) {
  const result = await model.evaluateDataset(testingData);
  const testLoss = result[0].dataSync()[0];
  const testAcc = result[1].dataSync()[0];

  console.log(`  test-set loss: ${testLoss.toFixed(4)}`);
  console.log(`  test-set accuracy: ${testAcc.toFixed(4)}`);
};

const run = async function () {

  const trainData = loadData(trainDataUrl);
  const testData = loadData(testDataUrl);

  // Determine how many batches to take for reduced training set.
  const amount = Math.floor(3000 / batchSize);
  const trainDataSubset = trainData.take(amount);

  const baseModelUrl = 'file://./fashion-mnist-tfjs/model.json';
  const saveModelPath = 'file://./fashion-mnist-tfjs-transfer';

  const baseModel =  await tf.loadLayersModel(baseModelUrl);
  const newModel = buildModel(baseModel);
  newModel.summary();

  const info = await trainModel(newModel, trainDataSubset);
  console.log('\r\n', info);

  console.log('\r\nEvaluating model...');
  await evaluateModel(newModel, testData);

  console.log('\r\nSaving model...');
  await newModel.save(saveModelPath);
};

run();
