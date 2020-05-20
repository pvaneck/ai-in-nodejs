// export the node module
module.exports = function(RED) {
  // import helper module
  const tfutil = require('./tfjs-tutorial-util.js');

  // load the model
  async function loadModel (config, node) {
    node.model = await tfutil.loadModel(config.modelUrl, config.fromHub);
  }

  // define the node's behavior
  function TfjsTutorialNode(config) {
    // initialize the features
    RED.nodes.createNode(this, config);
    const node = this

    loadModel(config, node)
    
    // register a listener to get called whenever a message arrives at the node
    node.on('input', function (msg) {
      // preprocess the incoming image
      const inputTensor = tfutil.processInput(msg.payload);
      // get image/input shape
      const height = inputTensor.shape[1];
      const width = inputTensor.shape[2];

      // get the prediction
      node.model
        .executeAsync(inputTensor)
        .then(prediction => {
          msg.payload = tfutil.processOutput(prediction, height, width);
          // send the prediction out
          node.send(msg);
        });
    });
  }

  // register the node with the runtime
  RED.nodes.registerType('tfjs-tutorial-node', TfjsTutorialNode);
}
