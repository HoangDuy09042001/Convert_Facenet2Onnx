const fs = require('fs');
const onnx = require('onnxjs-node');
const { Tensor, InferenceSession } = onnx;

// Create an ONNX Runtime session
const session = new InferenceSession({ backendHint: 'webgl' });
const modelPath = 'facenet.onnx';

// Load the ONNX model
session.loadModel(modelPath);

// Preprocess the input image
const { createCanvas, loadImage } = require('canvas');
const canvas = createCanvas(160, 160);
const ctx = canvas.getContext('2d');

const inputImagePath = '3_org.png';

loadImage(inputImagePath).then((image) => {
    ctx.drawImage(image, 0, 0, 160, 160);

    // Normalize and convert the input image to a tensor
    const inputTensor = new Tensor(new Float32Array(canvas.toBuffer()), 'float32', [1, 3, 160, 160]);

    // Perform inference with the ONNX Runtime session
    const feeds = {};
    feeds['input'] = inputTensor;

    session.run(feeds).then((outputMap) => {
        // The outputMap now contains the model's predictions
        const predictions = outputMap['output']; // Replace 'output' with the actual output name

        console.log('Model outputs:');
        console.log(predictions);
    });
});
