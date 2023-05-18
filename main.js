const NUM_CLASSES = 3;
const IMAGE_SIZE = 224;
let mouseDown = false;
let model;
let truncated_model;
let classify = document.getElementById('classify');
let find = false;
const video_webcam = document.getElementById('webcam');

const btn_train = document.getElementById('btn-class-train');
const btn_class_1 = document.getElementById('btn-class-1');
const btn_class_2 = document.getElementById('btn-class-2');
const btn_class_3 = document.getElementById('btn-class-3');

class ControllerDataset {
    constructor(numClasses) {
        this.numClasses = numClasses;
    }

    addExample(example, label) {
        const y = tf.tidy(
            () => tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses));

        if (this.xs == null) {
            this.xs = tf.keep(example);
            this.ys = tf.keep(y);
        } else {
            const oldX = this.xs;
            this.xs = tf.keep(oldX.concat(example, 0));

            const oldY = this.ys;
            this.ys = tf.keep(oldY.concat(y, 0));

            oldX.dispose();
            oldY.dispose();
            y.dispose();
        }
    }
}
const controllerDataset = new ControllerDataset(NUM_CLASSES);

navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
    .then(function (stream) {
        var videoElement = document.getElementById('webcam');

        videoElement.srcObject = stream;
    })
    .catch(function (error) {
        console.error('Error access:', error);
    });

async function loadTruncatedMobileNet() {
    const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

    // Return a model that outputs an internal activation.
    const layer = model.getLayer('conv_pw_13_relu');

    return tf.model({inputs: model.inputs, outputs: layer.output});
}
async function init() {
    truncated_model = await loadTruncatedMobileNet();
}

async function predict () {
    while (true) {
        const img = tf.browser.fromPixels(video_webcam);

        const cropped = cropImage(img);
        const resized = tf.image.resizeBilinear(cropped, [IMAGE_SIZE, IMAGE_SIZE]);
        const processedImg = tf.tidy(() => resized.expandDims(0).toFloat().div(127).sub(1));

        let embeddings = truncated_model.predict(processedImg);

        let pr = model.predict(embeddings);

        let predictedClass = pr.as1D().argMax();
        let classId = (await predictedClass.data())[0];

        classify.innerHTML = classId + 1;

        resized.dispose();
        cropped.dispose();
        img.dispose();
        processedImg.dispose();
        predictedClass.dispose();
        await tf.nextFrame();
    }
}


async function train() {
    if (controllerDataset.xs == null) {
        throw new Error('Add some examples before training!');
    }

    // Creates a 2-layer fully connected model. By creating a separate model,
    // rather than adding layers to the mobilenet model, we "freeze" the weights
    // of the mobilenet model, and only train weights from the new model.
    model = tf.sequential({
        layers: [
            // Flattens the input to a vector so we can use it in a dense layer. While
            // technically a layer, this only performs a reshape (and has no training
            // parameters).
            tf.layers.flatten(
                {inputShape: truncated_model.outputs[0].shape.slice(1)}),
            // Layer 1.
            tf.layers.dense({
                units: 100,
                activation: 'relu',
                kernelInitializer: 'varianceScaling',
                useBias: true
            }),
            // Layer 2. The number of units of the last layer should correspond
            // to the number of classes we want to predict.
            tf.layers.dense({
                units: NUM_CLASSES,
                kernelInitializer: 'varianceScaling',
                useBias: false,
                activation: 'softmax'
            })
        ]
    });

    const optimizer = tf.train.adam(0.0001);

    model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

    const batchSize = Math.floor(controllerDataset.xs.shape[0] * 0.4);

    if (!(batchSize > 0)) {
        throw new Error(
            `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
    }

    await model.fit(controllerDataset.xs, controllerDataset.ys, {
        batchSize,
        epochs: 100,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                console.log('Loss: ' + logs.loss.toFixed(5));
            }
        }
    });

    await predict();
}

function findSubstrings(str, substrings) {
    return substrings.some(substring => str.includes(substring));
}

cropImage = (img) => {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}

video_webcam.addEventListener('loadeddata', (e) => {
    if(video_webcam.readyState >= 3){

        init();
    }
});

async function handler(label) {
    mouseDown = true;
    const btn = document.getElementById('btn-class-' + (label + 1));

    while (mouseDown) {
        //addExampleHandler(label);

        const img = tf.browser.fromPixels(video_webcam);

        const cropped = cropImage(img);
        const resized = tf.image.resizeBilinear(cropped, [IMAGE_SIZE, IMAGE_SIZE]);
        const processedImg = tf.tidy(() => resized.expandDims(0).toFloat().div(127).sub(1));

        controllerDataset.addExample(truncated_model.predict(processedImg), label);

        let num = (1 + parseInt(btn.getAttribute('data-num')));
        btn.setAttribute('data-num', num);
        btn.innerText = 'Class (' + num + ')';


        resized.dispose();
        cropped.dispose();
        img.dispose();
        processedImg.dispose();
        await tf.nextFrame();
    }
}

btn_class_1.addEventListener('mousedown', () => handler(0));
btn_class_1.addEventListener('mouseup', () => mouseDown = false);

btn_class_2.addEventListener('mousedown', () => handler(1));
btn_class_2.addEventListener('mouseup', () => mouseDown = false);

btn_class_3.addEventListener('mousedown', () => handler(2));
btn_class_3.addEventListener('mouseup', () => mouseDown = false);

btn_train.addEventListener('click', () => train());


