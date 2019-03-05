// Polynomial Regression
// dataset
// predict function
// loss function
// optimizer


let x_vals = [];
let y_vals = [];

let a, b, c;

// adam optimizer
const learningRate = 0.01;
const optimizer = tf.train.adam(learningRate);

// predictions are x-values from the predict function as tensor
function loss(predictions, labels) {
    // MSE - Mean Squared Error
    // MSE = (predictions - labels)^2, (x - y)^2
    // Subtract our labels (actual values) from predictions, square the results,
    // and take the mean.
    const mse = predictions.sub(labels).square().mean();
    return mse;
  }
// predict ys with x_vals
// returns a tensor
function predict(x_vals) {
    return tf.tidy(() => {
        // x_vals is a plain array and needs to converted into a tensor
        const xs = tf.tensor1d(x_vals);
        // formula
        // y = a*x^2 + b*x + c
        const ys = xs.square().mul(a).add(xs.mul(b)).add(c);
        return ys;

    });
}

function setup() {
    createCanvas(400, 400);
    // initialize a,b and c with zeros
    // these points change over time
    // and must be wrapped with tf.variable
    const initialValues = tf.zeros([1]);
    a = tf.variable(initialValues);
    b = tf.variable(initialValues);
    c = tf.variable(initialValues);
}

function mouseDragged() {
    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);
    x_vals.push(x);
    y_vals.push(y);
}

function draw() {

    if (x_vals.length > 0) {
        const ys = tf.tensor1d(y_vals);
        optimizer.minimize(() => {
            const predsYs = predict(x_vals);
            return loss(predsYs, ys);
        });
        ys.dispose();
    }


    background(0);
    stroke(255);
    strokeWeight(2);
    for (let i = 0; i < x_vals.length; i++) {
        let px = map(x_vals[i], -1, 1, 0, width);
        let py = map(y_vals[i], -1, 1, height, 0);
        point(px, py);
    }

    const curveX = [];
    for (let x = -1; x < 1.01; x += 0.05) {
        curveX.push(x);
    }
    const ys = tf.tidy(() => predict(curveX));
    let curveY = ys.dataSync();
    ys.dispose();
    beginShape();
    noFill();
    stroke(255);
    strokeWeight(2);
    for (let i = 0; i < curveX.length; i++) {
        let x = map(curveX[i], -1, 1, 0, width);
        let y = map(curveY[i], -1, 1, height, 0);
        vertex(x, y);
    }
    endShape();
    console.log(tf.memory().numTensors);

}
