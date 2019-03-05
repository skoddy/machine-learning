// Linear Regression
// dataset
// predict function
// loss function
// optimizer


let x_vals = [];
let y_vals = [];


let m; // slope
let b; // y intercept

// stochastic gradient descent optimizer
const learningRate = 0.01;
const optimizer = tf.train.sgd(learningRate);

// pred are the y-values from the predict function
function loss(pred, label) {
    // MSE - Mean Squared Error
    // MSE = (pred - y)^2
    return pred.sub(label).square().mean();
}

// predict ys with x_vals
// returns a tensor
function predict(x_vals) {
    // xs is a plain array and needs to converted into a tensor
    const tfxs = tf.tensor1d(x_vals);
    // formula for a line
    // y = mx + b or y = slope x + yIntercept
    const ys = tfxs.mul(m).add(b);
    return ys;
}

function setup() {
    createCanvas(400, 400);
    // initialize m and b
    // these points change over time
    // and must be wrapped with tf.variable
    m = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));
}

function mouseDragged() {
    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);
    x_vals.push(x);
    y_vals.push(y);
}

function draw() {
    tf.tidy(() => {
        if (x_vals.length > 0) {
            const ys = tf.tensor1d(y_vals);
            optimizer.minimize(() => loss(predict(x_vals), ys))
        }
    });

    background(0);
    stroke(255);
    strokeWeight(2);
    for (let i = 0; i < x_vals.length; i++) {
        let px = map(x_vals[i], -1, 1, 0, width);
        let py = map(y_vals[i], -1, 1, height, 0);
        point(px, py);
    }

    const lineX = [-1, 1];
    const ys = tf.tidy(() => predict(lineX));
    let lineY = ys.dataSync();
    ys.dispose();
    let x1 = map(lineX[0], -1, 1, 0, width);
    let x2 = map(lineX[1], -1, 1, 0, width);

    // ys.data().then((data) => lineY = data);

    let y1 = map(lineY[0], -1, 1, height, 0);
    let y2 = map(lineY[1], -1, 1, height, 0);

    line(x1, y1, x2, y2);

    console.log(tf.memory().numTensors);
    // ys.print();
}









/* const values = [];

for (let i = 0; i < 15; i++) {
    values[i] = Math.random() * (1, 100) + 1;
}

const shape = [5, 3];
const data = tf.tensor(values, shape, 'int32');

data.print(); */