import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';
console.log('Model training worker initialized');
let _globalCtx = {};
let _model = {};
const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
}

const normalize = (value, min, max) => (value - min) / (max - min);

function makeContext(products, users) {
    const ages = users.map(u => u.age);
    const prices = products.map(p => p.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    const colors = [...new Set(products.map(p => p.color))];
    const categories = [...new Set(products.map(p => p.category))];

    const colorIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index];
        })
    );

    const categoryIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index];
        })
    );

    // computar a media de idade dos compradores do produto
    const midAge = (minAge + maxAge) / 2;
    const ageSums = {};
    const ageCounts = {};

    // calcular a soma das idades e contagem de compradores para cada produto
    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age;
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
        });
    });

    // calcular a média de idade normalizada para cada produto
    const productAvgAgeNorm = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ?
                ageSums[product.name] / ageCounts[product.name] :
                midAge;

            return [product.name, normalize(avg, minAge, maxAge)];
        })
    );

    return {
        products,
        users,
        colorIndex,
        categoryIndex,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        // price + age + categories + colors
        dimensions: 2 + categories.length + colors.length,
        productAvgAgeNorm
    };
}

const oneHotWeighted = (index, length, weight) =>
    tf.oneHot(index, length).cast('float32').mul(weight);

function encodeProduct(product, ctx) {
    // normalizando os dados para ficar entre 0 e 1
    // aplicar peso na representação
    const price = tf.tensor1d([
        normalize(
            product.price,
            ctx.minPrice,
            ctx.maxPrice
        ) * WEIGHTS.price
    ])

    const age = tf.tensor1d([
        (ctx.productAvgAgeNorm[product.name] ?? 0.5) * WEIGHTS.age
    ]);

    const category = oneHotWeighted(
        ctx.categoryIndex[product.category],
        ctx.numCategories,
        WEIGHTS.category
    );

    const color = oneHotWeighted(
        ctx.colorIndex[product.color],
        ctx.numColors,
        WEIGHTS.color
    );

    return tf.concat1d([price, age, category, color]);
}

function encodeUser(user, ctx) {
    if (user.purchases.length) {
        return tf.stack(
            user.purchases.map(
                p => encodeProduct(p, ctx)
            )
        )
            .mean(0)
            .reshape([1, ctx.dimensions]);
    }

    return tf.concat1d(
        [
            tf.zeros([1]), // preco é ignorado
            tf.tensor1d([normalize(user.age, ctx.minAge, ctx.maxAge) * WEIGHTS.age]),
            tf.zeros([ctx.numCategories]), // categorias são ignoradas
            tf.zeros([ctx.numColors]) // cores são ignoradas
        ]
    ).reshape([1, ctx.dimensions]);
}

function createTrainingData(ctx) {
    const input = []
    const labels = []

    ctx.users
        .filter(u => u.purchases.length) // considerar apenas usuários com compras
        .forEach(user => {
            const userVector = encodeUser(user, ctx).dataSync();
            ctx.products.forEach(product => {
                const productVector = encodeProduct(product, ctx).dataSync();

                const label = user.purchases.some(p => p.name === product.name) ? 1 : 0;

                // combinar usuario + product
                input.push([...userVector, ...productVector]);
                labels.push(label);
            })
        });

    return {
        xs: tf.tensor2d(input),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        // tamanho do vetor de entrada: vetor do usuário + vetor do produto
        inputDimensions: ctx.dimensions * 2,
    }
}

async function configureNeuralNetAndTrain(trainData) {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [trainData.inputDimensions], units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    model.compile({ optimizer: tf.train.adam(0.01), loss: 'binaryCrossentropy', metrics: ['accuracy'] });

    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100, batchSize: 32, shuffle: true, callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    })

    return model
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)
    postMessage({
        type: workerEvents.progressUpdate, progress: { progress: 50 }
    });
    const products = await (await fetch('/data/products.json')).json();

    const context = makeContext(products, users);

    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: { ...product },
            vector: encodeProduct(product, context).dataSync()
        }
    });

    _globalCtx = context;

    const trainingData = createTrainingData(context);
    _model = await configureNeuralNetAndTrain(trainingData);
    console.log('products loaded:', products);

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}
function recommend(user, ctx) {
    if (!_model) return;

    const userVector = encodeUser(user, ctx).dataSync();
    const inputs = ctx.productVectors.map(({ vector }) => [...userVector, ...vector]);

    const inputTensor = tf.tensor2d(inputs);
    const predictions = _model.predict(inputTensor);

    const scores = predictions.dataSync();
    const recommendations = ctx.productVectors
        .map((item, index) => { return { ...item.meta, name: item.name, score: scores[index] } })

    const sortedItems = recommendations.sort((a, b) => b.score - a.score);

    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedItems
    });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
