import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {
    // Criamos um modelo sequencial simples
    const model = tf.sequential();

    // quanto mais neurons, mais complexidade o modelo pode aprender
    // pouca quantiade de base de treino 
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' })); // Camada oculta

    // saida: 3 neuronios categorias (premium, medium, basic)
    // activation softmax normaliza a saída para representar probabilidades
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    model.compile({
        optimizer: 'adam', // Otimizador, aprende com historico de erros e acertos
        loss: 'categoricalCrossentropy', // Ele compara o que o modelo acha (os scores de cada categoria) com a reposta certa
        metrics: ['accuracy'] // Métrica para avaliar o desempenho do modelo
    });

    await model.fit(inputXs, outputYs, {
        verbose: 0, //
        epochs: 100, // Número de vezes que o modelo verá todo o dataset de treino
        shuffle: true, // Embaralha os dados a cada época para melhorar o aprendizado,  
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: loss = ${logs.loss}`);
            }
        }
    });

    return model;
}

async function predict(model, inputTensor) {
    const input = tf.tensor2d([inputTensor]);

    const prediction = model.predict(input);
    const predictionArray = await prediction.array();

    return predictionArray[0].map((prob, index) => ({ prob, index }));
}


// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

const model = await trainModel(inputXs, outputYs)
const pessoasTeste = { nome: "Júlio", idade: 28, cor: "vermelho", localizacao: "Rio" }

// normalizamos os dados de teste da mesma forma que os dados de treino
// Exemplo: idade_min = 25, idade_max = 40
const idadeMin = 25;
const idadeMax = 40;
const idadeNormalizada = (pessoasTeste.idade - idadeMin) / (idadeMax - idadeMin);

const pessoaTensorNormalizado = [
    idadeNormalizada,
    pessoasTeste.cor === "azul" ? 1 : 0,
    pessoasTeste.cor === "vermelho" ? 1 : 0,
    pessoasTeste.cor === "verde" ? 1 : 0,
    pessoasTeste.localizacao === "São Paulo" ? 1 : 0,
    pessoasTeste.localizacao === "Rio" ? 1 : 0,
    pessoasTeste.localizacao === "Curitiba" ? 1 : 0
];

const predictions = await predict(model, pessoaTensorNormalizado)
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(pred => `${labelsNomes[pred.index]}: ${(pred.prob * 100).toFixed(2)}%`)
    .join('\n');

console.log(results)