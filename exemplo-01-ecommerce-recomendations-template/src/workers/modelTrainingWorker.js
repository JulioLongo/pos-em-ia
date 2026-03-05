// Importa o TensorFlow.js diretamente da CDN para usar dentro do Web Worker
import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
// Importa os nomes dos eventos usados para comunicação entre o worker e a página principal
import { workerEvents } from '../events/constants.js';
console.log('Model training worker initialized');

// Contexto global que será preenchido após o treinamento, armazenando produtos e vetores gerados
let _globalCtx = {};
// Modelo de rede neural que será criado e treinado
let _model = {};

// Pesos que definem a importância de cada característica durante a codificação
// Categoria tem mais influência (0.4), cor vem em seguida (0.3), e preço e idade têm menor peso
const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
}

// Normaliza um valor para o intervalo [0, 1] com base no mínimo e máximo do conjunto
// Ex: idade 30 entre min=18 e max=60 → (30-18)/(60-18) ≈ 0.28
const normalize = (value, min, max) => (value - min) / (max - min);

// Monta o "contexto" global: um objeto com todas as informações pré-calculadas
// necessárias para codificar produtos e usuários de forma numérica
function makeContext(products, users) {
    // Extrai todas as idades e preços para calcular os limites de normalização
    const ages = users.map(u => u.age);
    const prices = products.map(p => p.price);

    // Faixa de idades dos usuários (usada para normalizar a idade)
    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    // Faixa de preços dos produtos (usada para normalizar o preço)
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    // Lista de cores e categorias únicas encontradas nos produtos
    const colors = [...new Set(products.map(p => p.color))];
    const categories = [...new Set(products.map(p => p.category))];

    // Mapeia cada cor para um índice numérico (ex: { 'red': 0, 'blue': 1, ... })
    // Necessário para criar o one-hot encoding
    const colorIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index];
        })
    );

    // Mapeia cada categoria para um índice numérico (ex: { 'shoes': 0, 'bags': 1, ... })
    const categoryIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index];
        })
    );

    // Idade média do dataset, usada como fallback para produtos sem nenhuma compra registrada
    const midAge = (minAge + maxAge) / 2;
    const ageSums = {};
    const ageCounts = {};

    // Percorre todos os usuários e suas compras para acumular a soma de idades
    // e o número de compradores por produto
    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age;
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
        });
    });

    // Para cada produto, calcula a média de idade de quem o comprou e normaliza para [0, 1]
    // Se ninguém comprou ainda, usa a idade média do dataset como estimativa
    const productAvgAgeNorm = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ?
                ageSums[product.name] / ageCounts[product.name] :
                midAge;

            return [product.name, normalize(avg, minAge, maxAge)];
        })
    );

    // Retorna o contexto completo com tudo que será usado nas etapas seguintes
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
        // Tamanho total do vetor de um produto: 1 (preço) + 1 (idade média) + nº categorias + nº cores
        dimensions: 2 + categories.length + colors.length,
        productAvgAgeNorm
    };
}

// Cria um vetor one-hot: todos os valores são 0 exceto o índice correspondente que recebe 1
// Em seguida multiplica pelo peso para dar mais ou menos importância à característica
// Ex: categoria 'shoes' (índice 2 de 4) com peso 0.4 → [0, 0, 0.4, 0]
const oneHotWeighted = (index, length, weight) =>
    tf.oneHot(index, length).cast('float32').mul(weight);

// Converte um produto em um vetor numérico que a rede neural consegue processar
// O vetor final concatena: [preço, idade_média, ...categorias_one_hot, ...cores_one_hot]
function encodeProduct(product, ctx) {
    // Normaliza o preço para [0,1] e aplica o peso de importância
    const price = tf.tensor1d([
        normalize(
            product.price,
            ctx.minPrice,
            ctx.maxPrice
        ) * WEIGHTS.price
    ])

    // Usa a média de idade dos compradores desse produto (normalizada), ou 0.5 se não há compras
    const age = tf.tensor1d([
        (ctx.productAvgAgeNorm[product.name] ?? 0.5) * WEIGHTS.age
    ]);

    // Codifica a categoria do produto como vetor one-hot pesado
    const category = oneHotWeighted(
        ctx.categoryIndex[product.category],
        ctx.numCategories,
        WEIGHTS.category
    );

    // Codifica a cor do produto como vetor one-hot pesado
    const color = oneHotWeighted(
        ctx.colorIndex[product.color],
        ctx.numColors,
        WEIGHTS.color
    );

    // Junta todos os sub-vetores em um único vetor 1D que representa o produto
    return tf.concat1d([price, age, category, color]);
}

// Converte um usuário em um vetor numérico que representa seu perfil de compras
function encodeUser(user, ctx) {
    if (user.purchases.length) {
        // Se o usuário tem histórico de compras, codifica cada produto comprado
        // e tira a média dos vetores — isso representa o "gosto médio" do usuário
        return tf.stack(
            user.purchases.map(
                p => encodeProduct(p, ctx)
            )
        )
            .mean(0)
            .reshape([1, ctx.dimensions]);
    }

    // Se o usuário não tem compras, usa somente a idade como sinal
    // O restante do vetor é zerado (sem histórico para inferir preferências)
    return tf.concat1d(
        [
            tf.zeros([1]), // preco é ignorado
            tf.tensor1d([normalize(user.age, ctx.minAge, ctx.maxAge) * WEIGHTS.age]),
            tf.zeros([ctx.numCategories]), // categorias são ignoradas
            tf.zeros([ctx.numColors]) // cores são ignoradas
        ]
    ).reshape([1, ctx.dimensions]);
}

// Monta os dados de treinamento: para cada par (usuário, produto),
// cria um exemplo de entrada com o vetor combinado e um rótulo (1 = comprou, 0 = não comprou)
function createTrainingData(ctx) {
    const input = []
    const labels = []

    ctx.users
        .filter(u => u.purchases.length) // considerar apenas usuários com compras
        .forEach(user => {
            const userVector = encodeUser(user, ctx).dataSync();
            ctx.products.forEach(product => {
                const productVector = encodeProduct(product, ctx).dataSync();

                // Rótulo: 1 se o usuário comprou esse produto, 0 caso contrário
                const label = user.purchases.some(p => p.name === product.name) ? 1 : 0;

                // Entrada da rede: concatenação do vetor do usuário com o vetor do produto
                input.push([...userVector, ...productVector]);
                labels.push(label);
            })
        });

    return {
        // xs: matriz de entrada (cada linha = um par usuário+produto)
        xs: tf.tensor2d(input),
        // ys: vetor de saída esperada (1 = comprou, 0 = não comprou)
        ys: tf.tensor2d(labels, [labels.length, 1]),
        // tamanho do vetor de entrada: vetor do usuário + vetor do produto
        inputDimensions: ctx.dimensions * 2,
    }
}

// Cria, configura e treina a rede neural com os dados preparados
async function configureNeuralNetAndTrain(trainData) {
    // Modelo sequencial: camadas empilhadas uma após a outra
    const model = tf.sequential();

    // Camada de entrada + primeira camada oculta: 128 neurônios com ReLU
    // ReLU elimina valores negativos, ajudando a rede a aprender padrões não-lineares
    model.add(tf.layers.dense({ inputShape: [trainData.inputDimensions], units: 128, activation: 'relu' }));
    // Segunda camada oculta: reduz para 64 neurônios, comprimindo a representação
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    // Terceira camada oculta: reduz ainda mais para 32 neurônios
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    // Camada de saída: 1 neurônio com sigmoid → gera uma probabilidade entre 0 e 1
    // (próximo de 1 = alta chance de o usuário comprar; próximo de 0 = baixa chance)
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    // Compila o modelo definindo:
    // - optimizer Adam com taxa de aprendizado 0.01: algoritmo que ajusta os pesos da rede
    // - binaryCrossentropy: função de perda ideal para classificação binária (comprou / não comprou)
    // - accuracy: métrica para acompanhar a precisão durante o treino
    model.compile({ optimizer: tf.train.adam(0.01), loss: 'binaryCrossentropy', metrics: ['accuracy'] });

    // Inicia o treinamento: 100 épocas, processando 32 exemplos por vez (batchSize)
    // A cada época concluída, envia um log (loss + accuracy) de volta para a página principal
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

// Orquestra todo o fluxo de treinamento: carrega dados, constrói contexto, treina e notifica
async function trainModel({ users }) {
    console.log('Training model with users:', users)

    // Notifica a página principal que o processo começou (50% de progresso)
    postMessage({
        type: workerEvents.progressUpdate, progress: { progress: 50 }
    });

    // Busca a lista de produtos do servidor
    const products = await (await fetch('/data/products.json')).json();

    // Monta o contexto com todos os dados normalizados e mapeamentos necessários
    const context = makeContext(products, users);

    // Pré-calcula e armazena os vetores de todos os produtos no contexto
    // Assim não precisamos recodificar os produtos a cada recomendação
    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: { ...product },
            vector: encodeProduct(product, context).dataSync()
        }
    });

    // Salva o contexto globalmente para ser reutilizado na função de recomendação
    _globalCtx = context;

    // Prepara os dados de treinamento e treina a rede neural
    const trainingData = createTrainingData(context);
    _model = await configureNeuralNetAndTrain(trainingData);
    console.log('products loaded:', products);

    // Notifica que o treinamento finalizou com sucesso
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}
// Gera recomendações para um usuário usando o modelo treinado
function recommend(user, ctx) {
    // Garante que o modelo foi treinado antes de tentar recomendar
    if (!_model) return;

    // Codifica o usuário em um vetor e combina com cada produto para formar os pares de entrada
    const userVector = encodeUser(user, ctx).dataSync();
    const inputs = ctx.productVectors.map(({ vector }) => [...userVector, ...vector]);

    // Transforma todos os pares em um tensor 2D e passa pelo modelo para obter predições
    const inputTensor = tf.tensor2d(inputs);
    const predictions = _model.predict(inputTensor);

    // Extrai os scores (probabilidades) e associa cada um ao produto correspondente
    const scores = predictions.dataSync();
    const recommendations = ctx.productVectors
        .map((item, index) => { return { ...item.meta, name: item.name, score: scores[index] } })

    // Ordena os produtos do maior para o menor score (mais relevante primeiro)
    const sortedItems = recommendations.sort((a, b) => b.score - a.score);

    // Envia a lista ordenada de recomendações de volta para a página principal
    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedItems
    });
}


// Tabela de ações: mapeia cada tipo de evento recebido à função que deve ser executada
const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

// Listener de mensagens do worker: recebe mensagens da página principal,
// identifica a ação solicitada e chama o handler correspondente
self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
