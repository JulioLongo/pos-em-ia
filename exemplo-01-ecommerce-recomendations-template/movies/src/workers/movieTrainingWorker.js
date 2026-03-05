// Importa o TensorFlow.js diretamente da CDN para usar dentro do Web Worker
import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
// Importa os nomes dos eventos usados para comunicação entre o worker e a página principal
import { workerEvents } from '../events/constants.js';

// Contexto global preenchido após o treinamento (filmes, vetores pré-calculados, índices)
let _globalCtx = {};
// Modelo de rede neural treinado
let _model = null;

// Pesos que definem a importância de cada característica na codificação dos filmes
// Gênero tem maior influência (0.4), diretor em seguida (0.3), nota e idade do espectador têm menor peso
const WEIGHTS = {
    genre: 0.4,
    director: 0.3,
    rating: 0.2,
    age: 0.1
};

// Normaliza um valor para o intervalo [0, 1] com base no mínimo e máximo do conjunto
const normalize = (value, min, max) => (value - min) / (max - min);

// ─────────────────────────────────────────────
// ETAPA 1: Construir o contexto de pré-processamento
// ─────────────────────────────────────────────

/**
 * Monta o "contexto" global com todas as informações pré-calculadas necessárias
 * para codificar filmes e usuários numericamente.
 */
function makeContext(movies, users) {
    // Extrai anos e notas para calcular os limites de normalização
    const years = movies.map(m => m.year);
    const ratings = movies.map(m => m.rating);

    const minYear = Math.min(...years);
    const maxYear = Math.max(...years);
    const minRating = Math.min(...ratings);
    const maxRating = Math.max(...ratings);

    // Listas de gêneros e diretores únicos encontrados no catálogo
    const genres = [...new Set(movies.map(m => m.genre))];
    const directors = [...new Set(movies.map(m => m.director))];

    // Mapeia cada gênero e diretor para um índice numérico (necessário para one-hot encoding)
    const genreIndex = Object.fromEntries(genres.map((g, i) => [g, i]));
    const directorIndex = Object.fromEntries(directors.map((d, i) => [d, i]));

    // Calcula a média de idade dos espectadores de cada filme para usar como feature demográfica
    const ages = users.map(u => u.age);
    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);
    const midAge = (minAge + maxAge) / 2;

    const ageSums = {};
    const ageCounts = {};

    // Soma as idades e conta os espectadores por filme
    users.forEach(user => {
        user.watched.forEach(m => {
            ageSums[m.name] = (ageSums[m.name] || 0) + user.age;
            ageCounts[m.name] = (ageCounts[m.name] || 0) + 1;
        });
    });

    // Para cada filme, calcula a idade média dos espectadores normalizada para [0, 1]
    // Se ninguém assistiu ainda, usa a idade média do dataset como estimativa
    const movieAvgAgeNorm = Object.fromEntries(
        movies.map(movie => {
            const avg = ageCounts[movie.name]
                ? ageSums[movie.name] / ageCounts[movie.name]
                : midAge;
            return [movie.name, normalize(avg, minAge, maxAge)];
        })
    );

    return {
        movies,
        users,
        genreIndex,
        directorIndex,
        minYear, maxYear,
        minRating, maxRating,
        minAge, maxAge,
        numGenres: genres.length,
        numDirectors: directors.length,
        // Tamanho total do vetor: 1 (nota) + 1 (idade média espectadores) + nº gêneros + nº diretores
        dimensions: 2 + genres.length + directors.length,
        movieAvgAgeNorm
    };
}

// ─────────────────────────────────────────────
// ETAPA 2: Codificação de filmes e usuários em vetores
// ─────────────────────────────────────────────

// Cria um vetor one-hot pesado: todos os valores são 0 exceto o índice correspondente
// Ex: gênero "ação" (índice 1 de 7) com peso 0.4 → [0, 0.4, 0, 0, 0, 0, 0]
const oneHotWeighted = (index, length, weight) =>
    tf.oneHot(index, length).cast('float32').mul(weight);

/**
 * Converte um filme em um vetor numérico 1D que a rede neural consegue processar.
 * Vetor final: [nota, idade_média_espectadores, ...gêneros_one_hot, ...diretores_one_hot]
 */
function encodeMovie(movie, ctx) {
    // Nota do filme normalizada e pesada
    const rating = tf.tensor1d([
        normalize(movie.rating, ctx.minRating, ctx.maxRating) * WEIGHTS.rating
    ]);

    // Idade média dos espectadores desse filme (ou 0.5 como fallback), pesada
    const age = tf.tensor1d([
        (ctx.movieAvgAgeNorm[movie.name] ?? 0.5) * WEIGHTS.age
    ]);

    // Gênero codificado como one-hot pesado
    const genre = oneHotWeighted(
        ctx.genreIndex[movie.genre],
        ctx.numGenres,
        WEIGHTS.genre
    );

    // Diretor codificado como one-hot pesado
    const director = oneHotWeighted(
        ctx.directorIndex[movie.director],
        ctx.numDirectors,
        WEIGHTS.director
    );

    // Concatena todos os sub-vetores em um único vetor 1D representando o filme
    return tf.concat1d([rating, age, genre, director]);
}

/**
 * Converte um usuário em um vetor numérico que representa seu perfil de gosto cinematográfico.
 */
function encodeUser(user, ctx) {
    if (user.watched.length) {
        // Se tem histórico: tira a média dos vetores dos filmes assistidos
        // → representa o "gosto médio" do usuário
        return tf.stack(
            user.watched.map(m => encodeMovie(m, ctx))
        )
            .mean(0)
            .reshape([1, ctx.dimensions]);
    }

    // Sem histórico: usa apenas a idade como sinal demográfico
    // O restante do vetor é zerado (sem histórico para inferir preferências)
    return tf.concat1d([
        tf.zeros([1]),                                                                     // nota ignorada
        tf.tensor1d([normalize(user.age, ctx.minAge, ctx.maxAge) * WEIGHTS.age]),         // só idade
        tf.zeros([ctx.numGenres]),                                                         // gêneros ignorados
        tf.zeros([ctx.numDirectors])                                                       // diretores ignorados
    ]).reshape([1, ctx.dimensions]);
}

// ─────────────────────────────────────────────
// ETAPA 3: Preparação dos dados de treinamento
// ─────────────────────────────────────────────

/**
 * Para cada par (usuário, filme), cria um exemplo de entrada com o vetor combinado
 * e um rótulo binário: 1 = usuário assistiu esse filme, 0 = não assistiu.
 */
function createTrainingData(ctx) {
    const input = [];
    const labels = [];

    ctx.users
        .filter(u => u.watched.length) // considera apenas usuários com histórico
        .forEach(user => {
            const userVector = encodeUser(user, ctx).dataSync();
            ctx.movies.forEach(movie => {
                const movieVector = encodeMovie(movie, ctx).dataSync();
                const label = user.watched.some(m => m.name === movie.name) ? 1 : 0;

                // Entrada da rede: concatenação do vetor do usuário com o vetor do filme
                input.push([...userVector, ...movieVector]);
                labels.push(label);
            });
        });

    return {
        xs: tf.tensor2d(input),                            // matriz de entrada
        ys: tf.tensor2d(labels, [labels.length, 1]),       // rótulos (1=assistiu, 0=não assistiu)
        inputDimensions: ctx.dimensions * 2,               // vetor usuário + vetor filme
    };
}

// ─────────────────────────────────────────────
// ETAPA 4: Configuração e treinamento da rede neural
// ─────────────────────────────────────────────

/**
 * Cria, compila e treina a rede neural com os dados preparados.
 * Arquitetura: 3 camadas ocultas (128→64→32) com ReLU + saída sigmoid para classificação binária.
 */
async function configureNeuralNetAndTrain(trainData) {
    const model = tf.sequential();

    // Camadas ocultas com ReLU: aprendem padrões não-lineares de compatibilidade usuário-filme
    model.add(tf.layers.dense({ inputShape: [trainData.inputDimensions], units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));

    // Saída sigmoid: gera probabilidade entre 0 e 1 (alta = usuário provavelmente vai gostar)
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    // Adam com lr=0.01 + binaryCrossentropy: ideal para classificação binária (assistiu/não assistiu)
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    // Treina por 100 épocas, enviando logs de progresso a cada época para a UI
    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    });

    return model;
}

// ─────────────────────────────────────────────
// ETAPA 5: Persistência dos vetores no pgvector
// ─────────────────────────────────────────────

/**
 * Após o treinamento, envia os vetores de filmes e usuários para o servidor Express,
 * que os persiste no PostgreSQL com a extensão pgvector.
 * Isso permite buscas de similaridade eficientes no banco na hora de recomendar.
 */
async function saveVectorsToDB(ctx) {
    // Serializa vetores de filmes (Float32Array → Array comum para JSON)
    const movieVectors = ctx.movieVectors.map(mv => ({
        movie_id: mv.meta.id,
        name: mv.name,
        meta: mv.meta,
        vector: Array.from(mv.vector)
    }));

    // Serializa vetores de usuários calculados a partir do histórico de filmes assistidos
    const userVectors = ctx.users.map(user => ({
        user_id: user.id,
        name: user.name,
        age: user.age,
        watched: user.watched,
        vector: Array.from(encodeUser(user, ctx).dataSync())
    }));

    // Envia para o servidor; o servidor cria as tabelas e insere os registros no pgvector
    const res = await fetch('/api/vectors/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ movieVectors, userVectors, dimensions: ctx.dimensions })
    });

    if (!res.ok) {
        const err = await res.json();
        console.warn('Aviso: falha ao salvar vetores no banco.', err.error);
    } else {
        console.log('Vetores persistidos no pgvector com sucesso.');
    }
}

// ─────────────────────────────────────────────
// ETAPA 6: Orquestração do treinamento completo
// ─────────────────────────────────────────────

/**
 * Ponto de entrada do treinamento: busca filmes, monta contexto, treina a rede
 * e ao final persiste os vetores no banco para uso em recomendações futuras.
 */
async function trainModel({ users }) {
    console.log('Iniciando treinamento com', users.length, 'usuários');

    // Notifica a UI que o processo começou (50% de progresso)
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    // Busca o catálogo de filmes do servidor
    const movies = await (await fetch('/data/movies.json')).json();

    // Monta o contexto com todos os dados normalizados e mapeamentos necessários
    const context = makeContext(movies, users);

    // Pré-calcula e armazena os vetores de todos os filmes do catálogo
    // Evita recodificar os filmes a cada chamada de recomendação
    context.movieVectors = movies.map(movie => ({
        name: movie.name,
        meta: { ...movie },
        vector: encodeMovie(movie, context).dataSync()
    }));

    // Salva o contexto globalmente para reutilização na função de recomendação
    _globalCtx = context;

    // Prepara os dados de treinamento e treina a rede neural
    const trainingData = createTrainingData(context);
    _model = await configureNeuralNetAndTrain(trainingData);

    // Persiste os vetores no pgvector para buscas de similaridade rápidas
    await saveVectorsToDB(context);

    // Notifica a UI que o treinamento foi concluído
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}

// ─────────────────────────────────────────────
// ETAPA 7: Recomendação com busca de vizinhos no pgvector
// ─────────────────────────────────────────────

/**
 * Gera recomendações para um usuário usando busca vetorial + modelo neural:
 *
 * 1. Codifica o usuário em um vetor
 * 2. Consulta o pgvector para encontrar os 100 usuários mais similares
 * 3. Coleta os filmes assistidos por esses usuários como candidatos
 * 4. Roda o modelo neural para predizer qual candidato o usuário vai gostar
 * 5. Retorna a lista ordenada por score descendente
 */
async function recommend(user, ctx) {
    if (!_model) return;

    // Codifica o usuário em um vetor (Float32Array para os inputs, Array para o JSON)
    const userVecTyped = encodeUser(user, ctx).dataSync();
    const userVecArray = Array.from(userVecTyped);

    // Busca os 100 usuários com vetor mais próximo ao do usuário atual no pgvector
    let similarUsers = [];
    try {
        const res = await fetch('/api/similar-users', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ vector: userVecArray, limit: 100 })
        });
        const data = await res.json();
        similarUsers = data.similarUsers ?? [];
    } catch (err) {
        console.warn('Não foi possível buscar usuários similares no banco. Usando catálogo completo.', err.message);
    }

    // Monta o pool de filmes candidatos a partir do histórico dos usuários similares
    // Usa um Map para garantir unicidade pelo nome do filme
    const candidateMap = new Map();
    similarUsers.forEach(su => {
        su.watched.forEach(m => {
            if (!candidateMap.has(m.name)) candidateMap.set(m.name, m);
        });
    });

    // Fallback: se nenhum vizinho foi encontrado (banco vazio, erro de conexão, etc.)
    // usa o catálogo completo como candidatos
    if (candidateMap.size === 0) {
        ctx.movieVectors.forEach(mv => candidateMap.set(mv.name, mv.meta));
    }

    // Para cada filme candidato, combina o vetor do usuário com o vetor do filme
    // para formar o input da rede neural de predição
    const candidates = [...candidateMap.values()];
    const inputs = candidates.map(movie => {
        const existing = ctx.movieVectors.find(mv => mv.name === movie.name);
        // Usa vetor pré-calculado se disponível; caso contrário recodifica o filme
        const movieVec = existing
            ? existing.vector
            : encodeMovie(movie, ctx).dataSync();
        return [...userVecTyped, ...movieVec];
    });

    // Roda o modelo sobre todos os pares (usuário, filme candidato)
    const inputTensor = tf.tensor2d(inputs);
    const predictions = _model.predict(inputTensor);
    const scores = predictions.dataSync();

    // Associa cada score ao respectivo filme e ordena do maior para o menor
    const recommendations = candidates
        .map((movie, i) => ({ ...movie, score: scores[i] }))
        .sort((a, b) => b.score - a.score);

    // Envia a lista ordenada de recomendações de volta para a página principal
    postMessage({ type: workerEvents.recommend, user, recommendations });
}

// ─────────────────────────────────────────────
// Roteamento de mensagens do worker
// ─────────────────────────────────────────────

// Tabela de ações: mapeia cada tipo de evento recebido à função correspondente
const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

// Listener principal: recebe mensagens da página, identifica a ação e chama o handler
self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
