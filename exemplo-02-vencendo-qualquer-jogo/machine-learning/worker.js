// ============================================================================
// WORKER.JS — Web Worker para detecção de objetos com YOLOv5
// ============================================================================
// Este arquivo roda em uma thread separada (Web Worker) para não travar a UI.
// Ele carrega o modelo YOLOv5 (rede neural de detecção de objetos) e processa
// frames do jogo para encontrar alvos ("kites") e retornar suas coordenadas.
// ============================================================================

// --- ETAPA 1: Importação do TensorFlow.js ---
// Carrega a biblioteca TensorFlow.js dentro do Web Worker.
// Como Workers não suportam "import", usamos importScripts (forma clássica).
// O TensorFlow.js permite rodar modelos de machine learning diretamente no navegador.
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest');

// --- ETAPA 2: Configurações e constantes ---
// Caminho para o modelo YOLOv5 convertido para formato web (TensorFlow.js)
const MODEL_PATH = `yolov5n_web_model/model.json`;
// Caminho para o arquivo com os nomes das classes que o modelo reconhece (ex: "kite", "person", etc.)
const LABELS_PATH = `yolov5n_web_model/labels.json`;
// Tamanho da imagem de entrada que o YOLOv5 espera (640x640 pixels)
const INPUT_MODEL_SIZE = 640;
// Limiar de confiança: só aceita detecções com score acima desse valor (0.4 = 40%)
// NOTA: há um bug aqui — "0, 4" deveria ser "0.4" (ponto ao invés de vírgula)
CLASS_THRESHOLD = 0, 4;

// --- ETAPA 3: Variáveis globais do Worker ---
// Array que vai guardar os nomes das classes (labels) carregadas do JSON
let _labels = [];
// Referência para o modelo carregado em memória
let _model = null;

// ============================================================================
// ETAPA 4: Carregamento do modelo YOLOv5
// ============================================================================
// Esta função é chamada uma única vez quando o Worker inicia.
// Ela prepara todo o pipeline de IA antes de começar a processar frames.
async function loadModel() {
    // 4.1: Aguarda o TensorFlow.js estar pronto (inicializa o backend WebGL/WASM)
    await tf.ready();

    // 4.2: Carrega os nomes das classes (labels) do arquivo JSON via fetch
    // Exemplo: ["person", "bicycle", "car", ..., "kite", ...]
    _labels = await (await fetch(LABELS_PATH)).json();

    // 4.3: Carrega o modelo YOLOv5 no formato GraphModel do TensorFlow.js
    // O model.json contém a arquitetura e referencia os pesos (weights)
    _model = await tf.loadGraphModel(MODEL_PATH);

    // 4.4: "Aquecimento" do modelo — roda uma inferência com dados falsos (tensor de 1s)
    // Isso força a compilação dos shaders WebGL, evitando lag na primeira predição real
    const dummyInput = tf.ones(_model.inputs[0].shape);
    await _model.executeAsync(dummyInput);
    tf.dispose(dummyInput); // Libera a memória do tensor de teste

    // 4.5: Avisa a thread principal que o modelo está pronto para uso
    postMessage({ type: 'model-loaded   ' });
}

// ============================================================================
// ETAPA 5: Execução da inferência (predição) no modelo
// ============================================================================
// Recebe um tensor (imagem pré-processada) e retorna as detecções brutas.
async function runInference(tensor) {
    // 5.1: Executa o modelo com o tensor de entrada — esta é a "predição" propriamente dita
    // O executeAsync é usado porque o modelo pode ter operações dinâmicas
    const output = await _model.executeAsync(tensor);
    // Libera o tensor de entrada da memória (não precisamos mais dele)
    tf.dispose(tensor);

    // 5.2: O YOLOv5 retorna múltiplos tensores de saída:
    // - boxes: coordenadas das caixas delimitadoras (bounding boxes) [x1, y1, x2, y2]
    // - scores: nível de confiança de cada detecção (0 a 1)
    // - classes: índice da classe detectada (ex: 0=person, 38=kite, etc.)
    const [boxes, scores, classes] = output.slice(0, 3);

    // 5.3: Converte os tensores GPU para arrays JavaScript normais (CPU)
    // Usamos Promise.all para fazer as 3 conversões em paralelo (mais rápido)
    const [boxesData, scoresData, classesData] = await Promise.all([
        boxes.data(),
        scores.data(),
        classes.data()
    ]);

    // 5.4: Libera todos os tensores de saída da memória GPU
    output.forEach(t => tf.dispose(t));

    // 5.5: Retorna os dados já como arrays JavaScript simples
    return { boxes: boxesData, scores: scoresData, classes: classesData };
}

// ============================================================================
// ETAPA 6: Processamento das predições (pós-processamento)
// ============================================================================
// Generator function (function*) que filtra e converte as detecções brutas
// em coordenadas úteis para o jogo. Usa "yield" para retornar uma por uma.
function* processPredictions({ boxes, scores, classes }, imageWidth, imageHeight) {
    for (let i = 0; i < scores.length; i++) {
        // 6.1: Filtra detecções com confiança abaixo do limiar
        if (scores[i] < CLASS_THRESHOLD) continue;

        // 6.2: Verifica se a classe detectada é "kite" (o alvo do jogo)
        // Ignora qualquer outro objeto (pessoas, carros, etc.)
        const label = _labels[classes[i]];
        if (label !== 'kite') continue;

        // 6.3: Extrai as coordenadas da bounding box (normalizadas entre 0 e 1)
        // Cada detecção tem 4 valores: x1, y1 (canto superior esquerdo) e x2, y2 (canto inferior direito)
        let [x1, y1, x2, y2] = boxes.slice(i * 4, (i + 1) * 4);

        // 6.4: Converte coordenadas normalizadas para pixels reais da imagem
        // Multiplica pela largura/altura original da imagem do jogo
        x1 *= imageWidth;
        x2 *= imageWidth;
        y1 *= imageHeight;
        y2 *= imageHeight;

        // 6.5: Calcula o centro da bounding box — é onde a IA vai "clicar"
        const boxWidth = x2 - x1;
        const boxHeight = y2 - y1;

        const centerX = x1 + boxWidth / 2;
        const centerY = y1 + boxHeight / 2;

        // 6.6: Retorna (yield) a predição com as coordenadas do centro e o score
        yield {
            x: centerX,
            y: centerY,
            score: (scores[i] * 100).toFixed(2), // Converte para porcentagem (ex: 0.95 → "95.00")
        };
    }
}

// ============================================================================
// ETAPA 7: Inicialização — carrega o modelo assim que o Worker é criado
// ============================================================================
loadModel();

// ============================================================================
// ETAPA 8: Listener de mensagens — recebe frames do jogo para processar
// ============================================================================
// A thread principal (main.js) envia mensagens com imagens do jogo.
// Este handler processa cada imagem e retorna as coordenadas dos alvos.
self.onmessage = async ({ data }) => {
    // 8.1: Ignora mensagens que não sejam do tipo 'predict'
    if (data.type !== 'predict') return

    // 8.2: Se o modelo ainda não carregou, avisa e não processa
    if (!_model) {
        postMessage({ type: 'model-not-loaded' });
        return;
    }

    // 8.3: Pré-processa a imagem recebida (bitmap) para o formato que o YOLOv5 espera
    const input = preprocessImage(data.image);

    // 8.4: Guarda as dimensões originais da imagem (usadas no pós-processamento)
    const { width, height } = data.image;

    // 8.5: Executa a inferência — passa a imagem pelo modelo e obtém as detecções
    const inferenceResults = await runInference(input);

    // --- Função auxiliar de pré-processamento ---
    // Converte a imagem bitmap em um tensor no formato esperado pelo YOLOv5:
    function preprocessImage(input) {
        return tf.tidy(() => { // tf.tidy limpa automaticamente tensores intermediários (evita vazamento de memória)
            const image = tf.browser.fromPixels(input)       // Converte imagem para tensor 3D [altura, largura, 3 canais RGB]
                .resizeBilinear([INPUT_MODEL_SIZE, INPUT_MODEL_SIZE]) // Redimensiona para 640x640 (tamanho que o modelo espera)
                .div(255.0)                                    // Normaliza pixels de [0-255] para [0-1] (padrão em redes neurais)
                .expandDims(0);                                // Adiciona dimensão de batch: [1, 640, 640, 3]
            return image;
        });
    }

    // 8.6: Itera sobre as predições filtradas e envia cada uma para a thread principal
    for (const prediction of processPredictions(inferenceResults, width, height)) {
        postMessage({
            type: 'prediction',
            ...prediction, // Envia { type: 'prediction', x, y, score }
        });
    }


};

console.log('🧠 YOLOv5n Web Worker initialized');
