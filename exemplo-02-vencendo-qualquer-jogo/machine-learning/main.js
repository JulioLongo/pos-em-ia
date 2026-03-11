// ============================================================================
// MAIN.JS — Orquestrador da IA que joga automaticamente
// ============================================================================
// Este é o ponto de entrada do módulo de machine learning.
// Ele conecta o jogo (PixiJS) ao Worker de IA (YOLOv5), criando um loop:
//   1. Captura a tela do jogo → 2. Envia para a IA → 3. Recebe a posição do alvo → 4. Clica automaticamente
// ============================================================================

// --- ETAPA 1: Importação do layout (HUD) ---
// Importa a função que cria a interface visual (Head-Up Display) com score e coordenadas
import { buildLayout } from "./layout";

// ============================================================================
// ETAPA 2: Função principal — inicializa a IA e conecta ao jogo
// ============================================================================
// Recebe o objeto "game" que contém a aplicação PixiJS, o stage (cena) e os controles
export default async function main(game) {

    // 2.1: Cria o HUD (painel visual com score e predições) e adiciona ao jogo
    const container = buildLayout(game.app);

    // 2.2: Cria um Web Worker para rodar a IA em uma thread separada
    // Isso evita que o processamento pesado do modelo trave a interface do jogo.
    // O "import.meta.url" garante que o caminho do worker.js seja resolvido corretamente
    const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

    // 2.3: Esconde a mira inicialmente (ela só aparece quando a IA detectar algo)
    game.stage.aim.visible = false;

    // ========================================================================
    // ETAPA 3: Recebendo predições da IA (Worker → Main Thread)
    // ========================================================================
    // Quando o Worker termina de processar um frame e encontra um alvo,
    // ele envia uma mensagem com as coordenadas. Este handler recebe essa mensagem.
    worker.onmessage = ({ data }) => {
        const { type, x, y } = data;

        if (type === 'prediction') {
            // 3.1: Loga no console a posição que a IA detectou
            console.log(`🎯 AI predicted at: (${x}, ${y})`);

            // 3.2: Atualiza o HUD com o score e as coordenadas da predição
            container.updateHUD(data);

            // 3.3: Torna a mira visível (a IA encontrou um alvo!)
            game.stage.aim.visible = true;

            // 3.4: Move a mira para a posição que a IA detectou (coordenadas locais)
            game.stage.aim.setPosition(data.x, data.y);

            // 3.5: Converte a posição local para posição global (coordenadas da tela)
            const position = game.stage.aim.getGlobalPosition();

            // 3.6: Simula um clique do mouse na posição detectada pela IA!
            // É aqui que a IA efetivamente "joga" — ela clica no alvo automaticamente
            game.handleClick({
                global: position,
            });

        }

    };

    // ========================================================================
    // ETAPA 4: Loop de captura — envia frames do jogo para a IA a cada 200ms
    // ========================================================================
    // Este setInterval cria um loop que periodicamente:
    //   1. Tira um "screenshot" do estado atual do jogo
    //   2. Envia para o Worker processar com o modelo YOLOv5
    setInterval(async () => {
        // 4.1: Captura o canvas (tela) atual do jogo como uma imagem
        // O renderer.extract.canvas() tira um "screenshot" do stage inteiro
        const canvas = game.app.renderer.extract.canvas(game.stage);

        // 4.2: Converte o canvas para ImageBitmap (formato otimizado para transferência)
        const bitmap = await createImageBitmap(canvas);

        // 4.3: Envia a imagem para o Worker processar
        // O segundo argumento [bitmap] é a "transfer list" — transfere a posse do bitmap
        // para o Worker (mais eficiente que copiar, pois usa zero-copy transfer)
        worker.postMessage({
            type: 'predict',
            image: bitmap,
        }, [bitmap]);

    }, 200); // Repete a cada 200ms (5 frames por segundo para a IA analisar)

    // Retorna o container do HUD para que o jogo possa manipulá-lo se necessário
    return container;
}
