// ============================================================================
// LAYOUT.JS — Interface visual (HUD) para exibir informações da IA
// ============================================================================
// Cria um painel visual usando PixiJS (biblioteca de renderização 2D/WebGL)
// que mostra o score de confiança da IA e as coordenadas da última detecção.
// O HUD fica posicionado no canto superior direito da tela do jogo.
// ============================================================================

import * as PIXI from 'pixi.js';

// --- Função principal: constrói e retorna o HUD ---
export function buildLayout(app) {

    // --- ETAPA 1: Criação do container principal do HUD ---
    // Container é um agrupador de elementos visuais no PixiJS
    const hud = new PIXI.Container();
    hud.y = 50;          // Posição vertical (50px do topo)
    hud.zIndex = 1000;   // Garante que o HUD fique acima de todos os outros elementos

    // --- ETAPA 2: Texto do Score ---
    // Mostra o score de confiança da última detecção da IA
    const scoreText = new PIXI.Text({
        text: 'Score: 0',
        style: {
            fontFamily: 'monospace',   // Fonte monoespaçada (fácil de ler)
            fontSize: 24,
            fill: 0xffffff,            // Cor do texto: branco
            stroke: 0x000000,          // Contorno preto (para legibilidade sobre qualquer fundo)
        }
    });
    hud.addChild(scoreText); // Adiciona o texto como filho do container HUD

    // --- ETAPA 3: Texto das Predições ---
    // Mostra as coordenadas (x, y) de onde a IA detectou o alvo
    const predictionsText = new PIXI.Text({
        text: 'Predictions:',
        style: {
            fontFamily: 'monospace',
            fontSize: 16,
            fill: 0xfff666,            // Cor amarela (destaque)
            stroke: 0x333300,          // Contorno escuro
            wordWrap: true,            // Quebra de linha automática
            wordWrapWidth: 420,        // Largura máxima antes de quebrar linha
        }
    });
    predictionsText.y = 36; // Posiciona abaixo do scoreText
    hud.addChild(predictionsText);

    // --- ETAPA 4: Adiciona o HUD ao stage (cena) do jogo ---
    app.stage.sortableChildren = true; // Permite ordenação por zIndex (HUD sempre no topo)
    app.stage.addChild(hud);

    // --- ETAPA 5: Função de posicionamento (canto superior direito) ---
    // Recalcula a posição horizontal do HUD baseado na largura da tela
    function positionHUD() {
        const margin = 16; // Margem da borda direita
        // Calcula a largura do HUD (pega o maior entre os dois textos)
        const hudWidth = Math.max(scoreText.width, predictionsText.width);
        // Posiciona o HUD no canto direito: largura da tela - largura do HUD - margem
        hud.x = app.renderer.width - hudWidth - margin;
    }

    // --- ETAPA 6: Função para atualizar os dados exibidos no HUD ---
    // Chamada toda vez que a IA faz uma nova detecção
    function updateHUD(data) {
        scoreText.text = `Score: ${data.score}`;                                    // Atualiza o score
        predictionsText.text = `Predictions: (${Math.round(data.x)}, ${Math.round(data.y)})`; // Atualiza coordenadas
        positionHUD(); // Reposiciona (largura do texto pode ter mudado)
    }

    // --- ETAPA 7: Posiciona o HUD inicialmente e reposiciona ao redimensionar a janela ---
    positionHUD();
    window.addEventListener('resize', () => {
        positionHUD();
    });

    // --- ETAPA 8: Retorna a interface pública do módulo ---
    // Expõe apenas a função updateHUD para que o main.js possa atualizar os dados
    return {
        updateHUD,
    };
}
