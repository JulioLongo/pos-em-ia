import { View } from './View.js';

export class TFVisorView extends View {
    #lossPoints = [];
    #accPoints = [];

    constructor() {
        super();
        tfvis.visor().open();
    }

    resetDashboard() {
        this.#lossPoints = [];
        this.#accPoints = [];
    }

    handleTrainingLog(log) {
        const { epoch, loss, accuracy } = log;
        this.#lossPoints.push({ x: epoch, y: loss });
        this.#accPoints.push({ x: epoch, y: accuracy });

        tfvis.render.linechart(
            {
                name: 'Precisão do Modelo',
                tab: 'Treinamento',
                style: { display: 'inline-block', width: '49%' }
            },
            { values: this.#accPoints, series: ['precisão'] },
            { xLabel: 'Época', yLabel: 'Precisão (%)', height: 300 }
        );

        tfvis.render.linechart(
            {
                name: 'Erro de Treinamento',
                tab: 'Treinamento',
                style: { display: 'inline-block', width: '49%' }
            },
            { values: this.#lossPoints, series: ['erro'] },
            { xLabel: 'Época', yLabel: 'Valor do Erro', height: 300 }
        );
    }
}
