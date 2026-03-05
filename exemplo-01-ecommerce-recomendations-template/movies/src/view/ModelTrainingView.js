import { View } from './View.js';

export class ModelView extends View {
    #trainModelBtn = document.querySelector('#trainModelBtn');
    #watchedArrow = document.querySelector('#watchedArrow');
    #watchedDiv = document.querySelector('#watchedDiv');
    #allUsersWatchedList = document.querySelector('#allUsersWatchedList');
    #runRecommendationBtn = document.querySelector('#runRecommendationBtn');
    #onTrainModel;
    #onRunRecommendation;

    constructor() {
        super();
        this.attachEventListeners();
    }

    registerTrainModelCallback(callback) {
        this.#onTrainModel = callback;
    }
    registerRunRecommendationCallback(callback) {
        this.#onRunRecommendation = callback;
    }

    attachEventListeners() {
        this.#trainModelBtn.addEventListener('click', () => this.#onTrainModel());
        this.#runRecommendationBtn.addEventListener('click', () => this.#onRunRecommendation());

        // Expand/collapse do painel de histórico geral de usuários
        this.#watchedDiv.addEventListener('click', () => {
            const list = this.#allUsersWatchedList;
            const isHidden = window.getComputedStyle(list).display === 'none';

            if (isHidden) {
                list.style.display = 'block';
                this.#watchedArrow.classList.replace('bi-chevron-down', 'bi-chevron-up');
            } else {
                list.style.display = 'none';
                this.#watchedArrow.classList.replace('bi-chevron-up', 'bi-chevron-down');
            }
        });
    }

    enableRecommendButton() {
        this.#runRecommendationBtn.disabled = false;
    }

    updateTrainingProgress(progress) {
        this.#trainModelBtn.disabled = true;
        this.#trainModelBtn.innerHTML =
            '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Treinando...';

        if (progress.progress === 100) {
            this.#trainModelBtn.disabled = false;
            this.#trainModelBtn.innerHTML = '<i class="bi bi-cpu"></i> Treinar Modelo';
        }
    }

    renderAllUsersWatched(users) {
        const html = users.map(user => {
            const watchedHtml = user.watched.map(m =>
                `<span class="badge bg-light text-dark me-1 mb-1">${m.name}</span>`
            ).join('');

            return `
                <div class="user-watched-summary">
                    <h6>${user.name} (${user.age} anos)</h6>
                    <div class="watched-badges">
                        ${watchedHtml || '<span class="text-muted">Nenhum filme assistido</span>'}
                    </div>
                </div>
            `;
        }).join('');

        this.#allUsersWatchedList.innerHTML = html;
    }
}
