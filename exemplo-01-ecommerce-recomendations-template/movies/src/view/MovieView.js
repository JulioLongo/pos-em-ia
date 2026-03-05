import { View } from './View.js';

export class MovieView extends View {
    #movieList = document.querySelector('#movieList');
    #buttons;
    #movieTemplate;
    #onWatchMovie;

    constructor() {
        super();
        this.init();
    }

    async init() {
        this.#movieTemplate = await this.loadTemplate('./src/view/templates/movie-card.html');
    }

    onUserSelected(user) {
        // Habilita os botões quando um usuário está selecionado
        this.setButtonsState(user.id ? false : true);
    }

    registerWatchMovieCallback(callback) {
        this.#onWatchMovie = callback;
    }

    render(movies, disableButtons = true) {
        if (!this.#movieTemplate) return;

        const html = movies.map(movie => {
            return this.replaceTemplate(this.#movieTemplate, {
                id: movie.id,
                name: movie.name,
                genre: movie.genre,
                year: movie.year,
                rating: movie.rating,
                director: movie.director,
                movie: JSON.stringify(movie)
            });
        }).join('');

        this.#movieList.innerHTML = html;
        this.attachWatchButtonListeners();
        this.setButtonsState(disableButtons);
    }

    setButtonsState(disabled) {
        if (!this.#buttons) {
            this.#buttons = document.querySelectorAll('.watch-btn');
        }
        this.#buttons.forEach(button => {
            button.disabled = disabled;
        });
    }

    attachWatchButtonListeners() {
        this.#buttons = document.querySelectorAll('.watch-btn');
        this.#buttons.forEach(button => {
            button.addEventListener('click', () => {
                const movie = JSON.parse(button.dataset.movie);
                const originalHtml = button.innerHTML;

                button.innerHTML = '<i class="bi bi-check-circle-fill"></i> Adicionado!';
                button.classList.replace('btn-primary', 'btn-success');
                setTimeout(() => {
                    button.innerHTML = originalHtml;
                    button.classList.replace('btn-success', 'btn-primary');
                }, 500);

                this.#onWatchMovie(movie);
            });
        });
    }
}
