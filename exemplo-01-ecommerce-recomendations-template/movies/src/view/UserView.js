import { View } from './View.js';

export class UserView extends View {
    #userSelect = document.querySelector('#userSelect');
    #userAge = document.querySelector('#userAge');
    #pastWatchedList = document.querySelector('#pastWatchedList');

    #watchedTemplate;
    #onUserSelect;
    #onWatchRemove;
    #watchedElements = [];

    constructor() {
        super();
        this.init();
    }

    async init() {
        this.#watchedTemplate = await this.loadTemplate('./src/view/templates/watched-movie.html');
        this.attachUserSelectListener();
    }

    registerUserSelectCallback(callback) {
        this.#onUserSelect = callback;
    }

    registerWatchRemoveCallback(callback) {
        this.#onWatchRemove = callback;
    }

    renderUserOptions(users) {
        const options = users.map(user =>
            `<option value="${user.id}">${user.name}</option>`
        ).join('');
        this.#userSelect.innerHTML += options;
    }

    renderUserDetails(user) {
        this.#userAge.value = user.age;
    }

    renderWatchedMovies(watchedMovies) {
        if (!this.#watchedTemplate) return;

        if (!watchedMovies || watchedMovies.length === 0) {
            this.#pastWatchedList.innerHTML = '<p>Nenhum filme assistido.</p>';
            return;
        }

        const html = watchedMovies.map(movie =>
            this.replaceTemplate(this.#watchedTemplate, {
                ...movie,
                movie: JSON.stringify(movie)
            })
        ).join('');

        this.#pastWatchedList.innerHTML = html;
        this.attachWatchClickHandlers();
    }

    addWatchedMovie(movie) {
        if (this.#pastWatchedList.innerHTML.includes('Nenhum filme assistido')) {
            this.#pastWatchedList.innerHTML = '';
        }

        const html = this.replaceTemplate(this.#watchedTemplate, {
            ...movie,
            movie: JSON.stringify(movie)
        });

        this.#pastWatchedList.insertAdjacentHTML('afterbegin', html);

        const newest = this.#pastWatchedList.firstElementChild.querySelector('.past-watched');
        newest.classList.add('past-watched-highlight');
        setTimeout(() => newest.classList.remove('past-watched-highlight'), 1000);

        this.attachWatchClickHandlers();
    }

    attachUserSelectListener() {
        this.#userSelect.addEventListener('change', (event) => {
            const userId = event.target.value ? Number(event.target.value) : null;

            if (userId) {
                if (this.#onUserSelect) this.#onUserSelect(userId);
            } else {
                this.#userAge.value = '';
                this.#pastWatchedList.innerHTML = '';
            }
        });
    }

    attachWatchClickHandlers() {
        this.#watchedElements = [];
        const elements = document.querySelectorAll('.past-watched');

        elements.forEach(el => {
            this.#watchedElements.push(el);

            el.onclick = () => {
                const movie = JSON.parse(el.dataset.movie);
                const userId = this.getSelectedUserId();
                const wrapper = el.closest('.col-md-6');

                this.#onWatchRemove({ wrapper, userId, movie });

                wrapper.style.transition = 'opacity 0.5s ease';
                wrapper.style.opacity = '0';
                setTimeout(() => wrapper.remove(), 500);
            };
        });
    }

    getSelectedUserId() {
        return Number(this.#userSelect.value);
    }
}
