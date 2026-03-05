export class MovieController {
    #movieView;
    #currentUser = null;
    #events;
    #movieService;

    constructor({ movieView, events, movieService }) {
        this.#movieView = movieView;
        this.#movieService = movieService;
        this.#events = events;
        this.init();
    }

    static init(deps) {
        return new MovieController(deps);
    }

    async init() {
        this.setupCallbacks();
        this.setupEventListeners();
        const movies = await this.#movieService.getMovies();
        this.#movieView.render(movies, true);
    }

    setupEventListeners() {
        this.#events.onUserSelected((user) => {
            this.#currentUser = user;
            this.#movieView.onUserSelected(user);
            this.#events.dispatchRecommend(user);
        });

        this.#events.onRecommendationsReady(({ recommendations }) => {
            this.#movieView.render(recommendations, false);
        });
    }

    setupCallbacks() {
        this.#movieView.registerWatchMovieCallback(this.handleWatchMovie.bind(this));
    }

    async handleWatchMovie(movie) {
        const user = this.#currentUser;
        this.#events.dispatchWatchAdded({ user, movie });
    }
}
