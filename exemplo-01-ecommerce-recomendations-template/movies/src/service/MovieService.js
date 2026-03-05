export class MovieService {
    async getMovies() {
        const response = await fetch('./data/movies.json');
        return await response.json();
    }

    async getMovieById(id) {
        const movies = await this.getMovies();
        return movies.find(movie => movie.id === id);
    }
}
