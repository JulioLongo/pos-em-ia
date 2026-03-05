export class UserService {
    #storageKey = 'movies-recommendation-users';

    async getDefaultUsers() {
        const response = await fetch('./data/users.json');
        const users = await response.json();
        this.#setStorage(users);
        return users;
    }

    async getUsers() {
        return this.#getStorage();
    }

    async getUserById(userId) {
        const users = this.#getStorage();
        return users.find(user => user.id === userId);
    }

    async updateUser(user) {
        const users = this.#getStorage();
        const index = users.findIndex(u => u.id === user.id);
        users[index] = { ...users[index], ...user };
        this.#setStorage(users);
        return users[index];
    }

    async addUser(user) {
        const users = this.#getStorage();
        this.#setStorage([user, ...users]);
    }

    #getStorage() {
        const data = sessionStorage.getItem(this.#storageKey);
        return data ? JSON.parse(data) : [];
    }

    #setStorage(data) {
        sessionStorage.setItem(this.#storageKey, JSON.stringify(data));
    }
}
