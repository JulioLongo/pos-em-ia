export class UserController {
    #userService;
    #userView;
    #events;

    constructor({ userView, userService, events }) {
        this.#userView = userView;
        this.#userService = userService;
        this.#events = events;
    }

    static init(deps) {
        return new UserController(deps);
    }

    async renderUsers(nonTrainedUser) {
        const users = await this.#userService.getDefaultUsers();

        this.#userService.addUser(nonTrainedUser);
        const allUsers = [nonTrainedUser, ...users];

        this.#userView.renderUserOptions(allUsers);
        this.setupCallbacks();
        this.setupWatchObserver();

        this.#events.dispatchUsersUpdated({ users: allUsers });
    }

    setupCallbacks() {
        this.#userView.registerUserSelectCallback(this.handleUserSelect.bind(this));
        this.#userView.registerWatchRemoveCallback(this.handleWatchRemove.bind(this));
    }

    setupWatchObserver() {
        this.#events.onWatchAdded(async (...data) => this.handleWatchAdded(...data));
    }

    async handleUserSelect(userId) {
        const user = await this.#userService.getUserById(userId);
        this.#events.dispatchUserSelected(user);
        return this.displayUserDetails(user);
    }

    async handleWatchAdded({ user, movie }) {
        const updatedUser = await this.#userService.getUserById(user.id);
        updatedUser.watched.push({ ...movie });
        await this.#userService.updateUser(updatedUser);

        const lastWatched = updatedUser.watched[updatedUser.watched.length - 1];
        this.#userView.addWatchedMovie(lastWatched);

        this.#events.dispatchUsersUpdated({ users: await this.#userService.getUsers() });
    }

    async handleWatchRemove({ userId, movie }) {
        const user = await this.#userService.getUserById(userId);
        const index = user.watched.findIndex(m => m.id === movie.id);

        if (index !== -1) {
            user.watched.splice(index, 1);
            await this.#userService.updateUser(user);

            const updatedUsers = await this.#userService.getUsers();
            this.#events.dispatchUsersUpdated({ users: updatedUsers });
        }
    }

    async displayUserDetails(user) {
        this.#userView.renderUserDetails(user);
        this.#userView.renderWatchedMovies(user.watched);
    }

    getSelectedUserId() {
        return this.#userView.getSelectedUserId();
    }
}
