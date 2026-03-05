import { UserController } from './controller/UserController.js';
import { MovieController } from './controller/MovieController.js';
import { ModelController } from './controller/ModelTrainingController.js';
import { TFVisorController } from './controller/TFVisorController.js';
import { TFVisorView } from './view/TFVisorView.js';
import { UserService } from './service/UserService.js';
import { MovieService } from './service/MovieService.js';
import { UserView } from './view/UserView.js';
import { MovieView } from './view/MovieView.js';
import { ModelView } from './view/ModelTrainingView.js';
import Events from './events/events.js';
import { WorkerController } from './controller/WorkerController.js';

// Serviços compartilhados
const userService = new UserService();
const movieService = new MovieService();

// Views
const userView = new UserView();
const movieView = new MovieView();
const modelView = new ModelView();
const tfVisorView = new TFVisorView();

// Cria o Web Worker responsável pelo treinamento e recomendação
const mlWorker = new Worker('/src/workers/movieTrainingWorker.js', { type: 'module' });

// Inicializa o controlador do worker e dispara o treinamento automático ao carregar
const w = WorkerController.init({ worker: mlWorker, events: Events });

const users = await userService.getDefaultUsers();
w.triggerTrain(users);

// Inicializa os demais controladores
ModelController.init({ modelView, userService, events: Events });

TFVisorController.init({ tfVisorView, events: Events });

MovieController.init({ movieView, userService, movieService, events: Events });

const userController = UserController.init({ userView, userService, movieService, events: Events });

// Adiciona um usuário "novo" sem histórico para demonstrar recomendação a frio
userController.renderUsers({
    id: 99,
    name: "Novo Espectador",
    age: 30,
    watched: []
});
