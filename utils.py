import json
from pathlib import Path, PosixPath
import shutil
import torch

from unityagents import UnityEnvironment


class Config:
    def __init__(self, n_agents=1):
        self.__env = None
        self.__results_dir = None
        self.__model_dir = None

        self.device = device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.n_agents = n_agents

        self.model_id_auto = True

        self.results_dir = Path('./results/')

        if self.model_id_auto:
            self.model_id = max([int(str(m).split(' ')[-1]) for m in self.results_dir.glob('model */')]) + 1
        else:
            self.model_id = 1

        self.model_dir = self.results_dir / f'model {self.model_id}'

        self.env_path = Path(f'./A{n_agents:02d}/') / 'Reacher.x86_64'
        self.env = UnityEnvironment(file_name=str(self.env_path))

        self.n_episodes = 400
        self.max_t = int(1e3)
        self.buffer_size = int(1e5)
        self.batch_size = 64
        self.gamma = 0.95
        self.tau = 1e-3
        self.lr_actor = 1e-3
        self.lr_critic = 1e-4
        self.critic_weight_decay = 0
        self.fc1_units = 256
        self.fc2_units = 128
        self.seed = 48
        self.n_print = 20
        self.target = 30
        self.window = 100

        self.target_episode = None
        self.target_score = None

    @property
    def env(self):
        return self.__env

    @env.setter
    def env(self, env):
        self.__env = env

        # get the default brain
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]

        # reset the environment
        self.env_info = env.reset(train_mode=True)[self.brain_name]

        # number of agents
        assert self.n_agents == len(self.env_info.agents)

        # examine the state space
        self.states = self.env_info.vector_observations
        self.state_dim = self.states.shape[1]

        # size of each action
        self.action_dim = self.brain.vector_action_space_size

    @property
    def results_dir(self):
        return self.__results_dir

    @results_dir.setter
    def results_dir(self, path):
        self.__results_dir = path
        try:
            path.mkdir()
        except FileExistsError:
            print(f'Path {path} already exists')

    @property
    def model_dir(self):
        return self.__model_dir

    @model_dir.setter
    def model_dir(self, path):
        self.__model_dir = path
        try:
            if path.exists() and not self.model_id_auto:
                shutil.rmtree(path)
                path.mkdir()
            else:
                path.mkdir()

        except FileExistsError:
            print(f'Path {path} already exists')

    def to_dict(self):
        excludes = ['env_info', 'brain', 'states']
        return {key: getattr(self, key) for key in vars(self) if not key.startswith('_') if key not in excludes}


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, PosixPath):
            return str(o)
        if isinstance(o, torch.device):
            return str(o)
        return json.JSONEncoder.default(self, o)
