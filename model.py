import numpy as np
from agent.agent import Agents

import gym
from pogema import GridConfig
import enum

class args():
    n_actions = 5
    n_agents = 200
    state_shape = 1
    obs_shape = (3, 11, 11)
    episode_limit = 256
    use_gpu = False
    rnn_hidden_dim = 64
    lr = 5e-4
    lr_actor = 5e-4
    lr_critic = 5e-4
    train_steps = 1
    save_cycle = 900
    grad_norm_clip = 10
    n_episodes = 32
    ppo_n_epochs = 15
    lamda = 0.95
    clip_param = 0.2
    entropy_coeff = 0.01
    reuse_network = True
    last_action = True
    alg = 'mappo'
    model_dir = './model_ppo_2cv_lot/'
    map = '3m'
    optimizer = 'ADAM'



class Model:

    def __init__(self):
        self.agents = Agents(args)
        self.agents.policy.load_model('1')
        self.agents.policy.init_hidden(1)

        self.last_action = np.zeros((200, 5))


    def act(self, obs, dones, positions_xy, targets_xy) -> list:
        n_agents = len(obs)
        actions = []
        for agent_id in range(min(200, n_agents)):
            action = self.agents.choose_action(obs[agent_id], self.last_action[agent_id], agent_id, True)
            action_onehot = np.zeros(5)
            action_onehot[action.item()] = 1
            actions.append(action.item())
            self.last_action[agent_id] = action_onehot


        for agent_id in range(200, n_agents):
            actions.append(0)

        return actions

if __name__ == '__main__':
    classs = Model()

    num_agents = 6
    size = 30
    max_episode_steps = 256
    density = 0.3
    seed = None
    obs_radius = 5

    dict_grid = {'num_agents':num_agents,  # количество агентов на карте
                'size':size,  # размеры карты
                'density':density,  # плотность препятствий
                'seed':seed,  # сид генерации задания
                'max_episode_steps':max_episode_steps,  # максимальная длина эпизода
                'obs_radius':obs_radius
    }
    grid_config = GridConfig(**dict_grid)
    env = gym.make("Pogema-v0", grid_config=grid_config)

    score = 0
    for game in range(32):
        obs = env.reset()
        for step in range(max_episode_steps):
            act = classs.act(obs, 0, 0, 0)
            n_obs, rew, done, _ = env.step(act)
            obs = n_obs
            # env.render()
            score += sum(rew)

    print(score/32)



