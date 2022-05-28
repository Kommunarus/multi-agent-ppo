from runner import Runner
import gym
from pogema import GridConfig
from pogema.animation import AnimationMonitor
# from smac.env import StarCraft2Env
from common.arguments import get_mixer_args, get_common_args
import os

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    args = get_common_args()
    args = get_mixer_args(args)

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
    env = AnimationMonitor(env)
    env.reset()

    # env_info = env.get_env_info()
    args.n_actions = 5
    args.n_agents = num_agents
    args.state_shape = 1
    args.obs_shape = env.observation_space.shape
    args.episode_limit = max_episode_steps

    args.model_dir = './model_ppo_2cv_lot'
    args.result_dir = './model_ppo_2cv_lot'


    runner = Runner(env, args, dict_grid)
    runner.agents.policy.load_model('7')
    reward_rate = runner.evaluate_render()
    print('The win rate of {} is  {}'.format(args.alg, reward_rate))
    env.close()
