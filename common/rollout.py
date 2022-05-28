import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time
import cv2

class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args
        print('Init RolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, u_onehot, terminate, padded = [], [], [], [], [], []
        # o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        obs = self.env.reset()
        # copy_obs = []
        # for obs_agent in obs:
        #     copy_obs.append(obs_agent.flatten())
        # obs = copy_obs
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            # obs = self.env.get_obs()
            # state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):
                # avail_action = self.env.get_avail_agent_actions(agent_id)
                # action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                #                                        avail_action, evaluate)
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, evaluate)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                # avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            # reward, terminated, info = self.env.step(actions)
            act_for_env = [x.item() for x in actions]
            n_obs, reward, done, info = self.env.step(act_for_env)

            # copy_n_obs = []
            # for obs_agent in n_obs:
            #     copy_n_obs.append(obs_agent.flatten())

            # win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            # s.append(state)
            u.append(np.reshape([x.item() for x in actions], [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            # avail_u.append(avail_actions)
            r.append([sum(reward)])
            terminated = all(done)
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += sum(reward)
            step += 1
            obs = n_obs

        # last obs
        o.append(obs)
        o_next = o[1:]
        o = o[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append([np.zeros((self.obs_shape)) for _ in range(self.n_agents)])
            u.append(np.zeros([self.n_agents, 1]))
            r.append([0.])
            o_next.append([np.zeros((self.obs_shape)) for _ in range(self.n_agents)])
            # o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       o_next=o_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])

        # if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
        #     self.save_replay(self.env)
        #     self.env.close()
        return episode, episode_reward, win_tag, step

    def save_replay(self, num):
        state = self.env.reset()
        score = 0
        # self.env.render()
        # img = self.env.render(mode='rgb_array')
        # self.env.save_animation('out/temp.svg', egocentric_idx=None)
        # drawing = svg2rlg("out/temp.svg")
        # renderPM.drawToFile(drawing, "out/temp.png", fmt="PNG")
        # img = cv2.imread('out/temp.png')
        #
        # out_file = cv2.VideoWriter('out/video_{}.avi'.format(num + 1),
        #                            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, img.shape[:2][::-1])
        terminated = False
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        n_step = 1
        # self.env.save_animation('./renders/out-static.svg', egocentric_idx=None)
        while not terminated:
            self.env.render()
            # self.env.save_animation('out/temp.svg', egocentric_idx=None)
            # drawing = svg2rlg("out/temp.svg")
            # renderPM.drawToFile(drawing, "out/temp.png", fmt="PNG")
            # img = cv2.imread('out/temp.png')

            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # out_file.write(img)

            # time.sleep(0.1)  # to slow down the action for the video
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):
                action = self.agents.choose_action(state[agent_id], last_action[agent_id], agent_id, True)
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)

                last_action[agent_id] = action_onehot

            act_for_env = [x.item() for x in actions]

            n_obs, reward, done, info = self.env.step(act_for_env)
            # print(act_for_env, reward)

            score += sum(reward)
            terminated = all(done)

            state = n_obs
            n_step += 1

        # out_file.release()
        # self.env.save_animation('./renders/out.svg', egocentric_idx=None)


        return score
