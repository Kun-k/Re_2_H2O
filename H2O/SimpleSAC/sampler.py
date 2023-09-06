import numpy as np
from datetime import datetime

class StepSampler(object):

    def __init__(self, env, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0

    def sample(self, ego_policy, adv_policy, n_steps, deterministic=False,
               replay_buffer_ego=None, replay_buffer_adv=None, joint_noise_std=0.):
        observations = []
        actions_ego = []
        actions_adv = []
        rewards_ego = []
        rewards_adv = []
        next_observations = []
        dones = []
        self.env.traci_start()
        self._current_observation = self.env.reset()
        for _ in range(n_steps):
            self._traj_steps += 1
            observation = self._current_observation

            #TODO sample actions from current policy
            if ego_policy != None:
                action_ego = ego_policy(
                    np.expand_dims(observation, 0), deterministic=deterministic
                )[0, :]
            else:
                action_ego = None
            if adv_policy != None:
                action_adv = adv_policy(
                    np.expand_dims(observation, 0), deterministic=deterministic
                )[0, :]
            else:
                action_adv = None

            if joint_noise_std > 0.:
                # normal distribution
                next_observation, reward, done, _ = self.env.step(
                    action_ego + np.random.randn(action_ego.shape[0],) * joint_noise_std,
                    action_adv + np.random.randn(action_adv.shape[0],) * joint_noise_std)
            else:
                next_observation, reward, done, _ = self.env.step(action_ego, action_adv)

            observations.append(observation)
            actions_ego.append(action_ego)
            actions_adv.append(action_adv)
            rewards_ego.append(reward[0])
            rewards_adv.append(reward[1])
            dones.append(done)
            next_observations.append(next_observation)

            # add samples derived from current policy to replay buffer
            if replay_buffer_ego is not None:
                replay_buffer_ego.append(
                    observation, action_ego, reward[0], next_observation, done
                )
            if replay_buffer_adv is not None:
                replay_buffer_adv.append(
                    observation, action_adv, reward[1], next_observation, done
                )

            self._current_observation = next_observation

            if done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0
                self._current_observation = self.env.reset()
        self.env.traci_close()

        # return dict(
        #     observations=np.array(observations, dtype=np.float32),
        #     actions=np.array(actions, dtype=np.float32),
        #     rewards=np.array(rewards, dtype=np.float32),
        #     next_observations=np.array(next_observations, dtype=np.float32),
        #     dones=np.array(dones, dtype=np.float32),
        # )

    @property
    def env(self):
        return self._env

# with dones as a trajectory end indicator, we can use this sampler to sample trajectories
class TrajSampler(object):

    def __init__(self, env, rootsavepath=None, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env
        self.rootsavepath = rootsavepath

    def sample(self, ego_policy, adv_policy, n_trajs, scenario=None, deterministic=False,
               replay_buffer_ego=None, replay_buffer_adv=None):
        global info, done
        trajs = []
        self.env.traci_start()
        for _ in range(n_trajs):
            observations = []
            actions_ego = []
            actions_adv = []
            rewards_ego = []
            rewards_adv = []
            next_observations = []
            dones = []

            num_av_crash = 0
            num_bv_crash = 0
            # ego_col_cost = []
            # adv_col_cost = []
            # adv_road_cost = []
            # rewards_ego_col = []
            # rewards_ego_speed = []

            observation = self.env.reset(scenario)
            for _ in range(self.max_traj_length):
                if ego_policy != None:
                    action_ego = ego_policy(
                        np.expand_dims(observation, 0), deterministic=deterministic
                    )[0, :]
                else:
                    action_ego = None
                if adv_policy != None:
                    action_adv = adv_policy(
                        np.expand_dims(observation, 0), deterministic=deterministic
                    )[0, :]
                else:
                    action_adv = None
                next_observation, reward, done, info = self.env.step(action_ego, action_adv)

                observations.append(observation)
                actions_ego.append(action_ego)
                actions_adv.append(action_adv)
                rewards_ego.append(reward[0])
                rewards_adv.append(reward[1])
                dones.append(done)
                next_observations.append(next_observation)
                # ego_col_cost.append(info[1][0])
                # adv_col_cost.append(info[1][1])
                # adv_road_cost.append(info[1][2])
                # rewards_ego_col.append(info[1][5])
                # rewards_ego_speed.append(info[1][6])

                if replay_buffer_ego is not None:
                    replay_buffer_ego.add_sample(
                        observation, action_ego, reward[0], next_observation, done
                    )
                if replay_buffer_adv is not None:
                    replay_buffer_adv.add_sample(
                        observation, action_adv, reward[1], next_observation, done
                    )

                observation = next_observation
                if done:
                    break
            if info[0] == "AV crashed!":
                if self.rootsavepath != None:
                    self.env.record(filepath=self.rootsavepath + 'avcrash/record-' +
                                             datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f") + '.csv')
                num_av_crash += 1
            elif info[0] == "BV crashed!":
                if self.rootsavepath != None:
                    self.env.record(filepath=self.rootsavepath + 'bvcrash/record-' +
                                             datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f") + '.csv')
                num_bv_crash += 1
            elif info[0] == "AV arrived!" or not done:
                if self.rootsavepath != None:
                    self.env.record(filepath=self.rootsavepath + 'avarrive/record-' +
                                             datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f") + '.csv')

            traj_time = len(observations)
            traj_dis = observations[-1][0]
            collision_time = 0 if info[0] != "AV crashed!" else traj_time
            collision_dis = 0 if info[0] != "AV crashed!" else traj_dis

            trajs.append(dict(
                observations=np.array(observations, dtype=np.float32),
                actions_ego=np.array(actions_ego, dtype=np.float32),
                actions_adv=np.array(actions_adv, dtype=np.float32),
                rewards_ego=np.array(rewards_ego, dtype=np.float32),
                rewards_adv=np.array(rewards_adv, dtype=np.float32),
                next_observations=np.array(next_observations, dtype=np.float32),
                dones=np.array(dones, dtype=np.float32),
                metrics_av_crash=num_av_crash,
                metrics_bv_crash=num_bv_crash,
                traj_time=traj_time,
                traj_dis=traj_dis,
                collision_time=collision_time,
                collision_dis=collision_dis,
                # dis_av_crash=info[1][3],
                # dis_bv_crash=info[1][4],
                # ego_col_cost=np.array(ego_col_cost, dtype=np.float32),
                # adv_col_cost=np.array(adv_col_cost, dtype=np.float32),
                # adv_road_cost=np.array(adv_road_cost, dtype=np.float32),
                # rewards_ego_col=np.array(rewards_ego_col, dtype=np.float32),
                # rewards_ego_speed=np.array(rewards_ego_speed, dtype=np.float32),
            ))
        self.env.traci_close()

        return trajs

    @property
    def env(self):
        return self._env
