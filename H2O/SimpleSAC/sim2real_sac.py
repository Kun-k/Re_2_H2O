import torch
import torch.nn.functional as F
from ml_collections import ConfigDict
from .model import Scalar, soft_target_update
from .utils_h2o import prefix_metrics


class Sim2realSAC(object):

    @staticmethod
    def get_default_config(updates=None, device='cuda', cql_min_q_weight=0.1):
        config = ConfigDict()
        config.batch_size = 256
        config.device = device
        config.discount = 0.99
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.backup_entropy = False
        config.target_entropy = 0.0
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.d_sa_lr = 3e-4
        config.d_sas_lr = 3e-4
        config.d_early_stop_steps = 1000000
        config.noise_std_discriminator = 0.1
        config.optimizer_type = 'adam'
        config.soft_target_update_rate = 5e-3
        config.target_update_period = 1
        config.use_cql = True
        config.use_variant = False
        config.u_ablation = False
        config.use_td_target_ratio = True
        config.use_sim_q_coeff = True
        config.use_kl_baseline = False
        config.fix_baseline_steps = 10
        # kl divergence: E_pM log(pM/pM^)
        config.sim_q_coeff_min = 1e-45
        config.sim_q_coeff_max = 10
        config.sampling_n_next_states = 10
        config.s_prime_std_ratio = 1.
        config.cql_n_actions = 10
        config.cql_importance_sample = True
        config.cql_lagrange = False
        config.cql_target_action_gap = 1.0
        config.cql_temp = 1.0
        config.cql_min_q_weight = cql_min_q_weight
        config.cql_max_target_backup = False
        config.cql_clip_diff_min = -1000
        config.cql_clip_diff_max = 1000
        # pM/pM^
        config.clip_dynamics_ratio_min = 1e-5
        config.clip_dynamics_ratio_max = 1
        config.sa_prob_clip = 0.0

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, policy, qf1, qf2, target_qf1, target_qf2, d_sa, d_sas, replay_buffer,
                 device='cuda', dynamics_model=None, is_real_q_lip=False):
        self.config = Sim2realSAC.get_default_config(config, device=device)
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.d_sa = d_sa
        self.d_sas = d_sas
        self.replay_buffer = replay_buffer
        self.mean, self.std = self.replay_buffer.get_mean_std()
        self.next_observation_sampler = dynamics_model
        self.kl_baseline = 1

        '''
        Optimizers
        '''
        optimizer_class = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }[self.config.optimizer_type]

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(), self.config.policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), self.config.qf_lr
        )

        self.d_sa_optimizer = optimizer_class(self.d_sa.parameters(), self.config.d_sa_lr)
        self.d_sas_optimizer = optimizer_class(self.d_sas.parameters(), self.config.d_sas_lr)

        # whether to use automatic entropy tuning (True in default)
        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = optimizer_class(
                self.log_alpha.parameters(),
                lr=self.config.policy_lr,
            )
        else:
            self.log_alpha = None

        # whether to use the lagrange version of CQL: False by default
        if self.config.cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            self.alpha_prime_optimizer = optimizer_class(
                self.log_alpha_prime.parameters(),
                lr=self.config.qf_lr,
            )

        self.update_target_network(1.0)
        self._total_steps = 0

        self.tau_lip = 1
        self.threshold_lip = 10
        self.is_sim_q_lip = False
        self.is_target_lip = False
        self.is_real_q_lip = is_real_q_lip

    def update_target_network(self, soft_target_update_rate):
        soft_target_update(self.qf1, self.target_qf1, soft_target_update_rate)
        soft_target_update(self.qf2, self.target_qf2, soft_target_update_rate)

    def train(self, real_batch_size, sim_batch_size, bc=False):
        self._total_steps += 1

        real_batch = self.replay_buffer.sample(real_batch_size, scope="real")
        sim_batch = self.replay_buffer.sample(sim_batch_size, scope="sim")

        # real transitions from d^{\pi_\beta}_\mathcal{M}
        real_observations = real_batch['observations']
        real_actions = real_batch['actions']
        real_rewards = real_batch['rewards']
        real_next_observations = real_batch['next_observations']
        real_dones = real_batch['dones']

        # sim transitions from d^\pi_\mathcal{\widehat{M}}
        sim_observations = sim_batch['observations']
        sim_actions = sim_batch['actions']
        sim_rewards = sim_batch['rewards']
        sim_next_observations = sim_batch['next_observations']
        sim_dones = sim_batch['dones']

        # mixed transitions
        df_observations = torch.cat([real_observations, sim_observations], dim=0)
        df_actions = torch.cat([real_actions, sim_actions], dim=0)
        df_rewards = torch.cat([real_rewards, sim_rewards], dim=0)


        df_new_actions, df_log_pi = self.policy(df_observations)

        # True by default
        if self.config.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (df_log_pi + self.config.target_entropy).detach()).mean()
            alpha = self.log_alpha().exp() * self.config.alpha_multiplier
        else:
            alpha_loss = df_observations.new_tensor(0.0)
            alpha = df_observations.new_tensor(self.config.alpha_multiplier)

        """ Policy loss """
        # Improve policy under state marginal distribution d_f
        if bc:
            log_probs = self.policy.log_prob(df_observations, df_actions)
            policy_loss = (alpha * df_log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.qf1(df_observations, df_new_actions),
                self.qf2(df_observations, df_new_actions),
            )
            policy_loss = (alpha * df_log_pi - q_new_actions).mean()

        """ Q function loss """
        # Q function in real data and sim data
        if self.is_real_q_lip:
            real_q1_pred, real_q1_grad = self.qf1(real_observations, real_actions, True)
            real_q2_pred, real_q2_grad = self.qf2(real_observations, real_actions, True)
            real_q1_grad = torch.mean(real_q1_grad)
            real_q2_grad = torch.mean(real_q2_grad)
            real_q1_lip = self.tau_lip * ((max(0, real_q1_grad - self.threshold_lip)) ** 2)
            real_q2_lip = self.tau_lip * ((max(0, real_q2_grad - self.threshold_lip)) ** 2)
            real_q_lip = real_q1_lip + real_q2_lip
        else:
            real_q1_pred = self.qf1(real_observations, real_actions)
            real_q2_pred = self.qf2(real_observations, real_actions)
            real_q_lip = 0

        sim_q1_pred = self.qf1(sim_observations, sim_actions)
        sim_q2_pred = self.qf2(sim_observations, sim_actions)

        real_new_next_actions, real_next_log_pi = self.policy(real_next_observations)
        real_target_q_values = torch.min(
            self.target_qf1(real_next_observations, real_new_next_actions),
            self.target_qf2(real_next_observations, real_new_next_actions),
        )
        sim_new_next_actions, sim_next_log_pi = self.policy(sim_next_observations)
        sim_target_q_values = torch.min(
            self.target_qf1(sim_next_observations, sim_new_next_actions),
            self.target_qf2(sim_next_observations, sim_new_next_actions),
        )

        if self.config.backup_entropy:
            real_target_q_values = real_target_q_values - alpha * real_next_log_pi
            sim_target_q_values = sim_target_q_values - alpha * sim_next_log_pi
        real_td_target = torch.squeeze(real_rewards, -1) + (1. - torch.squeeze(real_dones, -1)) * self.config.discount * real_target_q_values
        sim_td_target = torch.squeeze(sim_rewards, -1) + (1. - torch.squeeze(sim_dones, -1)) * self.config.discount * sim_target_q_values

        real_qf1_loss = F.mse_loss(real_q1_pred, real_td_target.detach())
        real_qf2_loss = F.mse_loss(real_q2_pred, real_td_target.detach())

        sim_qf1_loss = F.mse_loss(sim_q1_pred, sim_td_target.detach())
        sim_qf2_loss = F.mse_loss(sim_q2_pred, sim_td_target.detach())
        qf1_loss = real_qf1_loss + sim_qf1_loss
        qf2_loss = real_qf2_loss + sim_qf2_loss

        #
        ### Conservative Penalty loss: sim data
        if not self.config.use_cql:
            qf_loss = qf1_loss + qf2_loss

            if self.config.use_automatic_entropy_tuning:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.qf_optimizer.zero_grad()
            qf_loss.backward()
            self.qf_optimizer.step()

            if self.total_steps % self.config.target_update_period == 0:
                self.update_target_network(
                    self.config.soft_target_update_rate
                )

            metrics = dict(
                log_pi=df_log_pi.mean().item(),
                policy_loss=policy_loss.item(),
                real_qf1_loss=real_qf1_loss.item(),
                real_qf2_loss=real_qf2_loss.item(),
                sim_qf1_loss=sim_qf1_loss.item(),
                sim_qf2_loss=sim_qf2_loss.item(),
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha_loss=alpha_loss.item(),
                alpha=alpha.item(),
                average_real_qf1=real_q1_pred.mean().item(),
                average_real_qf2=real_q2_pred.mean().item(),
                average_sim_qf1=sim_q1_pred.mean().item(),
                average_sim_qf2=sim_q2_pred.mean().item(),
                average_real_target_q=real_target_q_values.mean().item(),
                average_sim_target_q=sim_target_q_values.mean().item(),
                total_steps=self.total_steps,
            )
            return metrics
        else:
            # shape [128]
            cql_q1 = self.qf1(sim_observations, sim_actions)
            cql_q2 = self.qf2(sim_observations, sim_actions)

            omega = torch.ones(sim_rewards.shape[0], device=self.config.device)
            std_omega = omega.std()

            # Q values on the actions sampled from the policy
            if self.config.use_variant:
                cql_qf1_gap = (omega * cql_q1).sum()
                cql_qf2_gap = (omega * cql_q2).sum()
            else:
                # cql_q1 += torch.log(omega)
                # cql_q2 += torch.log(omega)
                cql_qf1_gap = torch.logsumexp(cql_q1 / self.config.cql_temp, dim=0) * self.config.cql_temp
                cql_qf2_gap = torch.logsumexp(cql_q2 / self.config.cql_temp, dim=0) * self.config.cql_temp

            """Q values on the stat-actions with larger dynamics gap - Q values on data"""
            cql_qf1_diff = torch.clamp(
                - cql_qf1_gap + real_q1_pred.mean(),
                self.config.cql_clip_diff_min,
                self.config.cql_clip_diff_max,
            )
            cql_qf2_diff = torch.clamp(
                - cql_qf2_gap + real_q2_pred.mean(),
                self.config.cql_clip_diff_min,
                self.config.cql_clip_diff_max,
            )
            # False by default
            if self.config.cql_lagrange:
                alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0)
                cql_min_qf1_loss = alpha_prime * self.config.cql_min_q_weight * (cql_qf1_diff - self.config.cql_target_action_gap)
                cql_min_qf2_loss = alpha_prime * self.config.cql_min_q_weight * (cql_qf2_diff - self.config.cql_target_action_gap)

                self.alpha_prime_optimizer.zero_grad()
                alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss)*0.5
                alpha_prime_loss.backward(retain_graph=True)
                self.alpha_prime_optimizer.step()
            else:
                cql_min_qf1_loss = cql_qf1_diff * self.config.cql_min_q_weight
                cql_min_qf2_loss = cql_qf2_diff * self.config.cql_min_q_weight
                alpha_prime_loss = df_observations.new_tensor(0.0)
                alpha_prime = df_observations.new_tensor(0.0)

            qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss + real_q_lip

            if self.config.use_automatic_entropy_tuning:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.qf_optimizer.zero_grad()
            qf_loss.backward()
            self.qf_optimizer.step()

            if self.total_steps % self.config.target_update_period == 0:
                self.update_target_network(
                    self.config.soft_target_update_rate
                )

            metrics = dict(
                log_pi=df_log_pi.mean().item(),
                policy_loss=policy_loss.item(),
                real_qf1_loss=real_qf1_loss.item(),
                real_qf2_loss=real_qf2_loss.item(),
                sim_qf1_loss=sim_qf1_loss.item(),
                sim_qf2_loss=sim_qf2_loss.item(),
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha_loss=alpha_loss.item(),
                alpha=alpha.item(),
                average_real_qf1=real_q1_pred.mean().item(),
                average_real_qf2=real_q2_pred.mean().item(),
                average_sim_qf1=sim_q1_pred.mean().item(),
                average_sim_qf2=sim_q2_pred.mean().item(),
                average_real_target_q=real_target_q_values.mean().item(),
                average_sim_target_q=sim_target_q_values.mean().item(),
                total_steps=self.total_steps,
            )

            if self.config.use_cql:
                metrics.update(prefix_metrics(dict(
                    std_omega=std_omega.mean().item(),
                    cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                    cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                    cql_qf1_diff=cql_qf1_diff.mean().item(),
                    cql_qf2_diff=cql_qf2_diff.mean().item(),
                    cql_qf1_gap=cql_qf1_gap.mean().item(),
                    cql_qf2_gap=cql_qf2_gap.mean().item(),
                ), 'cql'))

            return metrics

    def torch_to_device(self, device):
        for module in self.modules:
            module.to(device)

    @property
    def modules(self):
        modules = [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2]
        if self.config.use_automatic_entropy_tuning:
            modules.append(self.log_alpha)
        if self.config.cql_lagrange:
            modules.append(self.log_alpha_prime)
        return modules

    @property
    def total_steps(self):
        return self._total_steps
