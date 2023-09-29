import datetime
import os
import sys
from copy import deepcopy
import absl.app
import absl.flags
import numpy as np
import wandb
from tqdm import trange

from H2O.SimpleSAC.envs import Env
from H2O.SimpleSAC.mixed_replay_buffer import MixedReplayBuffer
from H2O.SimpleSAC.replay_buffer import ReplayBuffer
from H2O.SimpleSAC.model import FullyConnectedQFunction, SamplerPolicy, TanhGaussianPolicy
from H2O.SimpleSAC.sampler import StepSampler, TrajSampler
from H2O.SimpleSAC.sim2real_sac import Sim2realSAC
from H2O.SimpleSAC.sac import SAC
from H2O.SimpleSAC.utils_h2o import (Timer, WandBLogger, define_flags_with_default,
                       get_user_flags, prefix_metrics,
                       set_random_seed)

from H2O.Network.Weight_net import ConcatDiscriminator
from H2O.viskit.logging import logger, setup_logger
import time
import argparse

import torch


waiting_time = [0, 0, 0]
while waiting_time[0] >= 0:
    while waiting_time[1] >= 0:
        while waiting_time[2] >= 0:
            time.sleep(1)
            print("Run after ",
                  waiting_time[0], "hours",
                  waiting_time[1], "minutes",
                  waiting_time[2], "seconds",
                  end='\r')
            waiting_time[2] -= 1
        waiting_time[1] -= 1
        waiting_time[2] = 60
    waiting_time[0] -= 1
    waiting_time[1] = 60



parser = argparse.ArgumentParser()
parser.add_argument('--USED_wandb', type=str, default="True")
parser.add_argument('--ego_policy', type=str, default="-sumo")  # "uniform", "sumo", "fvdm", "realdata", "genedata", "RL"
parser.add_argument('--adv_policy', type=str, default="-RL")  # "uniform", "sumo", "fvdm", "realdata", "genedata", "RL"
parser.add_argument('--n_epochs_ego', type=str, default="0")
parser.add_argument('--n_epochs_adv', type=str, default="1000")
parser.add_argument('--num_agents', type=int, default=4)
parser.add_argument('--r_ego', type=str, default="r1")
parser.add_argument('--r_adv', type=str, default="r3")
parser.add_argument('--realdata_path', type=str, default="dataset/Re_2_H2O/r3_dis_25_car_5/")  #
# parser.add_argument('--realdata_path', type=str, default="../byH2O/dataset/r1_dis_10_car_2/")  #
parser.add_argument('--batch_ratio', type=float, default=0.5)
parser.add_argument('--is_save', type=str, default="False")
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--save_model', type=str, default="False")
parser.add_argument('--cql_min_q_weight', type=float, default=0.1)

args = parser.parse_args()
args.is_save = True if args.is_save == "True" else False
args.USED_wandb = True if args.USED_wandb == "True" else False
args.save_model = True if args.save_model == "True" else False
print(args)
while len(sys.argv) > 1:
    sys.argv.pop()

FLAGS_DEF = define_flags_with_default(
    USED_wandb = args.USED_wandb,
    num_agents=args.num_agents,
    r_ego=args.r_ego,
    r_adv=args.r_adv,
    r_adv_replaybuffer=args.r_adv,
    realdata_path=args.realdata_path,
    batch_ratio=args.batch_ratio,
    is_save=args.is_save,
    device=args.device,
    cql_min_q_weight=args.cql_min_q_weight,
    seed=args.seed,
    save_model=args.save_model,

    ego_policy=args.ego_policy,
    adv_policy=args.adv_policy,
    n_epochs_ego=args.n_epochs_ego,
    n_epochs_adv=args.n_epochs_adv,

    replay_buffer_size=1000000,
    current_time=datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S'),
    replaybuffer_ratio=10,
    real_residual_ratio=1.0,
    dis_dropout=False,
    max_traj_length=100,
    batch_size=256,
    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=1.0,
    joint_noise_std=0.0,
    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    # train and evaluate policy
    bc_epochs=0,
    n_rollout_steps_per_epoch=1000,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=20,
    cql_adv=Sim2realSAC.get_default_config(device=args.device),
    cql_ego=SAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
)

def argparse():
    ...


def main(argv):
    FLAGS = absl.flags.FLAGS
    # tag = "reg_"
    tag = ""
    eval_savepath = "output/" + tag + \
                    f"simdata-ratio={FLAGS.batch_ratio}_av={FLAGS.ego_policy}_" \
                    f"bv={FLAGS.num_agents}-{FLAGS.adv_policy}_" \
                    f"r-ego={FLAGS.r_ego}_r-adv={FLAGS.r_adv}_" \
                    f"seed={FLAGS.seed}_time={FLAGS.current_time}" + "/"
    if FLAGS.is_save:
        if os.path.exists("output") is False:
            os.mkdir("output")
        if os.path.exists(eval_savepath) is False:
            os.mkdir(eval_savepath)
            os.mkdir(eval_savepath + "avcrash")
            os.mkdir(eval_savepath + "bvcrash")
            os.mkdir(eval_savepath + "avarrive")
        eval_savepath_train_ego = eval_savepath + "train_ego" + "/"
        if os.path.exists(eval_savepath_train_ego) is False:
            os.mkdir(eval_savepath_train_ego)
            os.mkdir(eval_savepath_train_ego + "avcrash")
            os.mkdir(eval_savepath_train_ego + "bvcrash")
            os.mkdir(eval_savepath_train_ego + "avarrive")
    else:
        if "genedata" in FLAGS.adv_policy or "genedata" in FLAGS.ego_policy:
            return
        eval_savepath = None
        eval_savepath_train_ego = None
    if FLAGS.save_model:
        path_save_model = eval_savepath + "models/"
        if os.path.exists("output") is False:
            os.mkdir("output")
        if os.path.exists(eval_savepath) is False:
            os.mkdir(eval_savepath)
        if os.path.exists(path_save_model) is False:
            os.mkdir(path_save_model)
    if FLAGS.USED_wandb:
        variant = get_user_flags(FLAGS, FLAGS_DEF)
        wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
        wandb.run.name = tag + \
                         f"simdata-ratio={FLAGS.batch_ratio}_av={FLAGS.ego_policy}_" \
                         f"bv={FLAGS.num_agents}-{FLAGS.adv_policy}_" \
                         f"r-ego={FLAGS.r_ego}_r-adv={FLAGS.r_adv}_" \
                         f"seed={FLAGS.seed}_time={FLAGS.current_time}"

        setup_logger(
            variant=variant,
            exp_id=wandb_logger.experiment_id,
            seed=FLAGS.seed,
            base_log_dir=FLAGS.logging.output_dir,
            include_exp_prefix_sub_dir=False
        )

    set_random_seed(FLAGS.seed)

    real_env = Env(realdata_path=FLAGS.realdata_path,
                   genedata_path=None if eval_savepath is None else eval_savepath + "avcrash/",
                   num_agents=FLAGS.num_agents, sim_horizon=FLAGS.max_traj_length,
                   ego_policy=FLAGS.ego_policy, adv_policy=FLAGS.adv_policy,
                   r_ego=FLAGS.r_ego, r_adv=FLAGS.r_adv, sim_seed=FLAGS.seed)
    sim_env = Env(realdata_path=FLAGS.realdata_path,
                  genedata_path=None if eval_savepath is None else eval_savepath + "avcrash/",
                  num_agents=FLAGS.num_agents, sim_horizon=FLAGS.max_traj_length,
                   ego_policy=FLAGS.ego_policy, adv_policy=FLAGS.adv_policy,
                   r_ego=FLAGS.r_ego, r_adv=FLAGS.r_adv, sim_seed=FLAGS.seed)

    train_sampler = StepSampler(sim_env, max_traj_length=FLAGS.max_traj_length)
    eval_sampler = TrajSampler(real_env, rootsavepath=eval_savepath, max_traj_length=FLAGS.max_traj_length)

    # replay buffer
    num_state = real_env.state_space[0]
    num_action_adv = real_env.action_space_adv[0]
    num_action_ego = real_env.action_space_ego[0]
    replay_buffer_ego = ReplayBuffer(num_state, num_action_ego, FLAGS.replay_buffer_size, device=FLAGS.device) \
        if "RL" in FLAGS.ego_policy else None
        # if FLAGS.ego_policy == "RL" else None
    replay_buffer_adv = MixedReplayBuffer(FLAGS.reward_scale, FLAGS.reward_bias, FLAGS.clip_action,
                                          num_state, num_action_adv,
                                          realdata_path=FLAGS.realdata_path,
                                          device=FLAGS.device, buffer_ratio=FLAGS.replaybuffer_ratio,
                                          residual_ratio=FLAGS.real_residual_ratio, r_adv=FLAGS.r_adv_replaybuffer) \
        if "RL" in FLAGS.adv_policy else None
        # if FLAGS.adv_policy == "RL" else None

    # if FLAGS.ego_policy == "RL":
    if "RL" in FLAGS.ego_policy:
        ego_policy = TanhGaussianPolicy(
            num_state,
            num_action_ego,
            arch=FLAGS.policy_arch,
            log_std_multiplier=FLAGS.policy_log_std_multiplier,
            log_std_offset=FLAGS.policy_log_std_offset,
            orthogonal_init=FLAGS.orthogonal_init,
        )

        qf1_ego = FullyConnectedQFunction(
            num_state,
            num_action_ego,
            arch=FLAGS.qf_arch,
            orthogonal_init=FLAGS.orthogonal_init,
        )
        target_qf1_ego = deepcopy(qf1_ego)

        qf2_ego = FullyConnectedQFunction(
            num_state,
            num_action_ego,
            arch=FLAGS.qf_arch,
            orthogonal_init=FLAGS.orthogonal_init,
        )
        target_qf2_ego = deepcopy(qf2_ego)

        if FLAGS.cql_ego.target_entropy >= 0.0:
            FLAGS.cql_ego.target_entropy = -np.prod(eval_sampler.env.action_space_ego).item()

        sac_ego = SAC(FLAGS.cql_ego, ego_policy, qf1_ego, qf2_ego, target_qf1_ego, target_qf2_ego)
        sac_ego.torch_to_device(FLAGS.device)

        sampler_ego_policy = SamplerPolicy(ego_policy, FLAGS.device)
    else:
        sac_ego = None
        sampler_ego_policy = None

    if "RL" in FLAGS.adv_policy:
        # discirminators
        d_sa = ConcatDiscriminator(num_state + num_action_adv, 256, 2,
                                   FLAGS.device, dropout=FLAGS.dis_dropout).float().to(FLAGS.device)
        d_sas = ConcatDiscriminator(2 * num_state + num_action_adv, 256, 2,
                                    FLAGS.device, dropout=FLAGS.dis_dropout).float().to(FLAGS.device)

        adv_policy = TanhGaussianPolicy(
            num_state,
            num_action_adv,
            arch=FLAGS.policy_arch,
            log_std_multiplier=FLAGS.policy_log_std_multiplier,
            log_std_offset=FLAGS.policy_log_std_offset,
            orthogonal_init=FLAGS.orthogonal_init,
        )

        qf1_adv = FullyConnectedQFunction(
            num_state,
            num_action_adv,
            arch=FLAGS.qf_arch,
            orthogonal_init=FLAGS.orthogonal_init,
            is_LN=True
        )
        target_qf1_adv = deepcopy(qf1_adv)

        qf2_adv = FullyConnectedQFunction(
            num_state,
            num_action_adv,
            arch=FLAGS.qf_arch,
            orthogonal_init=FLAGS.orthogonal_init,
            is_LN=True
        )
        target_qf2_adv = deepcopy(qf2_adv)

        if FLAGS.cql_adv.target_entropy >= 0.0:
            FLAGS.cql_adv.target_entropy = -np.prod(eval_sampler.env.action_space_adv).item()

        sac_adv = Sim2realSAC(FLAGS.cql_adv, adv_policy, qf1_adv, qf2_adv, target_qf1_adv, target_qf2_adv, d_sa, d_sas,
                              replay_buffer_adv, dynamics_model=None, device=FLAGS.device,
                              is_real_q_lip=True
                              )
        sac_adv.torch_to_device(FLAGS.device)

        # sampling policy is always the current policy: \pi
        sampler_adv_policy = SamplerPolicy(adv_policy, FLAGS.device)
    else:
        sac_adv = None
        sampler_adv_policy = None

    viskit_metrics = {}

    n_epochs_ego = [int(h) for h in FLAGS.n_epochs_ego.split('-')]
    n_epochs_adv = [int(h) for h in FLAGS.n_epochs_adv.split('-')]
    ego_policy_list = FLAGS.ego_policy.split('-')
    ego_policy_list.reverse()
    adv_policy_list = FLAGS.adv_policy.split('-')
    adv_policy_list.reverse()
    n_loops = max(len(n_epochs_adv), len(n_epochs_ego))
    for l in range(n_loops):
        if l < len(n_epochs_ego):
            eval_sampler.rootsavepath = eval_savepath_train_ego
            ego_policy_curr = ego_policy_list.pop()
            adv_policy_curr = adv_policy_list.pop()
            train_sampler.env.ego_policy = ego_policy_curr
            train_sampler.env.adv_policy = adv_policy_curr
            eval_sampler.env.ego_policy = ego_policy_curr
            eval_sampler.env.adv_policy = adv_policy_curr
            for epoch in trange(n_epochs_ego[l]):
                metrics = {}

                # TODO rollout from the simulator
                with Timer() as rollout_timer:
                    train_sampler.sample(
                        ego_policy=sampler_ego_policy, adv_policy=sampler_adv_policy, n_steps=FLAGS.n_rollout_steps_per_epoch,
                        deterministic=False, replay_buffer_ego=replay_buffer_ego, replay_buffer_adv=None,
                        joint_noise_std=FLAGS.joint_noise_std
                    )
                    metrics['epoch'] = epoch

                # TODO Train from the mixed data
                with Timer() as train_timer:
                    if ego_policy_curr == "RL":
                        for batch_idx in trange(FLAGS.n_train_step_per_epoch):
                            batch_ego = replay_buffer_ego.sample(FLAGS.batch_size)
                            if batch_idx + 1 == FLAGS.n_train_step_per_epoch:
                                metrics.update(
                                    prefix_metrics(sac_ego.train(batch_ego), 'sac_ego')
                                )
                            else:
                                sac_ego.train(batch_ego)

                with Timer() as eval_timer:
                    # eval_sampler.env.adv_policy = "fvdm"
                    if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                        trajs = eval_sampler.sample(
                            ego_policy=sampler_ego_policy, adv_policy=sampler_adv_policy,
                            n_trajs=FLAGS.eval_n_trajs, deterministic=True
                        )

                        metrics['average_return_adv'] = np.mean([np.mean(t['rewards_adv']) for t in trajs])
                        metrics['average_return_ego'] = np.mean([np.mean(t['rewards_ego']) for t in trajs])
                        metrics['average_traj_length'] = np.mean([len(t['rewards_adv']) for t in trajs])
                        metrics['metrics_av_crash'] = np.mean([t["metrics_av_crash"] for t in trajs])
                        metrics['metrics_bv_crash'] = np.mean([t["metrics_bv_crash"] for t in trajs])
                        metrics['ACT'] = 0 if metrics['metrics_av_crash'] == 0 else \
                            np.sum([t["collision_time"] for t in trajs]) / (metrics['metrics_av_crash'] * len(trajs))
                        metrics['ACD'] = 0 if metrics['metrics_av_crash'] == 0 else \
                            np.sum([t["collision_dis"] for t in trajs]) / (metrics['metrics_av_crash'] * len(trajs))
                        metrics['CPS'] = (metrics['metrics_av_crash'] * len(trajs)) / np.sum([t["traj_time"] for t in trajs])
                        metrics['CPM'] = (metrics['metrics_av_crash'] * len(trajs)) / np.sum([t["traj_dis"] for t in trajs])

                metrics['rollout_time'] = rollout_timer()
                metrics['train_time'] = train_timer()
                metrics['eval_time'] = eval_timer()
                metrics['epoch_time'] = rollout_timer() + train_timer() + eval_timer()
                if FLAGS.USED_wandb:
                    wandb_logger.log(metrics)
                viskit_metrics.update(metrics)
                logger.record_dict(viskit_metrics)
                logger.dump_tabular(with_prefix=False, with_timestamp=False)
            if n_epochs_ego[l] != 0 and FLAGS.save_model:
                torch.save(ego_policy, path_save_model + "model_av_loop" + str(l) + ".pkl")
                # torch.save(sac_ego, path_save_model + "model_av_loop" + str(l) + ".pkl")

        if l < len(n_epochs_adv):
            # 设置储存路径
            eval_sampler.rootsavepath = eval_savepath
            # 设置AV和BV策略
            ego_policy_curr = ego_policy_list.pop()
            adv_policy_curr = adv_policy_list.pop()
            train_sampler.env.ego_policy = ego_policy_curr
            train_sampler.env.adv_policy = adv_policy_curr
            eval_sampler.env.ego_policy = ego_policy_curr
            eval_sampler.env.adv_policy = adv_policy_curr
            # 开始训练
            for epoch in trange(n_epochs_adv[l]):
                metrics = {}

                with Timer() as rollout_timer:
                    train_sampler.sample(
                        ego_policy=sampler_ego_policy, adv_policy=sampler_adv_policy, n_steps=FLAGS.n_rollout_steps_per_epoch,
                        deterministic=False, replay_buffer_ego=None, replay_buffer_adv=replay_buffer_adv,
                        joint_noise_std=FLAGS.joint_noise_std
                    )
                    metrics['epoch'] = epoch

                with Timer() as train_timer:
                    if adv_policy_curr == "RL":
                        for batch_idx in trange(FLAGS.n_train_step_per_epoch):
                            real_batch_size = int(FLAGS.batch_size * (1 - FLAGS.batch_ratio))
                            sim_batch_size = int(FLAGS.batch_size * FLAGS.batch_ratio)
                            if batch_idx + 1 == FLAGS.n_train_step_per_epoch:
                                metrics.update(
                                    prefix_metrics(sac_adv.train(real_batch_size, sim_batch_size), 'sac_adv')
                                )
                            else:
                                sac_adv.train(real_batch_size, sim_batch_size)

                with Timer() as eval_timer:
                    if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                        trajs = eval_sampler.sample(
                            ego_policy=sampler_ego_policy, adv_policy=sampler_adv_policy,
                            n_trajs=FLAGS.eval_n_trajs, deterministic=True
                        )
                        eval_dsa_loss, eval_dsas_loss = sac_adv.discriminator_evaluate()
                        metrics['eval_dsa_loss'] = eval_dsa_loss
                        metrics['eval_dsas_loss'] = eval_dsas_loss
                        metrics['average_return_adv'] = np.mean([np.mean(t['rewards_adv']) for t in trajs])
                        metrics['average_return_ego'] = np.mean([np.mean(t['rewards_ego']) for t in trajs])
                        metrics['average_traj_length'] = np.mean([len(t['rewards_adv']) for t in trajs])
                        metrics['metrics_av_crash'] = np.mean([t["metrics_av_crash"] for t in trajs])
                        metrics['metrics_bv_crash'] = np.mean([t["metrics_bv_crash"] for t in trajs])
                        metrics['ACT'] = 0 if metrics['metrics_av_crash'] == 0 else \
                            np.sum([t["collision_time"] for t in trajs]) / (metrics['metrics_av_crash'] * len(trajs))
                        metrics['ACD'] = 0 if metrics['metrics_av_crash'] == 0 else \
                            np.sum([t["collision_dis"] for t in trajs]) / (metrics['metrics_av_crash'] * len(trajs))
                        metrics['CPS'] = (metrics['metrics_av_crash'] * len(trajs)) / np.sum([t["traj_time"] for t in trajs])
                        metrics['CPM'] = (metrics['metrics_av_crash'] * len(trajs)) / np.sum([t["traj_dis"] for t in trajs])

                metrics['rollout_time'] = rollout_timer()
                metrics['train_time'] = train_timer()
                metrics['eval_time'] = eval_timer()
                metrics['epoch_time'] = rollout_timer() + train_timer() + eval_timer()
                if FLAGS.USED_wandb:
                    wandb_logger.log(metrics)
                viskit_metrics.update(metrics)
                logger.record_dict(viskit_metrics)
                logger.dump_tabular(with_prefix=False, with_timestamp=False)
            if n_epochs_adv[l] != 0 and FLAGS.save_model:
                torch.save(adv_policy, path_save_model + "model_bv_loop" + str(l) + ".pkl")


if __name__ == '__main__':
    absl.app.run(main)
