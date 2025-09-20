import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.simplefilter("ignore")
    import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import warnings

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

from baselines.ppo2 import ppo2
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI
import argparse

parser = argparse.ArgumentParser(description='Process procgen training arguments.')
parser.add_argument('--env_name', type=str, default='coinrun')
parser.add_argument('--num_envs', type=int, default=64)
parser.add_argument('--distribution_mode', type=str, default='hard',
                    choices=["easy", "hard", "exploration", "memory", "extreme"])
parser.add_argument('--num_levels', type=int, default=0)
parser.add_argument('--start_level', type=int, default=0)
parser.add_argument('--test_worker_interval', type=int, default=0)
parser.add_argument('--timesteps_per_proc', type=int, default=25_000_000)
parser.add_argument('--rand_seed', type=int, default=2022)

args = parser.parse_args()


def train_fn(env_name, num_envs, distribution_mode, num_levels, start_level, timesteps_per_proc, is_test_worker=False,
             log_dir='./checkpoints', comm=None, rand_seed=None):
    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True

    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else num_levels

    if log_dir is not None:
        log_comm = comm.Split(1 if is_test_worker else 0, 0)
        format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
        logger.configure(comm=log_comm, dir=log_dir, format_strs=format_strs)

    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, start_level=start_level,
                      distribution_mode=distribution_mode, rand_seed=rand_seed)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=E1101
    sess = tf.compat.v1.Session(config=config)
    sess.__enter__()

    conv_fn = lambda x: build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)

    logger.info("training")
    ppo2.learn(
        env=venv,
        network=conv_fn,
        total_timesteps=timesteps_per_proc,
        save_interval=0,
        nsteps=nsteps,
        nminibatches=nminibatches,
        lam=lam,
        gamma=gamma,
        noptepochs=ppo_epochs,
        log_interval=1,
        ent_coef=ent_coef,
        mpi_rank_weight=mpi_rank_weight,
        clip_vf=use_vf_clipping,
        comm=comm,
        lr=learning_rate,
        cliprange=clip_range,
        update_fn=None,
        init_fn=None,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=rand_seed
    )


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False
    test_worker_interval = args.test_worker_interval

    if test_worker_interval > 0:
        is_test_worker = rank % test_worker_interval == (test_worker_interval - 1)

    saved_dir = './train_procgen/checkpoints/ppo-' + args.env_name + '_' + args.distribution_mode + '_' + str(
        args.num_levels) + '_' + str(args.start_level) + '_' + str(
        args.rand_seed)

    train_fn(args.env_name,
             args.num_envs,
             args.distribution_mode,
             args.num_levels,
             args.start_level,
             args.timesteps_per_proc,
             is_test_worker=is_test_worker,
             comm=comm,
             log_dir=saved_dir,
             rand_seed=args.rand_seed)


if __name__ == '__main__':
    main()
