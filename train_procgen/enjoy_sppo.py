import math
import os
import sys
import shutil

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

from baselines.ppo2 import sppo2
from baselines.common.models import build_skill_impala_cnn
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
import matplotlib.colors as c
import matplotlib.patches as mpatches
import gym
from sklearn.manifold import TSNE
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from baselines.common import set_global_seeds
from train_procgen.utils import *
from train_procgen.constants import EASY_GAME_RANGES
from copy import deepcopy

parser = argparse.ArgumentParser(description='Process procgen enjoying arguments.')
parser.add_argument('--env_name', type=str, default='coinrun')
parser.add_argument('--num_envs', type=int, default=64)
parser.add_argument('--distribution_mode', type=str, default='easy',
                    choices=["easy", "hard", "exploration", "memory", "extreme"])
parser.add_argument('--num_levels', type=int, default=0)
parser.add_argument('--start_level', type=int, default=0)
parser.add_argument('--test_worker_interval', type=int, default=0)
parser.add_argument('--timesteps_per_proc', type=int, default=25_000_000)
parser.add_argument('--rand_seed', type=int, default=2021)
parser.add_argument('--num_embeddings', type=int, default=8)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--alpha1', type=float, default=20)
parser.add_argument('--alpha2', type=float, default=20)
parser.add_argument('--mode', type=int, default=1,
                    help='0: generate clustering videos 1: generate embedding spaces')

args = parser.parse_args()


def load_model(env_name, num_envs, distribution_mode, num_levels, start_level,
               num_embeddings, beta, alpha1, alpha2,
               log_dir='./checkpoints', comm=None, rand_seed=None):
    ent_coef = .01
    nsteps = 256
    nminibatches = 8

    mpi_rank_weight = 1
    num_levels = num_levels

    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, start_level=start_level + 200000,
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

    conv_fn = lambda x: build_skill_impala_cnn(x, depths=[16, 32, 32], emb_dim=256, num_embeddings=num_embeddings,
                                               beta=beta, seed=rand_seed)

    logger.info("loading the model")

    model = sppo2.load(
        env=venv,
        network=conv_fn,
        nsteps=nsteps,
        nminibatches=nminibatches,
        ent_coef=ent_coef,
        mpi_rank_weight=mpi_rank_weight,
        comm=comm,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=rand_seed,
        load_path=log_dir,
        alpha1=alpha1,
        alpha2=alpha2,
    )

    return model


def main():
    plt.rcParams.update({'font.size': 15})
    comm = MPI.COMM_WORLD

    saved_dir = './train_procgen/checkpoints/sppo-' + args.env_name + '_' + args.distribution_mode + '_' + str(
        args.num_levels) + '_' + str(args.start_level) + '_' + str(
        args.rand_seed)

    saved_dir += '/checkpoints/' + str(args.timesteps_per_proc)

    if not os.path.isdir('./train_procgen/figures'):
        os.mkdir('./train_procgen/figures')

    if args.mode == 0:
        model = load_model(args.env_name,
                           1,
                           args.distribution_mode,
                           args.num_levels,
                           args.start_level,
                           args.num_embeddings,
                           comm=comm,
                           log_dir=saved_dir,
                           rand_seed=args.rand_seed,
                           beta=args.beta,
                           alpha1=args.alpha1,
                           alpha2=args.alpha2,
                           )

        cluster_path = './train_procgen/figures/' + args.env_name + '_skills'

        if not os.path.isdir(cluster_path):
            os.mkdir(cluster_path)
        for i in range(args.num_embeddings):
            if not os.path.isdir(cluster_path + '/cluster_' + str(i)):
                os.mkdir(cluster_path + '/cluster_' + str(i))

        # generate the skill clusters
        set_global_seeds(args.rand_seed)
        env_name = "procgen:procgen-" + args.env_name + "-v0"
        env = gym.make(env_name, num_levels=args.num_levels, start_level=args.start_level,
                       distribution_mode=args.distribution_mode, rand_seed=args.rand_seed, render_mode="human")
        obs = env.reset()
        counters = np.zeros(args.num_embeddings)

        episode_i = 0
        total_frames = 480
        episode_files = []
        sequence_files = [[] for _ in range(args.num_embeddings)]
        sequence_indices = [[] for _ in range(args.num_embeddings)]
        episode_rewards = 0
        episode_counters = np.zeros(args.num_embeddings)
        # avoid to collect the endless loop
        if args.env_name not in ['starpilot', 'fruitbot']:
            episode_max_steps = 200
        else:
            episode_max_steps = math.inf

        while True:
            rgb_img = env.render(mode="rgb_array")
            im = Image.fromarray(rgb_img)
            a, v, pure_latent, vq_latent, pure_vq_latent, vq_embeddings, encoding_indices, sl, lat = model.skill_step(
                obs)
            action = np.squeeze(a)
            if episode_counters[encoding_indices[0][0]] < total_frames:
                saved_dir = cluster_path + '/cluster_' + str(encoding_indices[0][0])
                file_name = str(episode_i + 1) + '_' + str(action) + '.png'
                im.save(saved_dir + '/' + file_name, dpi=(30, 30))
                episode_files.append(saved_dir + '/' + file_name)
                if len(sequence_files[encoding_indices[0][0]]) == 0:
                    sequence_files[encoding_indices[0][0]].append(saved_dir + '/' + file_name)
                    sequence_indices[encoding_indices[0][0]].append(episode_i + 1)
                else:
                    if (episode_i + 1 - int(sequence_indices[encoding_indices[0][0]][-1])) == 1:
                        sequence_files[encoding_indices[0][0]].append(saved_dir + '/' + file_name)
                        sequence_indices[encoding_indices[0][0]].append(episode_i + 1)
                    else:
                        if len(sequence_files[encoding_indices[0][0]]) <= 4 or len(
                                sequence_files[encoding_indices[0][0]]) > 20:
                            for f in sequence_files[encoding_indices[0][0]]:
                                if os.path.isfile(f):
                                    os.remove(f)
                                    episode_counters[encoding_indices[0][0]] -= 1
                                else:
                                    continue

                        sequence_files[encoding_indices[0][0]] = []
                        sequence_indices[encoding_indices[0][0]] = []
                        sequence_files[encoding_indices[0][0]].append(saved_dir + '/' + file_name)
                        sequence_indices[encoding_indices[0][0]].append(episode_i + 1)

            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            episode_counters[encoding_indices[0][0]] += 1
            episode_i += 1

            if done:
                if episode_rewards <= EASY_GAME_RANGES[args.env_name][-1] * 0.8 or len(
                        episode_files) > episode_max_steps:
                    for f in episode_files:
                        if os.path.isfile(f):
                            os.remove(f)
                            for i in range(args.num_embeddings):
                                if f in sequence_files[i]:
                                    sequence_indices[i].pop(sequence_files[i].index(f))
                                    sequence_files[i].remove(f)
                        else:
                            continue

                    episode_counters = deepcopy(counters)
                counters = deepcopy(episode_counters)
                episode_files = []
                episode_rewards = 0

            if np.min(counters) >= total_frames:
                break

        for i in range(args.num_embeddings):
            convert_frames_to_video(cluster_path + '/cluster_' + str(i) + '/',
                                    cluster_path + '/cluster_' + str(i) + '.avi', limits=total_frames)
            shutil.rmtree(cluster_path + '/cluster_' + str(i))

    if args.mode == 1:
        figure_name_prefix = args.env_name + '_' + str(args.num_levels) + '_'
        # visualize the embedding spaces of pure features, skill latents, and vq latents
        model = load_model(args.env_name,
                           args.num_envs,
                           args.distribution_mode,
                           args.num_levels,
                           args.start_level,
                           args.num_embeddings,
                           comm=comm,
                           log_dir=saved_dir,
                           rand_seed=args.rand_seed,
                           beta=args.beta,
                           alpha1=args.alpha1,
                           alpha2=args.alpha2,
                           )

        set_global_seeds(args.rand_seed)

        venv = ProcgenEnv(num_envs=args.num_envs, env_name=args.env_name, num_levels=args.num_levels,
                          start_level=args.start_level + 200000,
                          distribution_mode=args.distribution_mode, rand_seed=args.rand_seed)
        obs = venv.reset()

        pure_features = []
        skill_latents = []
        enc_indices = []
        values = []

        for i in range(500):
            a, v, pure_latent, vq_latent, pure_vq_latent, vq_embeddings, encoding_indices, sl, lat = model.skill_step(
                obs['rgb'])
            action = np.squeeze(a)
            # record 8/10 data to ensure explore more states
            if np.random.uniform(0, 1) > 0.8:
                pure_features.append(pure_latent)
                skill_latents.append(sl)
                enc_indices.append(encoding_indices)
                values.append(v)

            action = [np.random.choice([np.random.choice([i for i in range(15)]), action[i]], p=[0.2, 0.8]) for i in
                      range(args.num_envs)]
            obs, reward, done, info = venv.step(np.array(action, dtype=np.int32))

        pure_features = np.reshape(np.squeeze(np.array(pure_features)), (-1, 256))
        skill_latents = np.reshape(np.squeeze(np.array(skill_latents)), (-1, 2))
        enc_indices = np.reshape(np.squeeze(np.array(enc_indices)), (-1))
        values = np.reshape(np.squeeze(np.array(values)), (-1))

        n_components = 2
        tsne = TSNE(n_components)
        tsne_result = tsne.fit_transform(pure_features)

        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(8)
        np.random.seed(0)
        color_set = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(args.num_embeddings)]

        c_cet = []
        for cs in color_set:
            c_cet.append(c.to_hex(cs))
        c_cet = np.array(c_cet)
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=10, c=c_cet[enc_indices], alpha=0.5)

        pop_ = []
        for i in range(args.num_embeddings):
            if i in enc_indices:
                pop_.append(mpatches.Patch(color=color_set[i], label='Cluster ' + str(i)))

        plt.legend(handles=pop_, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)

        plt.title('The t-SNE space')
        plt.xlabel('t-SNE x')
        plt.ylabel('t-SNE y')

        fig.savefig('./train_procgen/figures/' + figure_name_prefix + 'feature_embeddings.png', dpi=600, bbox_inches='tight')
        # plt.show()

        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(8)
        np.random.seed(0)
        color_set = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(args.num_embeddings)]

        c_cet = []
        for cs in color_set:
            c_cet.append(c.to_hex(cs))
        c_cet = np.array(c_cet)
        plt.scatter(skill_latents[:, 0], skill_latents[:, 1], s=10, c=c_cet[enc_indices], alpha=0.5)

        pop_ = []
        for i in range(args.num_embeddings):
            if i in enc_indices:
                pop_.append(mpatches.Patch(color=color_set[i], label='Cluster ' + str(i)))

        plt.legend(handles=pop_, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)

        plt.title('The FDR space')
        plt.xlabel('FDR x')
        plt.ylabel('FDR y')

        fig.savefig('./train_procgen/figures/' + figure_name_prefix + 'skill_embeddings.png', dpi=600, bbox_inches='tight')
        # plt.show()

if __name__ == '__main__':
    main()
