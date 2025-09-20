import math
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
import gym
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from baselines.common import set_global_seeds
from train_procgen.utils import *
from copy import deepcopy
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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
parser.add_argument('--total_states', type=int, default=5000, help='choose how many states you want to have')

args = parser.parse_args()


def isEmpty(path):
    if os.path.exists(path) and not os.path.isfile(path):

        # Checking if the directory is empty or not
        if not os.listdir(path):
            return True
        else:
            return False
    else:
        return True


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
    cluster_path = './train_procgen/figures/' + args.env_name + '_cluster_images'
    # if empty, we need to collect states first
    if isEmpty(cluster_path):
        plt.rcParams.update({'font.size': 15})
        comm = MPI.COMM_WORLD

        saved_dir = './train_procgen/checkpoints/sppo-' + args.env_name + '_' + args.distribution_mode + '_' + str(
            args.num_levels) + '_' + str(args.start_level) + '_' + str(
            args.rand_seed)

        saved_dir += '/checkpoints/' + str(args.timesteps_per_proc)

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

        if not os.path.isdir(cluster_path):
            if not os.path.isdir('./train_procgen/figures'):
                os.mkdir('./train_procgen/figures')
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
        total_states = args.total_states
        upper_frames = args.total_states * 0.2
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
            if episode_counters[encoding_indices[0][0]] < upper_frames:
                saved_dir = cluster_path + '/cluster_' + str(encoding_indices[0][0])
                file_name = str(episode_i + 1) + '_' + str(sl[0][0]) + '_' + str(sl[0][1]) + '.png'
                im.save(saved_dir + '/' + file_name, dpi=(5, 5))
                episode_files.append(saved_dir + '/' + file_name)
                if len(sequence_files[encoding_indices[0][0]]) == 0:
                    sequence_files[encoding_indices[0][0]].append(saved_dir + '/' + file_name)
                    sequence_indices[encoding_indices[0][0]].append(episode_i + 1)
                else:
                    if (episode_i + 1 - int(sequence_indices[encoding_indices[0][0]][-1])) == 1:
                        sequence_files[encoding_indices[0][0]].append(saved_dir + '/' + file_name)
                        sequence_indices[encoding_indices[0][0]].append(episode_i + 1)
                    else:
                        if len(sequence_files[encoding_indices[0][0]]) > 20:
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
                if len(episode_files) > episode_max_steps:
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

            if np.sum(counters) >= total_states:
                env.close()
                break

    skill_latents = []
    cluster_indices = []
    images = []

    for i in range(args.num_embeddings):
        files = os.listdir(cluster_path + '/cluster_' + str(i))
        for file in files:
            if file.endswith('png'):
                _, sl_x, sl_y = file[:-4].split('_')
                arr_img = plt.imread(cluster_path + '/cluster_' + str(i) + '/' + file)
                images.append(arr_img)
                skill_latents.append([float(sl_x), float(sl_y)])
                cluster_indices.append(i)

    images = np.array(images)
    skill_latents = np.array(skill_latents)
    cluster_indices = np.array(cluster_indices)

    fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(8)
    np.random.seed(0)
    color_set = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(args.num_embeddings)]

    c_cet = []
    for cs in color_set:
        c_cet.append(c.to_hex(cs))
    c_cet = np.array(c_cet)

    figure = plt.scatter(skill_latents[:, 0], skill_latents[:, 1], s=10, c=c_cet[cluster_indices], alpha=0.5)
    line, = plt.plot(skill_latents[:, 0], skill_latents[:, 1], ls="")

    plt.title('The skill embedding space')
    plt.xlabel('skill x')
    plt.ylabel('skill y')

    # create the annotations box
    im = OffsetImage(images[0, :, :, :], zoom=0.2)
    xybox = (100., 100.)
    ab = AnnotationBbox(im, (0, 0), xybox=xybox, xycoords='data',
                        boxcoords="offset points", pad=0.3, arrowprops=dict(arrowstyle="->"))
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)

    def hover(event):
        # if the mouse is over the scatter points
        if line.contains(event)[0]:
            # find out the index within the array from the event
            try:
                ind, = line.contains(event)[1]["ind"]
            except:
                indices = line.contains(event)[1]["ind"]
                ind = indices[0]
            # get the figure size
            w, h = fig.get_size_inches() * fig.dpi
            ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
            hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
            # if event occurs in the top or right quadrant of the figure,
            # change the annotation box position relative to mouse.
            ab.xybox = (xybox[0] * ws, xybox[1] * hs)
            # make annotation box visible
            ab.set_visible(True)
            # place it at the position of the hovered scatter point
            ab.xy = (skill_latents[:, 0][ind], skill_latents[:, 1][ind])
            # set the image corresponding to that point
            im.set_data(images[ind, :, :, :])
        else:
            # if the mouse is not over a scatter point
            ab.set_visible(False)
        fig.canvas.draw_idle()

    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)
    plt.show()


if __name__ == '__main__':
    main()
