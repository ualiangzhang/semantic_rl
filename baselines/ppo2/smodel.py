import tensorflow as tf
import functools

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize
import numpy as np
from itertools import combinations

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None


class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """

    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, skill_coef, vq_coef, max_grad_norm, alpha1=20, alpha2=1, beta=0.25,
                 Cd=5, mpi_rank_weight=1, comm=None,
                 microbatch_size=None):
        self.sess = sess = get_session()

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        with tf.compat.v1.variable_scope('sppo2_model', reuse=tf.compat.v1.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.compat.v1.placeholder(tf.float32, [None])
        self.R = R = tf.compat.v1.placeholder(tf.float32, [None])
        self.skill_weight = skill_weight = tf.compat.v1.placeholder(tf.float32, [])

        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.compat.v1.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.compat.v1.placeholder(tf.float32, [None])
        self.LR = LR = tf.compat.v1.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.compat.v1.placeholder(tf.float32, [])

        self.train_skill = train_skill = tf.compat.v1.placeholder(tf.bool, [])

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(input_tensor=train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(input_tensor=tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # calculating the skill loss
        # To avoid the GPU memory exploded, please choose the proper chunk size for your GPUs
        chunk_size = 64

        def pairwise_sqd_distance(x):
            if x.shape[-1] <= chunk_size:
                tiled = tf.tile(tf.expand_dims(x, axis=1), tf.stack([1, x.shape[0], 1]))
                tiled_trans = tf.transpose(tiled, perm=[1, 0, 2])
                diffs = tiled - tiled_trans
                sqd_dist_mat = tf.reduce_sum(tf.square(diffs), axis=2)
            else:
                sqd_dist_mat = []
                for i in range(x.shape[-1] // chunk_size):
                    tiled = tf.tile(tf.expand_dims(x[:, i * chunk_size:i * chunk_size + chunk_size], axis=1),
                                    tf.stack([1, x.shape[0], 1]))

                    tiled_trans = tf.transpose(tiled, perm=[1, 0, 2])

                    diffs = tiled - tiled_trans
                    if sqd_dist_mat != []:
                        sqd_dist_mat = tf.add(sqd_dist_mat, tf.reduce_sum(tf.square(diffs), axis=2))
                    else:
                        sqd_dist_mat = tf.reduce_sum(tf.square(diffs), axis=2)

            return sqd_dist_mat

        def make_q(z, alpha):
            sqd_dist_mat = pairwise_sqd_distance(z)
            q = tf.pow((1 + sqd_dist_mat / alpha), -(alpha + 1) / 2)
            q = tf.linalg.set_diag(q, np.zeros(shape=[z.shape[0]]))
            q = q / tf.reduce_sum(q, axis=0, keepdims=True)
            q = tf.clip_by_value(q, 1e-10, 1.0)

            return q

        p = make_q(train_model.pure_latent, alpha=alpha1)
        q = make_q(train_model.skill_latent, alpha=alpha2)
        skill_losses = tf.reduce_sum(-(tf.multiply(p, tf.math.log(q)))) / (p.shape[0] * p.shape[1])
        skill_loss_ = tf.reduce_mean(skill_losses)

        # calculating the vq loss
        if beta != 0:
            commitment_loss = beta * tf.reduce_mean(
                input_tensor=(tf.stop_gradient(train_model.pure_vq_latent) - train_model.skill_latent) ** 2, axis=1
            )
        else:
            commitment_loss = 0
        codebook_loss = tf.reduce_mean(
            input_tensor=(train_model.pure_vq_latent - tf.stop_gradient(train_model.skill_latent)) ** 2, axis=1)

        vq_losses = commitment_loss + codebook_loss
        vq_loss_ = tf.reduce_mean(vq_losses)

        # Final PG loss
        pg_loss = tf.reduce_mean(input_tensor=tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(input_tensor=tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(input_tensor=tf.cast(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE), dtype=tf.float32))

        # Total loss

        loss = tf.cond(train_skill, lambda: pg_loss - entropy * ent_coef + vf_loss * vf_coef + skill_weight * (
                skill_loss_ * skill_coef + vq_loss_ * vq_coef),
                       lambda: pg_loss - entropy * ent_coef + vf_loss * vf_coef)

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.compat.v1.trainable_variables('sppo2_model')
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = ['loss', 'policy_loss', 'value_loss', 'skill_loss', 'vq_loss',
                           'policy_entropy',
                           'approxkl',
                           'clipfrac']
        self.stats_list = [loss, pg_loss, vf_loss, skill_loss_, vq_loss_, entropy, approxkl,
                           clipfrac]

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.skill_step = act_model.skill_step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm)  # pylint: disable=E1101

    def train(self, lr, cliprange, train_skill, skill_weight, obs, returns,
              masks, actions,
              values,
              neglogpacs,
              states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X: obs,
            self.A: actions,
            self.ADV: advs,
            self.R: returns,
            self.LR: lr,
            self.train_skill: train_skill,
            self.skill_weight: skill_weight,
            self.CLIPRANGE: cliprange,
            self.OLDNEGLOGPAC: neglogpacs,
            self.OLDVPRED: values
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]
