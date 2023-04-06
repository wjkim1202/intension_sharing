import numpy as np
import tensorflow as tf
from itertools import chain
import tensorflow.contrib.layers as layers
import math
EPS = 1e-8

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))
def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None, bn=False, is_training=np.bool(0)):
    for h in hidden_sizes[:-1]:
        if bn:
            x = tf.layers.dense(x, units=h, activation=None)
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.relu(x)
        else:
            x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)
def mlp_model(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None, bn=False, is_training=np.bool(0)):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=tf.nn.relu)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)
def mlp_cen(xx, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):

    x = xx[0]
    other = xx[1]
    x = tf.layers.dense(x, units=hidden_sizes[0], activation=activation)
    other = tf.layers.dense(other, units=hidden_sizes[0], activation=activation)
    x = tf.layers.dense(tf.concat([x,other],-1), units=hidden_sizes[1], activation=activation)

    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]
def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])
def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)
def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


"""
Policies
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20

"""
Actor-Critics
"""
def mlp_actor_critic(n_agents, x, a, args, m1=0, m2=0, hidden_sizes=(300,300),hidden_sizes_model=(128,64,128), action_scale=1, activation=tf.nn.relu,
                     output_activation=tf.tanh, msg_dim=4, is_training=np.bool(0), max_est_time=5, msg_idx=0):

    act_dim = a[0].shape.as_list()[-1]
    obs_dim = x[0].shape.as_list()[-1]
    act_limit = 1
    msg_dim = args.msg_dim
    om_msg_dim = msg_dim


    pi, q, q_pi, msg = [], [], [], []
    model, model_trj = [], []
    def f1(): return m1  # from previous step
    def f2(): return m2  # placeholder
    mm = tf.cond(is_training, f1, f2)
    m = mm
    bn = False


    for agent in range(n_agents):
        with tf.variable_scope(str(agent) + 'pi'):
            x_temp = tf.layers.dense(tf.concat(list(chain([x[agent]], m)), axis=-1), units=hidden_sizes[0],
                                     activation=activation)
            pi.append(act_limit * mlp(x_temp, list(hidden_sizes[1:]) + [act_dim], activation, output_activation))

        if msg_idx == 0:
            with tf.variable_scope('msg' + str(agent)):
                msg.append(x[agent])

    for agent in range(n_agents):
        with tf.variable_scope(str(agent) + 'q'):
            q.append(tf.squeeze(mlp_cen([tf.concat([x[agent], a[agent]], axis=-1),
                                         tf.concat(list(chain(x[0:agent], x[agent + 1:], a[0:agent], a[agent + 1:])),
                                                   axis=-1)], list(hidden_sizes) + [1], activation, None), axis=1))
        with tf.variable_scope(str(agent) + 'q', reuse=True):
            q_pi.append(tf.squeeze(mlp_cen([tf.concat([x[agent], pi[agent]], axis=-1), tf.concat(
                list(chain(x[0:agent], x[agent + 1:], pi[0:agent], pi[agent + 1:])), axis=-1)],
                                           list(hidden_sizes) + [1], activation, None), axis=1))
    W_a, V_a, M_a = [], [], []

    om_act_all, om_msg_all = [], []
    for agent in range(n_agents):
        om_act, om_msg = [], []
        with tf.variable_scope(str(agent) + 'om'):
            for agent2 in range(n_agents):
                x_temp = tf.layers.dense(x[agent], units=hidden_sizes[0], activation=activation)
                om_act.append(
                    act_limit * mlp(x_temp, list(hidden_sizes[1:]) + [act_dim], activation, output_activation))
        om_act_all.append(om_act)

    alpha_s = []
    for agent in range(n_agents):
        with tf.variable_scope(str(agent) + 'model'):
            model.append(mlp_model(tf.concat(list(chain([x[agent]], [a[agent]], om_act_all[agent])), axis=-1),
                                   list(hidden_sizes_model) + [obs_dim], activation, output_activation, bn=bn,
                                   is_training=is_training))

        model_trj_temp, model_trj_temp2, pi_trj = [], [], []
        est_next_x_temp = x[agent] + model[agent]
        est_next_x = est_next_x_temp
        model_trj_temp.append(est_next_x)
        model_trj_temp2.append(est_next_x)
        pi_trj.append(pi[agent])
        for j in range(max_est_time - 1):
            with tf.variable_scope(str(agent) + 'pi', reuse=tf.AUTO_REUSE):
                x_temp = tf.layers.dense(tf.stop_gradient(tf.concat(list(chain([model_trj_temp[j]], m)), axis=-1)),
                                         units=hidden_sizes[0], activation=activation)
                pi_temp = act_limit * mlp(x_temp, list(hidden_sizes[1:]) + [act_dim], activation, output_activation)
            pi_trj.append(pi_temp)
            with tf.variable_scope(str(agent) + 'om', reuse=tf.AUTO_REUSE):
                om_act_temp, om_msg_temp = [], []
                for agent2 in range(n_agents):
                    x_temp = tf.layers.dense(x[agent], units=hidden_sizes[0], activation=activation)
                    om_act_temp.append(
                        act_limit * mlp(x_temp, list(hidden_sizes[1:]) + [act_dim], activation, output_activation))

            with tf.variable_scope(str(agent) + 'model', reuse=tf.AUTO_REUSE):
                est_next_x_temp = mlp_model(
                    tf.concat(list(chain([model_trj_temp[j]], [pi_temp], om_act_temp)), axis=-1),
                    list(hidden_sizes_model) + [obs_dim], activation, output_activation, bn=bn, is_training=is_training)
                est_next_x_temp = est_next_x_temp + model_trj_temp[j]
                est_next_x = est_next_x_temp
                model_trj_temp.append(est_next_x)
                model_trj_temp2.append(est_next_x)

        msg_temp = [x[agent]] + model_trj_temp2
        model_trj.append(model_trj_temp2)

        # model
        with tf.variable_scope('attention' + str(agent)):
            W_a.append(tf.get_variable("W_a", shape=[int(obs_dim) + int(act_dim), args.att_dim],
                                       initializer=layers.xavier_initializer()))
            V_a.append(tf.get_variable("V_a", shape=[int(obs_dim) + int(act_dim), args.msg_dim],
                                       initializer=layers.xavier_initializer()))
            M_a.append(tf.get_variable("M_a", shape=[args.msg_dim * (n_agents), args.att_dim],
                                       initializer=layers.xavier_initializer()))

        msg_temp_att = []
        alpha = []

        for i in range(msg_idx):
            sig_temp = tf.matmul(tf.concat([msg_temp[i], pi_trj[i]], -1), W_a[agent])
            que_temp = tf.matmul(tf.stop_gradient(tf.concat(list(chain(m[0:])), axis=-1)), M_a[agent])
            alpha_temp = tf.multiply(sig_temp, que_temp)
            alpha_temp = tf.expand_dims(tf.reduce_sum(alpha_temp, axis=1), axis=1) / math.sqrt(args.att_dim)
            alpha.append(alpha_temp)
        alpha = tf.concat(alpha, axis=1)
        alpha = tf.nn.softmax(alpha, axis=1)
        alpha_s.append(alpha)
        msg_att = 0
        for i in range(msg_idx):
            alpha__ = tf.expand_dims(alpha[:, i], axis=1)
            msg_temp_att = alpha__ * tf.matmul(tf.concat([msg_temp[i], pi_trj[i]], -1), V_a[agent])
            msg_att = msg_att + msg_temp_att
        msg.append(tf.nn.tanh(msg_att))

    om_loss = []
    for agent in range(n_agents):
        om_loss_temp = 0
        ind = 0
        for agent2 in range(n_agents):
            if agent != agent2:
                om_loss_temp = om_loss_temp + tf.reduce_mean((om_act_all[agent][ind]-pi[agent2])**2)
                ind += 1
        om_loss.append(om_loss_temp)
    return pi, q, q_pi, msg, model, model_trj, om_loss, alpha_s

