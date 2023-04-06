import numpy as np
import tensorflow as tf
from core import get_vars
from utils.logx import EpochLogger
import core
from madrl_environments.predator_prey import Predator_prey as PP


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, msg_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.prev_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.msg_pprev_buf = np.zeros([size, msg_dim], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, prev_obs, msg_pprev):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.prev_obs_buf[self.ptr] = prev_obs
        self.msg_pprev_buf[self.ptr] = msg_pprev
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, idxs, batch_size=32):
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    obs3=self.prev_obs_buf[idxs],
                    msg=self.msg_pprev_buf[idxs])

    def size(self):
        return self.size


def main(n_agents, env_, test_env_, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(),
           steps_per_epoch=10000, replay_size=int(5e5), gamma=0.99,
           polyak=0.995, lr=3e-4, batch_size=128, start_steps=0,
           logger_kwargs=dict(), act_noise = 0.1, args=None, lm =0.1, model_lr=1e-4):
    def make_feed(l_place, l_value, is_one):
        feed_dict = {}
        if is_one:
            feed_dict[l_place] = l_value
        else:
            for inpt, value in zip(l_place, l_value):
                feed_dict[inpt] = value
        return feed_dict


    act_noise = args.act_noise
    replay_size = args.replay_size


    logger = EpochLogger(**logger_kwargs)
    a = locals()
    a.pop('env_')
    a.pop('test_env_')
    logger.save_config(a)

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    env, test_env = env_, test_env_
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    msg_dim = int(args.msg_dim)



    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]
    # Share information about action space with policy architecture

    # Inputs to computation graph
    # is_training = tf.placeholder(tf.bool)
    is_training = tf.placeholder_with_default(False, shape=[])
    x_ph, a_ph, x2_ph, r_ph, d_ph = [],[],[],[],[]
    m_ph, x3_ph = [],[]
    for i in range(n_agents):
        x_ph_temp, a_ph_temp, x2_ph_temp, r_ph_temp, d_ph_temp, m_ph_temp, x3_ph_temp = core.placeholders(obs_dim, act_dim, obs_dim, None, None, msg_dim, obs_dim)

        x_ph.append(x_ph_temp)
        a_ph.append(a_ph_temp)
        x2_ph.append(x2_ph_temp)
        r_ph.append(r_ph_temp)
        d_ph.append(d_ph_temp)
        m_ph.append(m_ph_temp)
        x3_ph.append(x3_ph_temp)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        _, _, _, m_ph__, _ , _, _, _= actor_critic(n_agents, x3_ph, a_ph, args=args, m1=m_ph, m2=m_ph, **ac_kwargs, is_training = is_training,  max_est_time=args.max_est_time,  msg_idx=args.msg_idx)

    with tf.variable_scope('main', reuse=True):
        pi, q, q_pi, msg, model, model_trj, om_loss, _ = actor_critic(n_agents, x_ph, a_ph, args=args, m1=m_ph__, m2=m_ph, **ac_kwargs, is_training = is_training, msg_idx=args.msg_idx, max_est_time=args.max_est_time)

    # Target value network
    with tf.variable_scope('target'):
        pi_targ, _, q_pi_targ, _, _, _ , _, _ = actor_critic(n_agents, x2_ph, a_ph, args=args, m1=m_ph__, m2=m_ph, **ac_kwargs, is_training = is_training, msg_idx=args.msg_idx, max_est_time=args.max_est_time)

    # Experience buffer

    replay_buffer = []
    for agent in range(n_agents):
        replay_buffer.append(ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, msg_dim=msg_dim, size=replay_size))

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in
                       ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n')%var_counts)

    min_q_pi = []
    q_backup, v_backup, pi_loss, q1_loss, q2_loss, v_loss, value_loss, min_q_pi, \
    pi_optimizer,train_pi_op,value_optimizer,value_params,train_value_op,target_update,step_ops,target_init \
        =[], [],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

    backup,q_loss, pi_loss,pi_optimizer,train_pi_op  = [],[],[],[],[]
    value_optimizer, value_params = [],[]
    model_params = []
    train_om_op, om_optimizer, om_params = [], [], []
    model_target, model_loss, train_model_op, model_optimizer = [], [], [], []
    update_ops = []
    for agent in range(n_agents):
        backup.append(tf.stop_gradient(r_ph[agent] + gamma*(1-d_ph[agent])*q_pi_targ[agent]))
        pi_loss.append(-tf.reduce_mean(q_pi[agent]))
        q_loss.append(tf.reduce_mean((q[agent]-backup[agent])**2))
        pi_optimizer.append(tf.train.AdamOptimizer(learning_rate=lr))
        train_pi_op.append(pi_optimizer[agent].minimize(pi_loss[agent], var_list=get_vars('main/'+str(agent)+'pi')+get_vars('main/attention')))
        value_optimizer.append(tf.train.AdamOptimizer(learning_rate=lr))
        train_value_op.append(value_optimizer[agent].minimize(q_loss[agent], var_list=get_vars('main/'+str(agent)+'q')))

        model_params.append(get_vars('main/'+str(agent)+'model'))
        model_target.append((x2_ph[agent]-x_ph[agent]))

        model_loss.append(1*tf.reduce_mean((model_target[agent]-model[agent])**2)*(1-d_ph[agent]))
        model_optimizer.append(tf.train.AdamOptimizer(learning_rate=model_lr))
        om_params.append(get_vars('main/'+str(agent)+'om'))
        om_optimizer.append(tf.train.AdamOptimizer(learning_rate=model_lr))
        update_ops.append(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

        with tf.control_dependencies(update_ops[agent]):
            train_model_op.append(model_optimizer[agent].minimize(model_loss[agent], var_list=model_params[agent]))
            train_om_op.append(om_optimizer[agent].minimize(om_loss[agent], var_list=om_params[agent]))


        with tf.control_dependencies([train_value_op[agent]]):
            target_update.append(tf.group([tf.assign(v_targ_temp, polyak*v_targ_temp + (1-polyak)*v_main_temp)
                                           for v_main_temp, v_targ_temp in zip(get_vars('main/'+str(agent)), get_vars('target/'+str(agent)))]))

        target_init.append(tf.group([tf.assign(v_targ_temp, v_main_temp)
                                     for v_main_temp, v_targ_temp in zip(get_vars('main/'+str(agent)), get_vars('target/'+str(agent)))]))

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for agent in range(n_agents):
        sess.run(target_init[agent])


    def get_action_message_model_trj(o, noise_scale, msgg):
        o_ = list(np.reshape(o, [n_agents, 1, -1]))
        msgg = list(np.reshape(msgg, [n_agents, 1, -1]))
        is_t = np.bool(0)
        list_feed = x_ph + m_ph + [is_training] + x3_ph + a_ph
        value_feed = o_ + msgg + [is_t] + o_+ list(np.zeros([n_agents,1 , act_dim]))
        feed_dict = make_feed(list_feed, value_feed, 0)
        a = sess.run(pi, feed_dict)

        for i in range (len(a)):
            a[i] += noise_scale * np.random.randn(act_dim)
            a[i] = np.clip(a[i], -act_limit, act_limit)

        a_ = list(np.reshape(a, [n_agents, 1, -1]))
        list_feed2 = x_ph + m_ph + [is_training] + x3_ph + a_ph
        value_feed2 = o_ + msgg + [is_t] + o_ +  a_
        feed_dict2 = make_feed(list_feed2, value_feed2, 0)
        msg_temp, mod, mod_trj = sess.run([msg, model, model_trj], feed_dict2)

        return a, msg_temp, mod, mod_trj


    def test_agent(n=10):
        global sess, mu, pi, q1, q2, q1_pi, q2_pi
        for j in range(n):
            o, r, d, ep_ret, ep_len,ep_ret2 = test_env.reset(), 0, False, 0, 0,0
            msg_prev = np.zeros([n_agents, msg_dim])
            time = 0
            num_c = 0
            while not(d or (ep_len == args.max_ep_len)):
                # Take deterministic actions at test time

                a, msg_temp, mod_temp, mod_temp_trj = get_action_message_model_trj(o, 0, msg_prev)

                o2, r, d, di , r2= test_env.step(a)
                o = np.copy(o2)
                num_c += di
                ep_ret += np.average(r)
                ep_len += 1
                ep_ret2 += np.average(r2)
                msg_prev = np.copy(msg_temp)
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len, TestEpRet2=ep_ret2, test_num_catch=num_c)


    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    o2 = np.copy(o)
    o_prev = np.copy(o)

    # o_prev = np.copy(o)
    msg_temp = np.zeros([n_agents, msg_dim])
    msg_prev = np.zeros([n_agents, msg_dim])
    msg_pprev = np.zeros([n_agents, msg_dim])
    reset = 1
    dist, ep_ret2 = 0 , 0
    total_steps = steps_per_epoch * args.epochs
    o2_esti = [0,0]
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        if t > args.start_steps:
            a, msg_temp, mod_temp, _ = get_action_message_model_trj(o, act_noise, msg_prev)
            o2_esti = np.copy(o2) + np.copy(mod_temp)[0,:,0]
        else:
            a = []
            for agent in range(n_agents):
                a.append(env.action_space.sample())
            msg_temp = np.zeros([n_agents, msg_dim])

        o2, r, d, di , r2 = env.step(a)

        ep_ret += np.average(r)
        ep_len += 1
        dist += di
        ep_ret2 += np.average(r2)

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==args.max_ep_len else d

        if reset == 0:
            for agent in range(n_agents):
                replay_buffer[agent].store(o[agent], a[agent], r[agent], o2[agent], d, o_prev[agent], msg_pprev[agent])
            msg_pprev = np.copy(msg_prev)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o_prev = np.copy(o)
        o = np.copy(o2)

        reset = 0
        msg_prev = np.copy(msg_temp)



        if d or (ep_len == args.max_ep_len):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            for j in range(ep_len):
                if j % args.update_freq == 0:
                    bf_size = replay_buffer[0].size
                    idxs = np.random.randint(0, bf_size, size=batch_size)
                    o_b, o_n_b, r_b, d_b, a_b = [],[],[],[],[]
                    o_p_b, m_pp_b = [], []
                    for agent in range(n_agents):
                        batch = replay_buffer[agent].sample_batch(idxs)
                        o_b.append(batch['obs1'])
                        o_n_b.append(batch['obs2'])
                        r_b.append(batch['rews'])
                        d_b.append(batch['done'])
                        a_b.append(batch['acts'])
                        o_p_b.append(batch['obs3'])
                        m_pp_b.append(batch['msg'])

                    is_t = np.bool(1)
                    list_feed = x_ph + x2_ph + a_ph + r_ph + d_ph + x3_ph + m_ph + [is_training]
                    value_feed = o_b + o_n_b + a_b + r_b + d_b + o_p_b + m_pp_b + [is_t]
                    feed_dict = make_feed(list_feed, value_feed, 0)
                    outs_all = sess.run([q_loss, q, train_value_op, pi_loss, train_pi_op, target_update, model_loss, train_model_op,om_loss, train_om_op], feed_dict)
                    # logger.store(LossQ=outs_all[0], QVals=outs_all[1], LossPi=outs_all[3])
                    for agent in range(n_agents):
                        outs = outs_all[agent]
                        logger.store(LossQ=outs_all[0][agent], QVals=outs_all[1][agent], LossPi=outs_all[3][agent])
                        logger.store(ModelLoss=outs_all[6][agent])


            #
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            if args.env == 'CN'+str(args.a_sensor_range):
                logger.store(Dist=dist)
            o, r, d, ep_ret, ep_len , ep_ret2= env.reset(), 0, False, 0, 0, 0
            logger.store(num_catch=dist)
            dist = 0
            o2 = np.copy(o)
            reset = 1
            msg_prev = np.zeros([n_agents, msg_dim])


        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch
            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('ModelLoss', average_only=True)
            epret = logger.log_tabular('EpRet', average_only=True, ret=True)
            test_epret = logger.log_tabular('TestEpRet',average_only=True, ret=True)
            print("args : ", args)



            ####################################################
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PP')

    parser.add_argument('--update_freq', type=int, default=4)
    parser.add_argument('--max_est_time', type=int, default=5)
    parser.add_argument('--mod_pred_time', type=int, default=10, help="This is only for PP")

    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--exp_name', type=str, default='IS')
    parser.add_argument('--save_model', type=bool, default=False)

    parser.add_argument('--replay_size', type=int, default=int(2e5), help="This is only for PP0.7")
    parser.add_argument('--touch_reward', type=float, default=0, help="touch_reward")
    parser.add_argument('--radius', type=float, default=0.1, help="raius") #original 0.1
    parser.add_argument('--action_scale', type=float, default=0.2, help="action scale") #original 0.05
    parser.add_argument('--a_sensor_range', type=float, default=5, help="a_sensor_range")

    parser.add_argument('--obs_type', type=int, default=5, help="This is only for PP0.7")
    parser.add_argument('--food_reward', type=float, default=1, help="a_sensor_range")
    parser.add_argument('--time_reward', type=float, default=-0.01, help="a_sensor_range")
    parser.add_argument('--rew_type', type=int, default=2)
    parser.add_argument('--act_noise', type=float, default=0.15)
    parser.add_argument('--att_dim', type=int, default=5, help="Dim of Attention feature")
    parser.add_argument('--msg_dim', type=int, default=5, help="Dim of Message")
    parser.add_argument('--max_ep_len', type=float, default=50.)
    parser.add_argument('--start_steps', type=int, default=10000)



    parser.add_argument('--n_agents', type=int, default=4)
    parser.add_argument("--n_targets", type=int, default=9, help="number of evaders")
    parser.add_argument('--n_coop', type=int, default=1, help="# agents to catch a prey")
    parser.add_argument('--epochs', type=int, default=301)
    parser.add_argument('--msg_idx', type=int, default=5, help="Rollout time-steps of ITGM")
    parser.add_argument('--seed', '-s', type=int, default=0)

    args = parser.parse_args()
    args.max_est_time = args.msg_idx + 1
    args.mod_pred_time = args.msg_idx + 1


    if args.n_agents >= 4:
        args.n_coop = 2
        args.radius = 0.15
        args.max_ep_len = 100


    ########################################################################################################

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    env = PP(args.n_agents, args.n_targets, args.n_coop, sensor_range=args.a_sensor_range, obs_type=args.obs_type,
             radius=args.radius, action_scale=args.action_scale, food_reward=args.food_reward,
             touch_reward=args.touch_reward, reward_type=args.rew_type)
    env_test = PP(args.n_agents, args.n_targets, args.n_coop, sensor_range=args.a_sensor_range, obs_type=args.obs_type,
                  radius=args.radius, action_scale=args.action_scale, food_reward=args.food_reward,
                  touch_reward=args.touch_reward, reward_type=args.rew_type)


    main(args.n_agents, env, env_test, actor_critic=core.mlp_actor_critic,
           ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
           logger_kwargs=logger_kwargs, args=args)
