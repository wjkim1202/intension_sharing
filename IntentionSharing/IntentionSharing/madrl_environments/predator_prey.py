import numpy as np
import scipy.spatial.distance as ssd
from gym import spaces
from gym.utils import seeding
import random

from madrl_environments import AbstractMAEnv, Agent
from rltools.util import EzPickle

class Archea(Agent):

    def __init__(self, idx, radius, n_sensors, sensor_range):
        self._idx = idx
        self._radius = radius
        self._n_sensors = n_sensors
        self._sensor_range = sensor_range
        # Number of observation coordinates from each sensor

        self._position = None
        self._velocity = None
        # Sensors
        angles_K = np.linspace(0., 2. * np.pi, self._n_sensors + 1)[:-1]
        sensor_vecs_K_2 = np.c_[np.cos(angles_K), np.sin(angles_K)]
        self._sensors = sensor_vecs_K_2

    @property
    def observation_space(self):
        return spaces.Box(low=-10, high=10, shape=(self._obs_dim,))

    @property
    def action_space(self):
        return spaces.Box(low=-1, high=1, shape=(2,))

    @property
    def position(self):
        assert self._position is not None
        return self._position

    @property
    def velocity(self):
        assert self._velocity is not None
        return self._velocity

    def set_position(self, x_2):
        assert x_2.shape == (2,)
        self._position = x_2

    def set_velocity(self, v_2):
        assert v_2.shape == (2,)
        self._velocity = v_2

    @property
    def sensors(self):
        assert self._sensors is not None
        return self._sensors

    def sensed(self, objx_N_2, same=False):
        """Whether `obj` would be sensed by the pursuers"""
        relpos_obj_N_2 = objx_N_2 - np.expand_dims(self.position, 0)
        sensorvals_K_N = self.sensors.dot(relpos_obj_N_2.T)
        sensorvals_K_N[(sensorvals_K_N < 0) | (sensorvals_K_N > self._sensor_range) | ((
                                                                                               relpos_obj_N_2**2).sum(axis=1)[None, :] - sensorvals_K_N**2 > self._radius**2)] = np.inf
        if same:
            sensorvals_K_N[:, self._idx - 1] = np.inf
        return sensorvals_K_N


class Predator_prey(AbstractMAEnv, EzPickle):

    def __init__(self, n_pursuers, n_evaders, n_coop, radius=0.1,
                 n_sensors=20, sensor_range=1, action_scale=0.05,
                 food_reward=10, touch_reward=0., obs_type = 5, global_reward = True, state=False, reward_type=1,
                 env_ver=1, circle=0.3, seed=0, **kwargs):
        EzPickle.__init__(self, n_pursuers, n_evaders, n_coop, radius,
                          n_sensors, sensor_range,
                          action_scale,  food_reward, touch_reward,obs_type, global_reward, state, reward_type,
                          env_ver,  **kwargs)

        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.n_coop = n_coop
        self.obs_range = sensor_range
        self.action_scale = action_scale
        self.obs_type = obs_type
        self.global_reward = global_reward
        self.state = state
        self.reward_type = reward_type



        self.n_sensors = n_sensors
        self.sensor_range = np.ones(self.n_pursuers) * sensor_range
        self.food_reward = food_reward
        self.touch_reward = touch_reward
        self.circle = circle
        self.radius = radius
        self.all_catched = 1


        self.evader_pos_list = []
        for i in range(self.n_evaders):
            self.evader_pos_list.append(
                [0.5 + self.circle*np.cos(np.pi * 2 / self.n_evaders * i), 0.5 + self.circle*np.sin(np.pi * 2 / self.n_evaders * i)])

        self.pu_p = 0

        self.n_obstacles = 1
        self.seed(seed)
        self._pursuers = [
            Archea(npu + 1, self.radius, self.n_sensors, self.sensor_range[npu]) for npu in range(self.n_pursuers)
        ]
        self._evaders = [
            Archea(nev + 1, self.radius/4, self.n_pursuers, self.sensor_range.mean() / 2)
            for nev in range(self.n_evaders)
        ]

        pos_sq = []
        for pp in range(int(np.sqrt(self.n_evaders))):
            p1 = pp * 0.8/(np.sqrt(self.n_evaders)-1) + 0.1
            for pp2 in range(int(np.sqrt(self.n_evaders))):
                p2 = pp2 * 0.8/(np.sqrt(self.n_evaders)-1) + 0.1
                pos_sq.append([p1, p2])
        self.pos_sq = pos_sq

        self._obs_dim = (self.n_pursuers+ self.n_evaders)*2
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self._obs_dim,))

        if self.n_coop == 1:
            self.init_same_position = True
        else:
            self.init_same_position = False


    @property
    def reward_mech(self):
        return self._reward_mech



    @property
    def agents(self):
        return self._pursuers
    def get_param_values(self):
        return self.__dict__
    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]
    def _respawn(self, objx_2, radius):
        objx_2 = self.np_random.rand(2)
        return objx_2

    def reset(self):
        self.is_removed = 0
        self.all_catched = 1
        # Initialize obstacles

        if self.init_same_position:
            # Initialize pursuers
            while True:
                pos_temp =self.np_random.rand(2)
                rea = 0
                for pp in range(len(self.pos_sq)):
                    d_t = np.linalg.norm(self.pos_sq[pp]-pos_temp)
                    if d_t > self.radius * (1.5):
                        rea += 1
                if rea == len(self.pos_sq):
                    break
            for pursuer in self._pursuers:
                pursuer.set_position(pos_temp)
                pursuer.set_velocity(np.zeros(2))
        else:
            # Initialize pursuers
            for pursuer in self._pursuers:
                while True:
                    pos_temp = self.np_random.rand(2)
                    rea = 0
                    for pp in range(len(self.pos_sq)):
                        d_t = np.linalg.norm(self.pos_sq[pp] - pos_temp)
                        if d_t > self.radius * (1.5):
                            rea += 1
                    if rea == len(self.pos_sq):
                        break
                pursuer.set_position(pos_temp)
                pursuer.set_velocity(np.zeros(2))

        init_eva_pos = []
        eva_pos_ind = []
        eva_pos_one = [[self.radius/2, self.radius/2], [1-self.radius/2, self.radius/2], [1-self.radius/2, 1-self.radius/2], [self.radius/2, 1-self.radius/2]]
        eva_pos_other = [[0,0], [0.5,0], [0,0.5], [0.5,0.5]]
        if self.obs_type==3:
            one_ind = np.random.randint(4)
            other_ind = np.random.randint(4)
        for i in range(self.n_evaders):
            if self.obs_type==3:
                if i == 0:
                    temp = eva_pos_one[one_ind]
                else:
                    temp = eva_pos_other[other_ind] + 0.5*self.np_random.rand(2)
            elif self.obs_type == 5 or self.obs_type ==7 or self.obs_type ==9:
                temp = self.pos_sq[i]
            else:
                temp = self.np_random.rand(2)
            temp2 = abs(temp[0]) + abs(temp[1])
            init_eva_pos.append(temp)
            eva_pos_ind.append(temp2)

        eva_pos_ind2 = np.argsort(eva_pos_ind)
        if self.obs_type % 2 == 0:
            init_eva_pos = np.array(init_eva_pos)[eva_pos_ind2]
        else:
            init_eva_pos = np.array(init_eva_pos)

        # Initialize evaders
        ii = 0
        for evader in self._evaders:
            evader.set_position(init_eva_pos[ii])
            evader.set_velocity(np.zeros(2))
            ii += 1

        return self.step(np.zeros((self.n_pursuers, 2)))[0]



    def _caught(self, is_colliding_N1_N2, n_coop):
        """ Checke whether collision results in catching the object
        This is because you need `n_coop` agents to collide with the object to actually catch it
        """
        # number of N1 colliding with given N2
        n_collisions_N2 = is_colliding_N1_N2.sum(axis=0)

        # get reward
        is_caught_cN2 = np.where(n_collisions_N2 >= 1)[0]
        who_caught_cN1_touch = []
        for cN2 in is_caught_cN2:
            who_collisions_N1_cN2_temp = is_colliding_N1_N2[:, cN2]
            who_caught_cN1_temp = np.where(who_collisions_N1_cN2_temp >= 1)[0]
            who_caught_cN1_touch.extend(who_caught_cN1_temp)



        is_caught_cN2 = np.where(n_collisions_N2 >= n_coop)[0]
        who_caught_cN1 = []
        for cN2 in is_caught_cN2:
            who_collisions_N1_cN2_temp = is_colliding_N1_N2[:, cN2]
            who_caught_cN1_temp = np.where(who_collisions_N1_cN2_temp >= 1)[0]
            if len(who_caught_cN1_temp)>n_coop:
                random.shuffle(who_caught_cN1_temp)
                who_caught_cN1.extend(who_caught_cN1_temp[0:int(n_coop)])
            else:
                who_caught_cN1.extend(who_caught_cN1_temp)

        # print("who_caught_cN1 : ", who_caught_cN1)
        # print("who_caught_cN1_touch : ", who_caught_cN1_touch)
        return is_caught_cN2, who_caught_cN1, who_caught_cN1_touch

    def _closest_dist(self, closest_obj_idx_Np_K, sensorvals_Np_K_N):
        """Closest distances according to `idx`"""
        sensorvals = []
        for inp in range(self.n_pursuers):
            sensorvals.append(sensorvals_Np_K_N[inp, ...][np.arange(self.n_sensors),
                                                          closest_obj_idx_Np_K[inp, ...]])
        return np.c_[sensorvals]
    def step(self, action_Np2):
        action_Np2 = np.asarray(action_Np2)
        action_Np_2 = action_Np2.reshape((self.n_pursuers, 2))
        # Players
        actions_Np_2 = action_Np_2 * self.action_scale

        rewards = np.zeros((self.n_pursuers,))
        assert action_Np_2.shape == (self.n_pursuers, 2)

        for npu, pursuer in enumerate(self._pursuers):
            pursuer.set_velocity(actions_Np_2[npu])
            pursuer.set_position(pursuer.position + pursuer.velocity)

        # Players stop on hitting a wall
        for npu, pursuer in enumerate(self._pursuers):
            clippedx_2 = np.clip(pursuer.position, 0, 1)
            vel_2 = pursuer.velocity
            vel_2[pursuer.position != clippedx_2] = 0
            pursuer.set_velocity(vel_2)
            pursuer.set_position(clippedx_2)


        # Find collisions
        pursuersx_Np_2 = np.array([pursuer.position for pursuer in self._pursuers])
        evadersx_Ne_2 = np.array([evader.position for evader in self._evaders])

        positions_all = np.concatenate([pursuersx_Np_2,evadersx_Ne_2])
        pos_dist = ssd.cdist(positions_all, positions_all)
        is_sensed = np.zeros([self.n_pursuers+self.n_evaders, self.n_pursuers+self.n_evaders])
        for i in range(self.n_pursuers+self.n_evaders):
            for j in range(self.n_pursuers+self.n_evaders):
                if i >= j:
                    if pos_dist[i][j] < self.obs_range:
                        is_sensed[i][j] = 1
                        is_sensed[j][i] = 1

        # Evaders
        evdists_Np_Ne = ssd.cdist(pursuersx_Np_2, evadersx_Ne_2)
        is_colliding_ev_Np_Ne = evdists_Np_Ne <= np.asarray([
            pursuer._radius + evader._radius for pursuer in self._pursuers
            for evader in self._evaders
        ]).reshape(self.n_pursuers, self.n_evaders)

        # num_collisions depends on how many needed to catch an evader
        ev_caught, which_pursuer_caught_ev, which_pursuer_touch_ev = self._caught(is_colliding_ev_Np_Ne, self.n_coop)
        self.is_removed += len(ev_caught)
        for evcaught in ev_caught:
            self._evaders[evcaught].set_position(np.array([1000, 1000]))
            self._evaders[evcaught].set_velocity(np.zeros(2))




        # Update reward based on these collisions

        rewards[which_pursuer_touch_ev] += self.touch_reward

        if self.global_reward:
            for ll in range(len(rewards)):
                rewards[ll] += self.food_reward * len(ev_caught) * self.all_catched
        else:
            rewards[which_pursuer_caught_ev] += (self.food_reward * self.all_catched)

        reset__ = 0
        if self.is_removed == self.n_evaders:
            # print("rest ")
            # print("elf.is_removed ; ", self.is_removed)
            reset__ = 1
            init_eva_pos = []
            eva_pos_ind = []
            eva_pos_one = [[self.radius/2, self.radius/2], [1-self.radius/2, self.radius/2], [1-self.radius/2, 1-self.radius/2], [self.radius/2, 1-self.radius/2]]
            eva_pos_other = [[0,0], [0.5,0], [0,0.5], [0.5,0.5]]
            if self.obs_type==3:
                one_ind = np.random.randint(4)
                other_ind = np.random.randint(4)
            for i in range(self.n_evaders):
                if self.obs_type==3:
                    if i == 0:
                        temp = eva_pos_one[one_ind]
                    else:
                        temp = eva_pos_other[other_ind] + 0.5*self.np_random.rand(2)
                elif self.obs_type == 5 or self.obs_type == 7:
                    temp = self.pos_sq[i]
                else:
                    temp = self.np_random.rand(2)
                temp2 = abs(temp[0]) + abs(temp[1])
                init_eva_pos.append(temp)
                eva_pos_ind.append(temp2)

            eva_pos_ind2 = np.argsort(eva_pos_ind)
            if self.obs_type % 2 == 0:
                init_eva_pos = np.array(init_eva_pos)[eva_pos_ind2]
            else:
                init_eva_pos = np.array(init_eva_pos)
            self.is_removed = 0
            if self.reward_type == 1:
                self.all_catched  = self.all_catched
            elif self.reward_type ==2:
                self.all_catched  = self.all_catched + 1
            elif self.reward_type ==3:
                self.all_catched  = self.all_catched * 2

            ii = 0
            for evader in self._evaders:
                evader.set_position(init_eva_pos[ii])
                # evader.set_position(self._respawn(evader.position, evader._radius))
                evader.set_velocity(np.zeros(2))  # TODO policies
                ii += 1

        if self.obs_type == 5:
            obslist = []
            for inp in range(self.n_pursuers):
                positions_all_temp = np.copy(positions_all)
                for l in range(len(is_sensed[inp])):
                    if is_sensed[inp][l] == 0:
                        if positions_all_temp[l][0] > 10:
                            positions_all_temp[l] = positions_all_temp[l]*is_sensed[inp][l]
                        else:
                            positions_all_temp[l] = np.array([0,0])
                    else:
                        positions_all_temp[l] = positions_all_temp[l]*is_sensed[inp][l]
                positions_all_temp = np.reshape(positions_all_temp, -1)
                obslist.append(
                    positions_all_temp)

        elif self.obs_type == 7:
            obslist = []
            for inp in range(self.n_pursuers):
                positions_all_temp = np.copy(positions_all)
                if self.obs_type>0:
                    position = positions_all_temp[inp]
                for l in range(len(is_sensed[inp])):
                    # print("is_sensed[inp][l]: ", is_sensed[inp][l])
                    if self.obs_type>0:
                        if l != inp:
                            if positions_all_temp[l][0] > 2 or is_sensed[inp][l]<0.5:
                                positions_all_temp[l] = np.array([3,3])
                            else:
                                positions_all_temp[l] = positions_all_temp[l] - position
                                positions_all_temp[l] = positions_all_temp[l]
                    # positions_all_temp[l] = positions_all_temp[l]*is_sensed[inp][l]

                positions_all_temp = np.reshape(positions_all_temp, -1)
                obslist.append(
                    positions_all_temp)

        done = False
        if self.state:
            return obslist, rewards, done, len(ev_caught), 1, np.reshape(positions_all, -1)
        else:
            return obslist, rewards, done, len(ev_caught), reset__

    def render(self, screen_size=800, rate=10, mode='human'):
        import cv2
        img = np.empty((screen_size, screen_size, 3), dtype=np.uint8)
        img[...] = 255
        # Pursuers
        for pursuer in self._pursuers:
            for k in range(pursuer._n_sensors):
                color = (0, 0, 0)
                cv2.circle(img,
                           tuple((pursuer.position * screen_size).astype(int)),
                           int(pursuer._radius * screen_size), (255, 0, 0), -1)
        # Evaders
        for evader in self._evaders:
            color = (0, 255, 0)
            cv2.circle(img,
                       tuple((evader.position * screen_size).astype(int)),
                       int(evader._radius * screen_size), color, -1)

        opacity = 0.4
        bg = np.ones((screen_size, screen_size, 3), dtype=np.uint8) * 255
        cv2.addWeighted(bg, opacity, img, 1 - opacity, 0, img)
        cv2.imshow('Waterworld', img)
        cv2.waitKey(rate)
        return np.asarray(img)[..., ::-1]