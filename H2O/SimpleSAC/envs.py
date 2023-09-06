'''
  Generate different type of dynamics mismatch.
  @python version : 3.6.4
'''

import numpy as np
import traci
from sumolib import checkBinary
from datetime import datetime
import time
import math
import os
from ego_policy.fvdm import fvdm_model
from utils.car_dis_comput import dist_between_cars


class Env:
    def __init__(self, realdata_path=None, genedata_path=None, num_agents=1, dt=0.04, sim_horizon=200,
                 r_ego='r1', r_adv='r1', ego_policy="sumo", adv_policy="sumo", sim_seed=42, gui=False):
        self.state_space = [(num_agents + 1) * 4]
        self.action_space_ego = [2]
        self.action_space_adv = [num_agents * 2]
        self.num_agents = num_agents
        self.dt = dt
        self.sim_horizon = sim_horizon
        self.max_speed = 40
        self.r_ego = r_ego
        self.r_adv = r_adv

        self.car_info = []
        for i in range(num_agents + 1):
            self.car_info.append([i, "lane", [5, 1.8]])

        self.road_info = {"lane": [0, 12]}

        # self.realdata_path = realdata_path
        if realdata_path is not None:
            realdata_filelist = []
            for f in os.listdir(realdata_path):
                for ff in os.listdir(realdata_path + f):
                    realdata_filelist.append(realdata_path + f + '/' + ff)
            self.realdata_filelist = realdata_filelist
        self.genedata_path = genedata_path

        self.ego_policy = ego_policy
        self.adv_policy = adv_policy

        self.up_cut = [0.6 * 9.8 * self.dt, math.pi / 3 * self.dt]
        self.low_cut = [-0.8 * 9.8 * self.dt, -math.pi / 3 * self.dt]

        self.gui = gui
        if self.gui:
            self.app = 'sumo-gui'
            self.cfg_sumo = 'config/lane_sim.sumocfg'
            self.command = [checkBinary(self.app), '-c', self.cfg_sumo]
        else:
            self.app = 'sumo'
            self.cfg_sumo = 'config/lane.sumocfg'
            self.command = [checkBinary(self.app), "--start", '-c', self.cfg_sumo]
        self.sim_seed = sim_seed
        self.command += ['--routing-algorithm', 'dijkstra']
        # self.command += ['--collision.action', 'remove']
        self.command += ['--seed', str(self.sim_seed)]
        self.command += ['--no-step-log', 'True']
        self.command += ['--time-to-teleport', '300']
        self.command += ['--no-warnings', 'True']
        self.command += ['--duration-log.disable', 'True']
        self.command += ['--waiting-time-memory', '1000']
        self.command += ['--eager-insert', 'True']
        self.command += ['--lanechange.duration', '1.5']
        self.command += ['--lateral-resolution', '0.0']

    def reset(self, scenario=None):
        self.ArrivedVehs = []
        self.CollidingVehs = []
        self.timestep = 0
        self.ego_col_cost_record = []
        for i in range(self.num_agents):
            self.ego_col_cost_record.append(0)
        self.adv_col_cost_record = float('inf')
        self.adv_dev_cost_record = 0
        self.states = []
        self.actions = []
        for t in range(self.sim_horizon + 1):
            state_t = []
            action_t = []
            for id in range(self.num_agents + 1):
                state_t.extend([0, 0, 0, 0])
                action_t.extend([0, 0])
            self.states.append(state_t)
            self.actions.append(action_t)

        if scenario is not None:
            dataset = np.loadtxt(scenario)
            observations = []
            observation = []
            t = 0
            for i in range(len(dataset)):
                state = dataset[i]
                [timestep, carid, typeid, x, y, speed, yaw, delta_speed, delta_yaw] = state
                if timestep != t:
                    t = timestep
                    observations.append(observation)
                    observation = []
                observation.extend([x, y, speed, yaw])
            observations.append(observation)
        elif (self.ego_policy == "genedata" or self.adv_policy == "genedata") and np.random.rand() < 0.5:
            genedata_filelist = os.listdir(self.genedata_path)
            filepath = self.genedata_path + genedata_filelist[np.random.randint(0, len(genedata_filelist))]
            dataset = np.loadtxt(filepath)
            observations = []
            observation = []
            t = 0
            for i in range(len(dataset)):
                state = dataset[i]
                [timestep, carid, typeid, x, y, speed, yaw, delta_speed, delta_yaw] = state
                if timestep != t:
                    t = timestep
                    observations.append(observation)
                    observation = []
                observation.extend([x, y, speed, yaw])
            observations.append(observation)
        else:
            self.ego_policy = "sumo"
            while True:
                filepath = self.realdata_filelist[np.random.randint(0, len(self.realdata_filelist))]
                dataset = np.load(filepath, allow_pickle=True).item()
                observations = dataset['observations']
                if len(observations) > 1:
                    break
        self.states[0] = observations[0]
        for i in range(self.num_agents + 1):
            self.states[0][i * 4 + 2] = min(self.states[0][i * 4 + 2], self.max_speed)
        for vehicle in traci.vehicle.getIDList():
            traci.vehicle.remove(vehicle)

        cur_time = float(traci.simulation.getTime())

        # ego
        if self.ego_policy == "uniform":
            self.states[0][3] = 0
        if self.ego_policy == "sumo":
            traci.vehicle.add(vehID="car0", routeID="straight", typeID="AV",
                              depart=cur_time, departLane=0, departPos=10.0, arrivalLane=np.random.randint(0, 3),
                              departSpeed=self.states[0][2])
        else:
            traci.vehicle.add(vehID="car0", routeID="straight", typeID="AV",
                              depart=cur_time, departLane=0, departPos=10.0,
                              departSpeed=self.states[0][2])
        traci.vehicle.moveToXY(vehID="car0",
                               x=self.states[0][0], y=self.states[0][1], angle=-self.states[0][3] * 180 / np.pi + 90,
                               lane=0, edgeID=0)
        if self.ego_policy == "fvdm":
            self.ego_policy_model = fvdm_model("car0")
        if self.ego_policy == "genedata" or self.ego_policy =="realdata":
            self.ego_policy_model = np.array(observations)[1:, 0: 4]

        # adv
        if self.adv_policy == "uniform":
            for i in range(self.num_agents):
                self.states[0][4 * (i + 1) + 3] = 0
        if self.adv_policy == "sumo":
            for i in range(self.num_agents):
                traci.vehicle.add(vehID="car" + str(i + 1), routeID="straight", typeID="BV",
                                  depart=cur_time, departLane=1, departPos=40.0,
                                  arrivalLane=np.random.randint(0, 3), arrivalPos=np.inf,
                                  departSpeed=self.states[0][4 * (i + 1) + 2])
        else:
            for i in range(self.num_agents):
                traci.vehicle.add(vehID="car" + str(i + 1), routeID="straight", typeID="BV",
                                  depart=cur_time, departLane=1,
                                  departPos=40.0, arrivalPos=np.inf,
                                  departSpeed=self.states[0][4 * (i + 1) + 2])
        for i in range(self.num_agents):
            traci.vehicle.moveToXY(vehID="car" + str(i + 1),
                                   x=self.states[0][4 * (i + 1)], y=self.states[0][4 * (i + 1) + 1],
                                   angle=-self.states[0][4 * (i + 1) + 3] * 180 / np.pi + 90,
                                   lane=0, edgeID=0)
        self.adv_policy_model = []
        if self.adv_policy == "fvdm":
            for i in range(self.num_agents):
                self.adv_policy_model.append(fvdm_model("car" + str(i + 1)))
        if self.adv_policy == "genedata" or self.ego_policy == "realdata":
            for i in range(self.num_agents):
                self.adv_policy_model.append(np.array(observations)[1:, 4 * (i + 1): 4 * (i + 2)])

        self.max_step = len(observations) - 1 if self.ego_policy == "genedata" or self.ego_policy == "realdata" \
                                                 or self.adv_policy == "genedata" or self.adv_policy == "realdata" \
            else self.sim_horizon
        traci.simulationStep()

        return self.states[0]

    def traci_start(self):
        traci.start(self.command)

    def traci_close(self):
        traci.close()

    def get_state(self, car_index):
        return self.states[self.timestep][car_index * 4: (car_index + 1) * 4]

    def set_state(self, state, car_index):
        self.states[self.timestep + 1][car_index * 4: (car_index + 1) * 4] = state

    def step(self, action_ego, action_adv):
        # ego
        if self.ego_policy == "uniform":
            av_action = [0, 0]
            new_state = self.motion_model(self.get_state(0), av_action)
            traci.vehicle.moveToXY(vehID="car0",
                                   x=new_state[0],
                                   y=new_state[1],
                                   angle=-new_state[3] * 180 / np.pi + 90,
                                   lane=0, edgeID=0)
            traci.vehicle.setSpeed(vehID='car0', speed=new_state[2])
            # traci.simulationStep()
        elif self.ego_policy == "fvdm":
            self.ego_policy_model.run()
        elif self.ego_policy == "RL" or self.ego_policy == "model":
            new_state = self.motion_model(self.get_state(0), action_ego)
            traci.vehicle.moveToXY(vehID="car0",
                                   x=new_state[0],
                                   y=new_state[1],
                                   angle=-new_state[3] * 180 / np.pi + 90,
                                   lane=0, edgeID=0)
            traci.vehicle.setSpeed(vehID='car0', speed=new_state[2])
        if self.ego_policy == "genedata" or self.ego_policy == "realdata":
            new_state = self.ego_policy_model[self.timestep]
            traci.vehicle.moveToXY(vehID="car0",
                                   x=new_state[0],
                                   y=new_state[1],
                                   angle=-new_state[3] * 180 / np.pi + 90,
                                   lane=0, edgeID=0)
            traci.vehicle.setSpeed(vehID='car0', speed=new_state[2])

        # adv
        if self.adv_policy == "uniform":
            for i in range(self.num_agents):
                new_state = self.motion_model(self.get_state(i + 1), [0, 0])
                traci.vehicle.moveToXY(vehID="car" + str(i + 1),
                                       x=new_state[0],
                                       y=new_state[1],
                                       angle=-new_state[3] * 180 / np.pi + 90,
                                       lane=0, edgeID=0)
                traci.vehicle.setSpeed(vehID="car" + str(i + 1), speed=new_state[2])
        elif self.adv_policy == "fvdm":
            for i in range(self.num_agents):
                self.adv_policy_model[i].run()
        elif self.adv_policy == "RL" or self.adv_policy == "model":
            for i in range(self.num_agents):
                new_state = self.motion_model(self.get_state(i + 1), action_adv[2 * i: 2 * (i + 1)])
                traci.vehicle.moveToXY(vehID="car" + str(i + 1),
                                       x=new_state[0],
                                       y=new_state[1],
                                       angle=-new_state[3] * 180 / np.pi + 90,
                                       lane=0, edgeID=0)
                traci.vehicle.setSpeed(vehID="car" + str(i + 1), speed=new_state[2])
        if self.adv_policy == "genedata" or self.adv_policy == "realdata":
            for i in range(self.num_agents):
                new_state = self.adv_policy_model[i][self.timestep]
                traci.vehicle.moveToXY(vehID="car" + str(i + 1),
                                       x=new_state[0],
                                       y=new_state[1],
                                       angle=-new_state[3] * 180 / np.pi + 90,
                                       lane=0, edgeID=0)
                traci.vehicle.setSpeed(vehID="car" + str(i + 1), speed=new_state[2])
        # else:
        #     ...
        traci.simulationStep()
        next_state = []
        for i in range(self.num_agents + 1):
            (x, y) = traci.vehicle.getPosition('car' + str(i))
            speed = traci.vehicle.getSpeed('car' + str(i))
            yaw = (90 - traci.vehicle.getAngle('car' + str(i))) * math.pi / 180
            new_state = [x, y, speed, yaw]
            next_state.extend(new_state)
            self.set_state(new_state, i)
            delta_speed = self.states[self.timestep + 1][i * 4 + 2] - self.states[self.timestep][i * 4 + 2]
            delta_yaw = self.states[self.timestep + 1][i * 4 + 3] - self.states[self.timestep][i * 4 + 3]
            action_speed = (delta_speed - self.low_cut[0]) * 2 / (self.up_cut[0] - self.low_cut[0]) - 1
            action_yaw = (delta_yaw - self.low_cut[1]) * 2 / (self.up_cut[1] - self.low_cut[1]) - 1
            car_action = [action_speed, action_yaw]
            if new_state[0] >= 240:  # 240为路径长度
                self.ArrivedVehs.append(0)
            self.actions[self.timestep][i * 2: (i + 2) * 2] = car_action

        if self.gui:
            time.sleep(0.04)

        self.timestep += 1
        self.collision_test()
        reward = self.compute_cost()

        done = True
        info = [[], []]
        # dis_av_crash = traci.vehicle.getDistance('car0')
        # distance_BV = []
        # for i in range(self.num_agents):
        #     distance_BV.append(traci.vehicle.getDistance('car' + str(i + 1)))
        # dis_bv_crash = min(distance_BV)

        if 0 in self.ArrivedVehs:
            info[0] = "AV arrived!"
        elif 0 in self.CollidingVehs:
            info[0] = "AV crashed!"
            # dis_bv_crash = 240
        elif len(self.CollidingVehs) != 0:
            info[0] = "BV crashed!"
            # dis_av_crash = 240
        else:
            if self.timestep == self.max_step:
                done = True
                info[0] = "AV arrived!"
            else:
                done = False

        # info[1] = [ego_col_cost, adv_col_cost, adv_road_cost, dis_av_crash, dis_bv_crash, col_cost, speed_cost]
        info[1] = []

        return next_state, reward, done, info

    def motion_model(self, state, action):
        # 读取位置
        pos_curr = [state[0], state[1]]
        speed_curr = state[2]
        yaw_curr = state[3]
        delta_speed = (action[0] + 1) * (self.up_cut[0] - self.low_cut[0]) / 2 + self.low_cut[0]
        delta_yaw = (action[1] + 1) * (self.up_cut[1] - self.low_cut[1]) / 2 + self.low_cut[1]
        speed = speed_curr + delta_speed
        yaw = yaw_curr + delta_yaw
        speed = 0 if speed <= 0 \
            else self.max_speed if speed >= self.max_speed \
            else speed
        yaw = -math.pi / 3 if yaw <= -math.pi / 3 \
            else math.pi / 3 if yaw >= math.pi / 3 \
            else yaw
        state = [pos_curr[0] + speed * np.cos(yaw) * self.dt,
                 pos_curr[1] + speed * np.sin(yaw) * self.dt,
                 speed, yaw]
        return state

    def compute_cost(self):
        ego_state = self.get_state(0)
        adv_state = []
        for i in range(1, 1 + self.num_agents):
            adv_state.append(self.get_state(i))

        # ego
        if self.r_ego == "r1":
            col_cost_ego = -20 if 0 in self.CollidingVehs else 0
            speed_cost_ego = ego_state[2] / self.max_speed - 1 / 2
            yaw_cost_ego = - abs(ego_state[3]) / (math.pi / 3) * 5 * 0
            cost_ego = col_cost_ego + speed_cost_ego + yaw_cost_ego
        else:
            cost_ego = np.nan

        # adv
        cost_ego_adv, cost_adv_adv, cost_adv_road = 0, 0, 0
        # r = "r1"
        if self.r_adv == "r1":
            if 0 in self.CollidingVehs:
                cost_adv = 100
            elif len(self.CollidingVehs) != 0:
                cost_adv = -100
            else:
                cost_adv = 0

        # r2
        elif self.r_adv[0:2] == "r2":
            bv_bv_thresh = 1.5
            bv_road_thresh = float("inf")
            Rb = [100, -100]
            a, b, c = list(map(float, self.r_adv[3:].split('-')))
            # a, b, c = 1, 1, 0

            ego_col_cost_record, adv_col_cost_record, adv_road_cost_record = float('inf'), float('inf'), float('inf')
            for i in range(self.num_agents):
                car_ego = [ego_state[0], ego_state[1],
                           self.car_info[0][2][0], self.car_info[0][2][1], ego_state[3]]
                car_adv = [adv_state[i][0], adv_state[i][1],
                           self.car_info[i + 1][2][0], self.car_info[i + 1][2][1], adv_state[i][3]]
                dis_ego_adv = dist_between_cars(car_ego, car_adv)
                # dis_ego_adv = math.sqrt((ego_state[0] - adv_state[i][0]) ** 2 +
                #                         (ego_state[1] - adv_state[i][1]) ** 2)
                if dis_ego_adv < ego_col_cost_record:
                    ego_col_cost_record = dis_ego_adv
            cost_ego_adv = ego_col_cost_record

            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    car_adv_j = [adv_state[j][0], adv_state[j][1],
                                 self.car_info[j + 1][2][0], self.car_info[j + 1][2][1], adv_state[j][3]]
                    car_adv_i = [adv_state[i][0], adv_state[i][1],
                                 self.car_info[i + 1][2][0], self.car_info[i + 1][2][1], adv_state[i][3]]
                    dis_adv_adv = dist_between_cars(car_adv_i, car_adv_j)
                    # dis_adv_adv = np.sqrt((adv_state[i][0] - adv_state[j][0]) ** 2 +
                    #                       (adv_state[i][1] - adv_state[j][1]) ** 2)
                    if dis_adv_adv < adv_col_cost_record:
                        adv_col_cost_record = dis_adv_adv
            cost_adv_adv = min(adv_col_cost_record, bv_bv_thresh)

            road_up, road_low = 12, 0
            car_width = 1.8
            for i in range(self.num_agents):
                y = adv_state[i][1]
                dis_adv_road = min(road_up - (y + car_width / 2), (y - car_width / 2) - road_low)
                if dis_adv_road < adv_road_cost_record:
                    adv_road_cost_record = dis_adv_road
            cost_adv_road = min(adv_road_cost_record, bv_road_thresh)

            cost_adv = - a * cost_ego_adv + b * cost_adv_adv + c * cost_adv_road
            if 0 in self.CollidingVehs:
                cost_adv += Rb[0]
            elif len(self.CollidingVehs) != 0:
                cost_adv += Rb[1]

        # r3
        elif self.r_adv == "r3":
            ego_col_cost_record = float('inf')
            for i in range(self.num_agents):
                car_ego = [ego_state[0], ego_state[1],
                           self.car_info[0][2][0], self.car_info[0][2][1], ego_state[3]]
                car_adv = [adv_state[i][0], adv_state[i][1],
                           self.car_info[i + 1][2][0], self.car_info[i + 1][2][1], adv_state[i][3]]
                dis_ego_adv = dist_between_cars(car_ego, car_adv)
                # dis_ego_adv = math.sqrt((ego_state[0] - adv_state[i][0]) ** 2 +
                #                         (ego_state[1] - adv_state[i][1]) ** 2)
                if dis_ego_adv < ego_col_cost_record:
                    ego_col_cost_record = dis_ego_adv
            cost_ego_adv = ego_col_cost_record
            cost_adv = -cost_ego_adv
            if 0 in self.CollidingVehs:
                cost_adv += 100
            elif len(self.CollidingVehs) != 0:
                cost_adv += -100
        else:
            cost_adv = np.nan

        return [cost_ego, cost_adv]  # , cost_ego_adv, cost_adv_adv, cost_adv_road, col_cost_ego, speed_cost_ego

    def collision_test(self):
        """
        Info:
            直线道路上的车辆碰撞检测、车辆驶出路面检测;
            在原版本上, 将car_state更改为H2O中的形式.

        Args:
            car_info (list): 所有车辆的信息, [[ID(int), lane(str), [length(float), width(float)]], ...]
            car_state (list): 所有车辆的位置, [[x(float), y(float), speed(float), yaw(float)], ...]
            road_info (dict): 道路信息, {lane(str): [minLaneMarking(float), maxLaneMarking(float)], ...}

        Return:
            ...
        """
        matrix_car_list = []  # 记录车辆四个顶点的位置
        car_info_list = []  # 记录还在世的车辆信息
        col_list = []
        # 车辆驶出路面检测
        # 可判断车辆是否压两侧车道线, 但如何应对车辆完全在道路外的情况
        for i in range(len(self.car_info)):
            [ID, lane, [length, width]] = self.car_info[i]
            if ID in self.ArrivedVehs:
                continue
            [x, y, speed, yaw] = self.states[self.timestep][i * 4: (i + 1) * 4]
            [minLaneMarking, maxLaneMarking] = self.road_info[lane]
            sin_yaw = math.sin(yaw)
            cos_yaw = math.cos(yaw)
            matrix_car = [[x - width / 2 * sin_yaw - length * cos_yaw, y + width / 2 * cos_yaw - length * sin_yaw],
                          [x - width / 2 * sin_yaw, y + width / 2 * cos_yaw],
                          [x + width / 2 * sin_yaw, y - width / 2 * cos_yaw],
                          [x + width / 2 * sin_yaw - length * cos_yaw, y - width / 2 * cos_yaw - length * sin_yaw]]
            matrix_car = np.array(matrix_car)
            matrix_car_list.append(matrix_car)
            car_info_list.append(self.car_info[i])
            y_max = np.max(matrix_car[:, 1])
            y_min = np.min(matrix_car[:, 1])
            if y_min < minLaneMarking < y_max or y_min < maxLaneMarking < y_max:
                col_list.append(ID)

        # 车辆碰撞检测
        # 通过叉乘判断是否线段相交，进而判断是否碰撞
        # 参考: https://blog.csdn.net/m0_37660632/article/details/123925503
        # 求向量ab和向量cd的叉乘
        def xmult(a, b, c, d):
            vectorAx = b[0] - a[0]
            vectorAy = b[1] - a[1]
            vectorBx = d[0] - c[0]
            vectorBy = d[1] - c[1]
            return (vectorAx * vectorBy - vectorAy * vectorBx)

        while len(car_info_list) != 0:
            ID_i = car_info_list.pop(0)[0]
            matrix_car_i = matrix_car_list.pop(0)
            j = 0
            while j < len(car_info_list):
                ID_j = car_info_list[j][0]
                matrix_car_j = matrix_car_list[j]
                collision = False
                for p in range(-1, 3):
                    c, d = matrix_car_i[p], matrix_car_i[p + 1]
                    for q in range(-1, 3):
                        a, b = matrix_car_j[q], matrix_car_j[q + 1]
                        xmult1 = xmult(c, d, c, a)
                        xmult2 = xmult(c, d, c, b)
                        xmult3 = xmult(a, b, a, c)
                        xmult4 = xmult(a, b, a, d)
                        if xmult1 * xmult2 < 0 and xmult3 * xmult4 < 0:
                            collision = True
                            break
                    if collision:
                        break
                if collision:
                    if ID_i not in col_list:
                        col_list.append(ID_i)
                    if ID_j not in col_list and ID_j not in self.CollidingVehs:
                        col_list.append(ID_j)
                j += 1
        self.CollidingVehs += col_list

    def record(self, filepath=None):
        if filepath is None:
            filepath = 'output/newoutput_test/record-' + datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f") + '.csv'
        record = []
        # for t in range(len(self.states)):
        for t in range(self.timestep + 1):
            state_t = self.states[t]
            action_t = self.actions[t]
            for id in range(self.num_agents + 1):
                if t > 0 and state_t[id * 4: (id + 1) * 4] == self.states[t - 1][id * 4: (id + 1) * 4]:
                    record_t_id = [t, id, -1]
                elif id == 0:
                    record_t_id = [t, id, 0]
                else:
                    record_t_id = [t, id, 1]
                record_t_id.extend(state_t[id * 4: (id + 1) * 4])
                record_t_id.extend(action_t[id * 2: (id + 1) * 2])
                record.append(record_t_id)
        # 2023.4.20: 之前储存的数据舍弃了实际上发生碰撞的最后一个时间步的状态，可根据action计算
        # 更改后，最后一个状态对应的动作为0
        np.savetxt(filepath, record)
        return filepath

