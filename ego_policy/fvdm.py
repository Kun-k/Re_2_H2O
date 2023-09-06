import time
import traci
import numpy as np
import random
from sumolib import checkBinary
import optparse
import sys


class fvdm_model(object):
    def __init__(self, carID, dt=0.04):
        self.carID = carID
        self.buslen = 10
        self.carlen = 5
        self.minGap = 1.5
        self.follow_distance = 50
        self.lanechange_time = 5
        self.p = 0
        self.dt = dt
        self.type = traci.vehicle.getTypeID(self.carID)
        self.vmax = traci.vehicle.getMaxSpeed(self.carID)
        self.maxacc = traci.vehicle.getAccel(self.carID)
        self.maxdec = traci.vehicle.getDecel(self.carID)
        self.length = traci.vehicle.getLength(self.carID)

        self.speed = traci.vehicle.getSpeed(self.carID)
        self.lane = traci.vehicle.getLaneID(self.carID)
        self.lanePosition = traci.vehicle.getLanePosition(self.carID)


    def frontCar(self):
        m = float('inf')
        vehicle_frontCarID = ""
        for carID in traci.vehicle.getIDList():
            lanePosition = traci.vehicle.getLanePosition(carID)
            if traci.vehicle.getLaneID(carID) == self.lane \
                    and self.lanePosition < lanePosition \
                    and lanePosition - self.lanePosition < self.follow_distance:
                if lanePosition - self.lanePosition < m:
                    m = lanePosition - self.lanePosition
                    vehicle_frontCarID = carID
        return vehicle_frontCarID

    def nearFrontCar(self):
        m = float('inf')
        vehicle_nearFrontCarID_0 = ""
        vehicle_nearFrontCarID_1 = ""
        vehicle_nearFrontCarID_2 = ""
        for carID in traci.vehicle.getIDList():
            lanePosition = traci.vehicle.getLanePosition(carID)
            if (self.lane == "lane0" or self.lane == "lane2") and traci.vehicle.getLaneID(carID) == "lane1" \
                    and self.lanePosition < lanePosition \
                    and lanePosition - self.lanePosition < self.follow_distance:
                if lanePosition - self.lanePosition < m:
                    m = lanePosition - self.lanePosition
                    vehicle_nearFrontCarID_1 = carID
            elif self.lane == "lane1" and self.lanePosition < lanePosition \
                    and lanePosition - self.lanePosition < self.follow_distance:
                if traci.vehicle.getLaneID(carID) == "lane0":
                    if lanePosition - self.lanePosition < m:
                        m = lanePosition - self.lanePosition
                        vehicle_nearFrontCarID_0 = carID
                if traci.vehicle.getLaneID(carID) == "lane2":
                    if lanePosition - self.lanePosition < m:
                        m = lanePosition - self.lanePosition
                        vehicle_nearFrontCarID_2 = carID
        return vehicle_nearFrontCarID_0, vehicle_nearFrontCarID_1, vehicle_nearFrontCarID_2

    def nearRearCar(self):
        m = float('inf')
        vehicle_nearRearCarID_0 = ""
        vehicle_nearRearCarID_1 = ""
        vehicle_nearRearCarID_2 = ""
        for carID in traci.vehicle.getIDList():
            lanePosition = traci.vehicle.getLanePosition(carID)
            if (self.lane == "lane0" or self.lane == "lane2") and traci.vehicle.getLaneID(carID) == "lane1" \
                    and lanePosition < self.lanePosition:
                if lanePosition - self.lanePosition < m:
                    m = lanePosition - self.lanePosition
                    vehicle_nearRearCarID_1 = carID
            elif self.lane == "lane1" and lanePosition < self.lanePosition:
                if traci.vehicle.getLaneID(carID) == "lane0":
                    if lanePosition - self.lanePosition < m:
                        m = lanePosition - self.lanePosition
                        vehicle_nearRearCarID_0 = carID
                if traci.vehicle.getLaneID(carID) == "lane2":
                    if lanePosition - self.lanePosition < m:
                        m = lanePosition - self.lanePosition
                        vehicle_nearRearCarID_2 = carID
        return vehicle_nearRearCarID_0, vehicle_nearRearCarID_1, vehicle_nearRearCarID_2

    def speed_generate(self):
        v_next = self.speed
        Pslow, Ps = 0, 0
        frontCar = self.frontCar()
        if frontCar != "":
            frontCarSpeed = traci.vehicle.getSpeed(frontCar)
            frontCarDistance = traci.vehicle.getLanePosition(frontCar)
            minAccSpeed = min(self.speed + self.maxacc, self.vmax)
            if self.speed == 0 and random.uniform(0, 1) < Pslow:
                v_next = 0
            elif frontCarSpeed + frontCarDistance - (
                    minAccSpeed + self.speed) / 2 - self.lanePosition > self.minGap + self.length:
                v_next = minAccSpeed
                if random.uniform(0, 1) < Ps:
                    v_next = max(v_next - self.maxdec, 0)
            elif frontCarSpeed + frontCarDistance - (
                    minAccSpeed + self.speed) / 2 - self.lanePosition == self.minGap + self.length:
                if random.uniform(0, 1) < Ps:
                    v_next = max(v_next - self.maxdec, 0)
            else:
                v_next = max(self.speed - self.maxdec, 0)
        else:
            v_next = min(self.speed + self.maxacc, self.vmax)
        return v_next

    def changeLane(self):
        ifChangeLane = False
        leftChangeLane = False
        rightChangeLane = False
        Prc, Plc = 0.6, 0.9
        nearFrontCar_0 = self.nearFrontCar()[0]
        nearFrontCar_1 = self.nearFrontCar()[1]
        nearFrontCar_2 = self.nearFrontCar()[2]
        nearRearCar_0 = self.nearRearCar()[0]
        nearRearCar_1 = self.nearRearCar()[1]
        nearRearCar_2 = self.nearRearCar()[2]
        frontCar = self.frontCar()
        minAccSpeed = min(self.speed + self.maxacc, self.vmax)
        if frontCar == "" or traci.vehicle.getLanePosition(frontCar) - self.lanePosition < self.minGap + self.length:
            ...
        elif self.lane == "lane2":
            if nearRearCar_1 != "" \
                    and self.lanePosition - traci.vehicle.getLanePosition(nearRearCar_1) < self.minGap + self.length:
                ...
            elif nearFrontCar_1 != "" \
                    and (traci.vehicle.getLanePosition(nearFrontCar_1) - self.lanePosition < self.minGap + self.length
                         or traci.vehicle.getLanePosition(nearFrontCar_1) - self.lanePosition <
                         traci.vehicle.getLanePosition(frontCar) - self.lanePosition):
                ...
            elif random.uniform(0, 1) <= Prc:
                ifChangeLane = True
        elif self.lane == "lane0":
            if nearRearCar_1 != "" \
                    and self.lanePosition - traci.vehicle.getLanePosition(nearRearCar_1) < self.minGap + self.length:
                ...
            elif nearFrontCar_1 != "" \
                    and (traci.vehicle.getLanePosition(nearFrontCar_1) - self.lanePosition < self.minGap + self.length
                         or traci.vehicle.getLanePosition(nearFrontCar_1) - self.lanePosition <
                         traci.vehicle.getLanePosition(frontCar) - self.lanePosition):
                ...
            elif random.uniform(0, 1) <= Plc:
                ifChangeLane = True
        elif self.lane == "lane1":
            if nearRearCar_2 != "" \
                    and self.lanePosition - traci.vehicle.getLanePosition(nearRearCar_2) < self.minGap + self.length:
                ...
            elif nearFrontCar_2 != "" \
                    and (traci.vehicle.getLanePosition(nearFrontCar_2) - self.lanePosition < self.minGap + self.length
                         or traci.vehicle.getLanePosition(nearFrontCar_2) - self.lanePosition <
                         traci.vehicle.getLanePosition(frontCar) - self.lanePosition):
                ...
            elif random.uniform(0, 1) <= Plc:
                ifChangeLane = True
                leftChangeLane = True
            if ifChangeLane:
                ...
            elif nearRearCar_0 != "" \
                    and self.lanePosition - traci.vehicle.getLanePosition(
                nearRearCar_0) < self.minGap + self.length:
                ...
            elif nearFrontCar_0 != "" \
                    and (
                    traci.vehicle.getLanePosition(nearFrontCar_0) - self.lanePosition < self.minGap + self.length
                    or traci.vehicle.getLanePosition(nearFrontCar_0) - self.lanePosition <
                    traci.vehicle.getLanePosition(frontCar) - self.lanePosition):
                ...
            elif random.uniform(0, 1) <= Prc:
                ifChangeLane = True
                rightChangeLane = True
        return [ifChangeLane, leftChangeLane, rightChangeLane]

    def run(self):
        self.speed = traci.vehicle.getSpeed(self.carID)
        self.lane = traci.vehicle.getLaneID(self.carID)
        self.lanePosition = traci.vehicle.getLanePosition(self.carID)
        changeLane = self.changeLane()
        if self.lane == "lane0" or self.lane == "lane2":
            if changeLane[0]:
                traci.vehicle.changeLane(self.carID, 1, self.lanechange_time)
            else:
                speed_next = self.speed_generate()
                traci.vehicle.setSpeed(self.carID, speed_next)
                traci.vehicle.changeLane(self.carID, traci.vehicle.getLaneIndex(self.carID), 0)
        elif self.lane == "lane1":
            if changeLane[0]:
                if changeLane[1]:
                    traci.vehicle.changeLane(self.carID, 2, self.lanechange_time)
                elif changeLane[2]:
                    traci.vehicle.changeLane(self.carID, 0, self.lanechange_time)
            else:
                speed_next = self.speed_generate()
                traci.vehicle.setSpeed(self.carID, speed_next)
                traci.vehicle.changeLane(self.carID, traci.vehicle.getLaneIndex(self.carID), 0)


if __name__ == "__main__":
    cfg_sumo = 'config\\lane.sumocfg'
    sim_seed = 42
    app = "sumo-gui"
    command = [checkBinary(app), '-c', cfg_sumo]
    command += ['--routing-algorithm', 'dijkstra']
    # command += ['--collision.action', 'remove']
    command += ['--seed', str(sim_seed)]
    command += ['--no-step-log', 'True']
    command += ['--time-to-teleport', '300']
    command += ['--no-warnings', 'True']
    command += ['--duration-log.disable', 'True']
    command += ['--waiting-time-memory', '1000']
    command += ['--eager-insert', 'True']
    command += ['--lanechange.duration', '2']
    command += ['--lateral-resolution', '0.0']
    traci.start(command)

    cur_time = float(traci.simulation.getTime())
    traci.vehicle.add(vehID="veh0", routeID="straight", typeID="AV",
                      depart=cur_time, departLane=1, arrivalLane=0, departPos=0.0, arrivalPos=float('inf'),
                      departSpeed=5)
    traci.vehicle.add(vehID="veh1", routeID="straight", typeID="BV", arrivalLane=2,
                      depart=cur_time, departLane=1, departPos=40.0,
                      departSpeed=5)
    traci.simulationStep()
    car0 = fvdm_model("veh0")
    car1 = fvdm_model("veh1")
    for t in range(500):
        time.sleep(0.04)
        car0.run()
        car1.run()
        traci.simulationStep()
    traci.close()
