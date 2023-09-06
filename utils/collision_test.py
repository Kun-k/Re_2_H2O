import math
import numpy as np
from copy import deepcopy


def col_test(car_info, car_state, road_info):

    matrix_car_list = []
    col_car_car_list = []
    col_car_road_list = []

    for i in range(len(car_info)):
        [ID, lane, [length, width]] = car_info[i]
        [x, y, speed, yaw] = car_state[i * 4: (i + 1) * 4]
        [minLaneMarking, maxLaneMarking] = road_info[lane]
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)
        matrix_car = [[x - width / 2 * sin_yaw - length * cos_yaw, y + width / 2 * cos_yaw - length * sin_yaw],
                      [x - width / 2 * sin_yaw, y + width / 2 * cos_yaw],
                      [x + width / 2 * sin_yaw, y - width / 2 * cos_yaw],
                      [x + width / 2 * sin_yaw - length * cos_yaw, y - width / 2 * cos_yaw - length * sin_yaw]]
        matrix_car = np.array(matrix_car)
        matrix_car_list.append(matrix_car)
        y_max = np.max(matrix_car[:, 1])
        y_min = np.min(matrix_car[:, 1])
        if y_min < minLaneMarking < y_max or y_min < maxLaneMarking < y_max:
            col_car_road_list.append(ID)

    def xmult(a, b, c, d):
        vectorAx = b[0] - a[0]
        vectorAy = b[1] - a[1]
        vectorBx = d[0] - c[0]
        vectorBy = d[1] - c[1]
        return (vectorAx * vectorBy - vectorAy * vectorBx)

    car_info_ = deepcopy(car_info)
    while len(car_info_) != 0:
        ID_i = car_info_.pop(0)[0]
        matrix_car_i = matrix_car_list.pop(0)
        j = 0
        while j < len(car_info_):
            matrix_car_j = matrix_car_list[j]
            collision = False
            for p in range(-1, 3):
                c, d = matrix_car_i[p],  matrix_car_i[p + 1]
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
                if ID_i not in col_car_car_list:
                    col_car_car_list.append(ID_i)
                col_car_car_list.append(car_info_.pop(j)[0])
            else:
                j += 1
    return col_car_car_list, col_car_road_list


if __name__ == "__main__":
    car_info = [[0, "lane0", [4, 2]],
                [1, "lane1", [4, 2]],
                [2, "lane0", [4, 2]],
                [3, "lane1", [4, 2]],
                [4, "lane1", [4, 2]]]
    car_state = [20, 1.5, 10, 5 * math.pi / 180,
                 30, 9.5, 10, 0 * math.pi / 180,
                 35, 0.8, 10, -30 * math.pi / 180,
                 10, 8, 10, 0 * math.pi / 180,
                 5.9, 8.5, 10, 7 * math.pi / 180]
    road_info = {"lane0": [0, 4], "lane1": [6, 10]}
    col_car_car_list, col_car_road_list = col_test(car_info, car_state, road_info)
    print(col_car_car_list)
    print(col_car_road_list)
