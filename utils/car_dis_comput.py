# https://zhuanlan.zhihu.com/p/569701615
# -*- coding:utf-8 -*-

import math
import time


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def rotate(self, theta):
        x = self.x * math.cos(theta) + self.y * math.sin(theta)
        y = -self.x * math.sin(theta) + self.y * math.cos(theta)
        return Point(x, y)

    def translate(self, x, y):
        self.x += x
        self.y += y
        return Point(self.x, self.y)


def dist_between_points(p1: Point, p2: Point) -> float:
    return math.sqrt(abs(p1.x - p2.x) ** 2 + abs(p1.y - p2.y) ** 2)


class Line:
    def __init__(self, p1: Point, p2: Point):
        self.P1 = p1
        self.P2 = p2

        self.A, self.B, self.C = self._solve()

    def _solve(self):
        if math.isclose(self.P1.y, self.P2.y):
            A = 0
            B = 1
            C = -self.P1.y
        elif math.isclose(self.P1.x, self.P2.x):
            A = 1
            B = 0
            C = -self.P1.x
        else:
            A = (self.P2.y - self.P1.y) / (self.P1.x - self.P2.x)
            B = 1
            C = -A * self.P1.x - self.P1.y
        return A, B, C

    def dist_to_point(self, p: Point) -> float:
        dist = abs(self.A * p.x + self.B * p.y + self.C) / math.sqrt(self.A ** 2 + self.B ** 2)

        X = (self.B ** 2 * p.x - self.A * self.B * p.y - self.A * self.C) / (self.A ** 2 + self.B ** 2)
        Y = (self.A ** 2 * p.y - self.A * self.B * p.x - self.B * self.C) / (self.A ** 2 + self.B ** 2)
        P = Point(X, Y)

        if P in self:
            return dist
        else:
            return float('inf')

    def __contains__(self, item: Point) -> bool:
        return min(self.P1.x, self.P2.x) <= item.x <= max(self.P1.x, self.P2.x) and min(
            self.P1.y, self.P2.y) <= item.y <= max(self.P1.y, self.P2.y)


def cross(l1: Line, l2: Line) -> bool:
    delta = l1.A * l2.B - l2.A * l1.B
    if math.isclose(delta, 0):
        if math.isclose(l1.C, l2.C):
            if l1.P1 in l2 or l1.P2 in l2:
                return True
    else:
        X = (l2.C * l1.B - l1.C * l2.B) / delta
        Y = (l1.C * l2.A - l2.C * l1.A) / delta
        P = Point(X, Y)
        if P in l1 and P in l2:
            return True
    return False


class Rect:
    def __init__(self, center_x: float, head_y: float, length: float, width: float, yaw: float):
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)

        self.A = Point(center_x - width / 2 * sin_yaw - length * cos_yaw, head_y + width / 2 * cos_yaw - length * sin_yaw)
        self.B = Point(center_x - width / 2 * sin_yaw, head_y + width / 2 * cos_yaw)
        self.C = Point(center_x + width / 2 * sin_yaw, head_y - width / 2 * cos_yaw)
        self.D = Point(center_x + width / 2 * sin_yaw - length * cos_yaw, head_y - width / 2 * cos_yaw - length * sin_yaw)
        self.points = [self.A, self.B, self.C, self.D]
        self.lines = [Line(self.A, self.B), Line(self.B, self.C), Line(self.C, self.D), Line(self.D, self.A)]


def dist_between_rectangles(ego: Rect, obj: Rect) -> float:
    for L1 in ego.lines:
        for L2 in obj.lines:
            if cross(L1, L2):
                return 0
    lst = []
    for P1 in ego.points:
        for P2 in obj.points:
            lst.append(dist_between_points(P1, P2))
    for L in ego.lines:
        for P in obj.points:
            lst.append(L.dist_to_point(P))
    for L in obj.lines:
        for P in ego.points:
            lst.append(L.dist_to_point(P))
    return min(lst)


def dist_between_cars(car1, car2):
    R1 = Rect(car1[0], car1[1], car1[2], car1[3], car1[4])
    R2 = Rect(car2[0], car2[1], car2[2], car2[3], car2[4])
    return dist_between_rectangles(R1, R2)


if __name__ == "__main__":
    car1 = [2, 1, 2, 2, 0]
    car2 = [4, 3, 2, 2, 0]
    t = time.time()
    for i in range(1000):
        dist_between_cars(car1, car2)
    print(time.time() - t)
