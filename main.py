import matplotlib.pyplot as plt
from math import pi
import numpy as np


class Wing:
    def __init__(self, max_camber: int, p_camber: int, max_thickness: int):
        self.f_ = max_camber / 100
        self.x_f = p_camber / 10
        self.c_ = max_thickness / 100
        self.cords = self.__naca()
        self.scale_size = 1

    def scale(self, chord: int | float) -> None:
        self.cords = self.cords * chord
        self.scale_size = self.scale_size * chord

    def __naca(self) -> np.ndarray:

        # 1 Шаг

        betta = np.arange(0, pi / 2, 0.01)

        x: np.ndarray = 1 - np.cos(betta)

        # 2 Шаг

        y_f = []
        d_y_f = []

        for x_i in x:
            if (x_i < self.x_f) and (x_i >= 0):
                y_f.append((self.f_ / (self.x_f ** 2)) * (2 * self.x_f * x_i - x_i ** 2))
                d_y_f.append(2 * self.f_ * (self.x_f - x_i) / (self.x_f ** 2))
            elif (self.x_f <= x_i) and (x_i <= 1):
                y_f.append((self.f_ / ((1 - self.x_f) ** 2)) * (1 - 2 * self.x_f + 2 * self.x_f * x_i - x_i ** 2))
                d_y_f.append(2 * self.f_ * (self.x_f - x_i) / ((1 - self.x_f) ** 2))

        y_f = np.array(y_f)
        d_y_f = np.array(d_y_f)

        # 3 Шаг

        a_0 = 0.2969
        a_1 = -0.126
        a_2 = -0.3516
        a_3 = 0.2843
        a_4 = -0.1036

        y_c: np.ndarray = self.c_ * (a_0 * np.sqrt(x) + a_1 * x + a_2 * (x ** 2) + a_3 * (x ** 3) + a_4 * (x ** 4))/0.2

        # 4 Шаг

        theta: np.ndarray = np.arctan(d_y_f)

        x_u: np.ndarray = x - y_c * np.sin(theta)
        y_u: np.ndarray = y_f + y_c * np.cos(theta)

        x_l: np.ndarray = x + y_c * np.sin(theta)
        y_l: np.ndarray = y_f - y_c * np.cos(theta)
        return np.array([x, y_f, x_u, y_u, x_l, y_l])

    def plot(self, x_lim: int | float = 1.0, y_lim: int | float = 1.0) -> None:
        plt.plot(self.cords[0], self.cords[1], c='b')
        plt.plot(self.cords[2], self.cords[3], c='black')
        plt.plot(self.cords[4], self.cords[5], c='black')
        plt.xlim(-0.1, x_lim * self.scale_size + 0.1)
        plt.ylim(-y_lim * self.scale_size, y_lim * self.scale_size)
        plt.grid()
        plt.show()


wing_1 = Wing(2, 4, 12)
wing_1.plot(y_lim=0.3)
