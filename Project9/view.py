# Ethan Seal
# 3/6/19
# CS251 Project 3

import numpy as np
import math


class View:
    def __init__(self,
                 vrp=np.matrix([0.5, 0.5, 1.]),
                 vpn=np.matrix([0., 0., -1.]),
                 vup=np.matrix([0., 1., 0.]),
                 u=np.matrix([-1., 0., 0.]),
                 extent=np.matrix([1., 1., 1.]),
                 screen=[400., 400.],
                 offset=np.matrix([20., 20.])):

        self.vtm = self.reset()

    # Extension 4 implements reset
    def reset(self):
        self.vrp = np.matrix([0.5, 0.5, 1.])
        self.vpn = np.matrix([0., 0., -1.])
        self.vup = np.matrix([0., 1., 0.])
        self.u = np.matrix([-1., 0., 0.])
        self.extent = np.matrix([1., 1., 1.])
        self.screen = [400., 400.]
        self.offset = np.matrix([20., 20.])

    def build(self):
        vtm = np.identity(4, float)
        # first transposition
        t1 = np.matrix([[1, 0, 0, -self.vrp[0, 0]],
                        [0, 1, 0, -self.vrp[0, 1]],
                        [0, 0, 1, -self.vrp[0, 2]],
                        [0, 0, 0, 1]])
        vtm = t1 * vtm

        # make temporary variables
        tu = np.cross(self.vup, self.vpn)
        tvup = np.cross(self.vpn, tu)
        tvpn = self.vpn

        # normalize the vectors
        self.normalize(tu)
        self.normalize(tvup)
        self.normalize(tvpn)

        # copy the vectors back over
        self.u = tu
        self.vup = tvup
        self.vpn = tvpn

        # axis alignment
        r1 = np.matrix([[tu[0, 0], tu[0, 1], tu[0, 2], 0.0],
                        [tvup[0, 0], tvup[0, 1], tvup[0, 2], 0.0],
                        [tvpn[0, 0], tvpn[0, 1], tvpn[0, 2], 0.0],
                        [0.0, 0.0, 0.0, 1.0]])

        vtm = r1 * vtm

        # second transposition
        vtm = np.matrix([[1, 0, 0, 0.5 * self.extent[0, 0]],
                         [0, 1, 0, 0.5 * self.extent[0, 1]],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]) * vtm

        # scale to
        vtm = np.matrix([[-self.screen[0] / self.extent[0, 0], 0, 0, 0],
                         [0, -self.screen[1] / self.extent[0, 1], 0, 0],
                         [0, 0, 1.0 / self.extent[0, 2], 0],
                         [0, 0, 0, 1]]) * vtm

        vtm = np.matrix([[1, 0, 0, self.screen[0] + self.offset[0, 0]],
                         [0, 1, 0, self.screen[1] + self.offset[0, 1]],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]) * vtm
        return vtm

    def normalize(self, vector):
        sum = 0
        # sum up the squares
        for i in range(vector.shape[1]):
            sum += vector[0, i] ** 2
        length = float(sum) ** (0.5)

        # divide by length
        for i in range(vector.shape[1]):
            vector.itemset((0, i), float(vector[0, i]) / length)
        return vector

    def rotateVRC(self, vup_angle, u_angle):
        # bring the point to the origin
        t1 = np.matrix([[1.0, 0.0, 0.0, -(self.vrp[0, 0] + self.vpn[0, 0] * 0.5 * self.extent[0, 2])],
                        [0.0, 1.0, 0.0, -(self.vrp[0, 1] + self.vpn[0, 1] * 0.5 * self.extent[0, 2])],
                        [0.0, 0.0, 1.0, -(self.vrp[0, 2] + self.vpn[0, 2] * 0.5 * self.extent[0, 2])],
                        [0.0, 0.0, 0.0, 1.0]])

        # print("t1",t1)
        # axis alignment matrix
        rxyz = np.matrix([[self.u[0, 0], self.u[0, 1], self.u[0, 2], 0.0],
                          [self.vup[0, 0], self.vup[0, 1], self.vup[0, 2], 0.0],
                          [self.vpn[0, 0], self.vpn[0, 1], self.vpn[0, 2], 0.0],
                          [0.0, 0.0, 0.0, 1.0]])
        # print("rxyz",rxyz)
        # rotation matrix for the y axis
        r1 = np.matrix([[math.cos(vup_angle), 0.0, math.sin(vup_angle), 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [-math.sin(vup_angle), 0.0, math.cos(vup_angle), 0.0],
                        [0.0, 0.0, 0.0, 1.0]])
        # print("r1",r1)
        # rotation matrix for the x axis
        r2 = np.matrix([[1.0, 0.0, 0.0, 0.0],
                        [0.0, math.cos(u_angle), -math.sin(u_angle), 0.0],
                        [0.0, math.sin(u_angle), math.cos(u_angle), 0.0],
                        [0.0, 0.0, 0.0, 1.0]])
        # print("r2",r2)
        # undo the effects of t1
        t2 = np.matrix([[1.0, 0.0, 0.0, self.vrp[0, 0] + self.vpn[0, 0] * 0.5 * self.extent[0, 2]],
                        [0.0, 1.0, 0.0, self.vrp[0, 1] + self.vpn[0, 1] * 0.5 * self.extent[0, 2]],
                        [0.0, 0.0, 1.0, self.vrp[0, 2] + self.vpn[0, 2] * 0.5 * self.extent[0, 2]],
                        [0.0, 0.0, 0.0, 1.0]])
        # print("t2",t2)
        # matrix to be affected by these other matrices
        tvrc = np.matrix([[self.vrp[0, 0], self.vrp[0, 1], self.vrp[0, 2], 1],
                          [self.u[0, 0], self.u[0, 1], self.u[0, 2], 0],
                          [self.vup[0, 0], self.vup[0, 1], self.vup[0, 2], 0],
                          [self.vpn[0, 0], self.vpn[0, 1], self.vpn[0, 2], 0]])
        # print("vrc",vrc)
        # our one-liner rotation
        tvrc = (t2 * rxyz.T * r2 * r1 * rxyz * t1 * tvrc.T).T

        # put the rotated, normalized values back into their vectors
        vecs = [self.vrp, self.u, self.vup, self.vpn]
        # print("vecs",vecs)
        for i in range(len(vecs)):
            for j in range(3):
                vecs[i][0, j] = tvrc[i, j]
            if i != 0:
                self.normalize(vecs[i])

    # print(tvrc)

    def clone(self):
        clone = View()
        clone.vrp = self.vrp.copy()
        clone.vpn = self.vpn.copy()
        clone.vup = self.vup.copy()
        clone.u = self.u.copy()
        clone.extent = self.extent.copy()
        clone.screen = self.screen.copy()
        clone.offset = self.offset.copy()
        return clone

    def getScreen(self):
        return self.screen

    def getExtent(self):
        return self.extent

    def setExtent(self, extent):
        self.extent = extent

    def getVRP(self):
        return self.vrp

    def setVRP(self, vrp):
        self.vrp = vrp

    def getVUP(self):
        return self.vup

    def getU(self):
        return self.u


def main():
    view = View()
    print(view.build())
    clone = view.clone()
    print(clone.build())
    view.rotateVRC(10, 1)


if __name__ == '__main__':
    main()