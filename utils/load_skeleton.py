import yaml
import numpy as np
import os
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASEPATH)
import BVH as BVH


class Skel:
    def __init__(self, filename=os.path.join(BASEPATH, "..", "style_transfer", "global_info", "skeleton_CMU.yml")):
        f = open(filename, "r")
        skel = yaml.load(f, Loader=yaml.Loader)
        self.bvh_name = os.path.join(os.path.dirname(filename), skel['BVH'])
        self.rest_bvh = BVH.load(self.bvh_name)
        self.offset = np.array(skel['offsets'])
        self.topology = np.array(skel['parents'])
        self.chosen_joints = np.array(skel['chosen_joints'])
        self.chosen_parents = np.array(skel['chosen_parents'])
        self.fid_l, self.fid_r = skel['left_foot'], skel['right_foot']
        self.hips, self.sdrs = skel['hips'], skel['shoulders']
        self.head = skel['head']
        self.visualization = skel['visualization']


if __name__ == '__main__':
    skel = Skel()
    print(skel.topology)
    print(skel.offset)
    print(skel.rest_bvh[0].offsets)
    print(skel.chosen_joints)
    print(skel.chosen_parents)

