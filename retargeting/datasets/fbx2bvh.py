"""
This code comes from https://github.com/rubenvillegas/cvpr2018nkn/blob/master/datasets/fbx2bvh.py
"""
import bpy
import numpy as np

from os import listdir

data_path = './Mixamo/'

directories = sorted([f for f in listdir(data_path) if not f.startswith(".")])
for d in directories:
    files = sorted([f for f in listdir(data_path + d) if f.endswith(".fbx")])

    for f in files:
        sourcepath = data_path + d + "/" + f
        dumppath = data_path+d + "/" + f.split(".fbx")[0] + ".bvh"

        bpy.ops.import_scene.fbx(filepath=sourcepath)

        frame_start = 9999
        frame_end = -9999
        action = bpy.data.actions[-1]
        if action.frame_range[1] > frame_end:
            frame_end = action.frame_range[1]
        if action.frame_range[0] < frame_start:
            frame_start = action.frame_range[0]

        frame_end = np.max([60, frame_end])
        bpy.ops.export_anim.bvh(filepath=dumppath,
                                frame_start=frame_start,
                                frame_end=frame_end, root_transform_only=True)
        bpy.data.actions.remove(bpy.data.actions[-1])

        print(data_path + d + "/" + f + " processed.")
