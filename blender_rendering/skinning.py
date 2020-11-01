import bpy
import sys
import numpy as np
import argparse
import os


def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def load_fbx(source):
    bpy.ops.import_scene.fbx(filepath=source, use_anim=False)


def load_bvh(source):
    bpy.ops.import_anim.bvh(filepath=source)
    return source.split('/')[-1][:-4]


def set_rest_pose_bvh(filename, source_arm):
    """
    This helps recover the rest pose position from the rest pose of fbx reference file
    """
    dest_filename = filename[:-4] + '_tmp.bvh'
    dest_file = open(dest_filename, 'w')
    rest_loc = source_arm.data.bones[0].head_local
    source_file = open(filename, 'r')
    content = source_file.readlines()

    flag = 0
    for i in range(len(content)):
        if 'ROOT' in content[i]:
            content[i + 2] = '\tOFFSET %.6f %.6f %.6f\n' % (rest_loc[0], rest_loc[1], rest_loc[2])
            flag = 1
            break
    
    if flag == 0:
        raise Exception('Illegal bvh file')

    dest_file.write(''.join(content))
    return dest_filename


def extract_weight(me):
    """
    Extract skinning weight from a given mesh
    """
    verts = me.data.vertices
    vgrps = me.vertex_groups

    weight = np.zeros((len(verts), len(vgrps)))
    mask = np.zeros(weight.shape, dtype=np.int)
    vgrp_label = vgrps.keys()

    for i, vert in enumerate(verts):
        for g in vert.groups:
            j = g.group
            weight[i, j] = g.weight
            mask[i, j] = 1

    return weight, vgrp_label, mask


def clean_vgrps(me):
    vgrps = me.vertex_groups
    for _ in range(len(vgrps)):
        vgrps.remove(vgrps[0])


def load_weight(me, label, weight):
    clean_vgrps(me)
    verts = me.data.vertices
    vgrps = me.vertex_groups

    for name in label:
        vgrps.new(name=name)

    for j in range(weight.shape[1]):
        idx = vgrps.find(label[j])
        if idx == -1:
            #pdb.set_trace()
            continue

        for i in range(weight.shape[0]):
            vgrps[idx].add([i], weight[i, j], 'REPLACE')


def set_modifier(me, arm):
    modifiers = me.modifiers
    for modifier in modifiers:
        if modifier.type == 'ARMATURE':
            modifier.object = arm
            modifier.use_vertex_groups = True
            modifier.use_deform_preserve_volume = True
            return
    
    modifiers.new(name='Armature', type='ARMATURE')
    modifier = modifiers[0]
    modifier.object = arm
    modifier.use_vertex_groups = True
    modifier.use_deform_preserve_volume = True


def adapt_weight(source_weight, source_label, source_arm, dest_arm):
    """
    The targeted armature could be a reduced one, e.g. no fingers. So move the skinning weight of each reduced armature to its nearest ancestor.
    """
    weight = np.zeros((source_weight.shape[0], len(dest_arm.data.bones)))

    # Skinning weight is bond to armature names. For simplicity, a common prefix 
    # is removed in our retargeting output. Here we solve this problem.
    prefix = ''
    ref_name = source_arm.data.bones[0].name
    if ':' in ref_name and ':' not in dest_arm.data.bones[0].name:
        idx = ref_name.index(':')
        prefix = ref_name[:idx + 1]
    dest_name = [prefix + bone.name for bone in dest_arm.data.bones]

    for j, name in enumerate(source_label):
        bone = source_arm.data.bones.find(name)
        bone = source_arm.data.bones[bone]
        while bone.parent is not None and bone.name not in dest_name:
            bone = bone.parent
        idx = dest_name.index(bone.name)
        weight[:, idx] += source_weight[:, j]
    
    return weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fbx_file', type=str, required=True, help='path of skinned model fbx file')
    parser.add_argument('--bvh_file', type=str, required=True, help='path of animation bvh file')

    if "--" not in sys.argv:
        argv = []
    else:
        argv = sys.argv[sys.argv.index("--") + 1:]

    args = parser.parse_args(argv)

    clean_scene()
    load_fbx(args.fbx_file)

    source_arm = bpy.data.objects['Armature']
    meshes = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            meshes.append(obj)
    
    bvh_file = set_rest_pose_bvh(args.bvh_file, source_arm)
    bvh_name = load_bvh(bvh_file)
    dest_arm = bpy.data.objects[bvh_name]
    dest_arm.scale = [0.01, 0.01, 0.01]  # scale the bvh to match the fbx

    for me in meshes:
        weight, label, _ = extract_weight(me)
        weight = adapt_weight(weight, label, source_arm, dest_arm)
        load_weight(me, dest_arm.data.bones.keys(), weight)
        set_modifier(me, dest_arm)

    source_arm.hide_viewport = True
    os.system('rm %s' % bvh_file) # remove temporary file 


if __name__ == "__main__":
    main()
