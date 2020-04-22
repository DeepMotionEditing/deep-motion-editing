import sys
sys.path.append('./')
import bpy

from options import Options
from load_bvh import load_bvh
from scene import make_scene, add_material_for_character, add_rendering_parameters

if __name__ == '__main__':
    args = Options(sys.argv).parse()

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    character = load_bvh(args.bvh_path)
    scene = make_scene()
    add_material_for_character(character)
    bpy.ops.object.select_all(action='DESELECT')

    add_rendering_parameters(bpy.context.scene, args, scene[1])

    if args.render:
        bpy.ops.render.render(animation=True, use_viewport=True)
