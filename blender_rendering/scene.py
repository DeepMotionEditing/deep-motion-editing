import bpy


def add_floor(size):
    bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = 'floor'

    floor_mat = bpy.data.materials.new(name="floorMaterial")
    floor_mat.use_nodes = True
    bsdf = floor_mat.node_tree.nodes["Principled BSDF"]
    floor_text = floor_mat.node_tree.nodes.new("ShaderNodeTexChecker")
    floor_text.inputs[3].default_value = 150
    floor_mat.node_tree.links.new(bsdf.inputs['Base Color'], floor_text.outputs['Color'])

    floor.data.materials.append(floor_mat)
    return floor


def add_camera(location, rotation):
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=location, rotation=rotation)
    camera = bpy.context.object
    return camera


def add_light(location):
    bpy.ops.object.light_add(type='SUN', location=location)
    sun = bpy.context.object
    return sun


def make_scene(floor_size=1000, camera_position=(37.54, -28.87, 16.34), camera_rotation=(1.30473, 0.0109881, 0.896417),
               light_position=(0, 0, 20)):
    floor = add_floor(floor_size)
    camera = add_camera(camera_position, camera_rotation)
    light = add_light(light_position)
    bpy.ops.object.select_all(action='DESELECT')
    floor.select_set(True)
    camera.select_set(True)
    light.select_set(True)
    bpy.ops.object.move_to_collection(collection_index=0, is_new=True, new_collection_name="Scene")
    bpy.ops.object.select_all(action='DESELECT')
    return [floor, camera, light]


def add_rendering_parameters(scene, args, camera):
    scene.render.resolution_x = args.resX
    scene.render.resolution_y = args.resY
    scene.frame_end = args.frame_end
    scene.camera = camera
    scene.render.filepath = args.save_path

    if args.render_engine == 'cycles':
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'
    elif args.render_engine == 'eevee':
        scene.render.engine = 'BLENDER_EEVEE'

    scene.render.image_settings.file_format = 'AVI_JPEG'
    return scene


def add_material_for_character(objs):
    char_mat = bpy.data.materials.new(name="characterMaterial")
    char_mat.use_nodes = True
    bsdf = char_mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs[0].default_value = (0.021219, 0.278894, 1, 1)   # character material color
    for obj in objs:
        obj.data.materials.append(char_mat)
