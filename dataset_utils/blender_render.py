"""
A simple script that uses blender to render views of a single object by
rotation the camera around it.

Also produces depth map at the same time.

Example:
blender --background --python blender_render.py -- --views 10 /path/to/my.obj

Original source:
https://github.com/panmari/stanford-shapenet-renderer
"""
import itertools
import os
import os.path as osp
from math import radians

import bpy
import numpy as np


def render(obj, scale, remove_doubles, edge_split, output_folder, view_dist, shape):

    bpy.ops.wm.read_homefile(app_template="")
    world = bpy.data.worlds['World']
    world.use_nodes = True

    # changing these values does affect the render.
    bg = world.node_tree.nodes['Background']
    bg.inputs[0].default_value[:3] = (1, 1, 1)
    bg.inputs[1].default_value = 5

    # Set up rendering of depth map:
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create input render layer node
    rl = tree.nodes.new('CompositorNodeRLayers')
    map = tree.nodes.new(type="CompositorNodeNormalize")
    # Size is chosen kind of arbitrarily, try out until you're satisfied with
    # resulting depth map.
    # map.offset = [0]
    # map.size = [0.5]
    # map.use_min = True
    # map.min = [0]
    # map.use_max = True
    # map.max = [255]

    links.new(rl.outputs['Depth'], map.inputs[0])
    # create a file output node and set the path
    depthFileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
    depthFileOutput.label = 'Depth Output'
    links.new(map.outputs[0], depthFileOutput.inputs[0])



    normalFileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
    normalFileOutput.label = 'Image Output'
    links.new(rl.outputs['Image'], normalFileOutput.inputs[0])


    # Delete default cube
    bpy.data.objects["Cube"].select_set(True)
    bpy.ops.object.delete()

    bpy.ops.import_scene.obj(filepath=obj)
    for object in bpy.context.scene.objects:
        if object.name in ['Camera', 'Light']:
            continue
        bpy.context.view_layer.objects.active = object
        if scale != 1:
            bpy.ops.transform.resize(value=(scale, scale, scale))
            bpy.ops.object.transform_apply(scale=True)
        if remove_doubles:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.remove_doubles()
            bpy.ops.object.mode_set(mode='OBJECT')
        if edge_split:
            bpy.ops.object.modifier_add(type='EDGE_SPLIT')
            bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
            bpy.ops.object.modifier_apply(
                apply_as='DATA', modifier="EdgeSplit")

    # Make light just directional, disable shadows.
    light = bpy.data.lights['Light']
    light.type = 'SUN'
    light.energy = 500
    light.use_shadow = True
    # Possibly disable specular shading:
    light.specular_factor = 0

    # Add another light source so stuff facing away from light is not
    # completely dark
    bpy.ops.object.light_add(type='SUN')
    light2 = bpy.data.lights['Sun']
    light2.use_shadow = True
    light2.specular_factor = 0
    light2.energy = .15
    sun = bpy.data.objects['Sun']
    sun.rotation_euler = bpy.data.objects['Light'].rotation_euler
    sun.rotation_euler[0] += 90


    ob = bpy.data.objects['model_normalized']
    mat = bpy.data.materials.new(name="Material")
    mat.diffuse_color = (0.001, 0.001, 0.001, 0)
    mat.specular_color = (0.001, 0.001, 0.001)

    #mat.alpha_threshold = .3
    mat.roughness = .8
    #mat.show_transparent_back = True

    for i, _ in enumerate(ob.data.materials):
        ob.data.materials[i] = mat


    def parent_obj_to_camera(b_camera):
        origin = (0, 0, 0)
        b_empty = bpy.data.objects.new("Empty", None)
        b_empty.location = origin
        b_camera.parent = b_empty  # setup parenting

        collection = bpy.context.collection
        collection.objects.link(b_empty)
        bpy.context.view_layer.objects.active = b_empty
        return b_empty

    scene = bpy.context.scene
    scene.render.resolution_x = shape[1]
    scene.render.resolution_y = shape[0]
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = False

    cam = scene.objects['Camera']

    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty

    #SAVE PATH ORGANIZATION
    class_id, obj_id = osp.split(obj)[0].split('/')[-3:-1]
    base_fp = osp.join(output_folder, "imgs")
    os.makedirs(base_fp, exist_ok=True)
    scene.render.image_settings.file_format = 'PNG'  # set output format to png

    # rotation_mode = 'XYZ'
    azimuths = np.linspace(view_dist['azimuth']['range'][0], view_dist['azimuth']['range'][1], num=view_dist['azimuth']['views'], endpoint=False).tolist()
    elevations = np.linspace(view_dist['elevation']['range'][0], view_dist['elevation']['range'][1], num=view_dist['elevation']['views']).tolist()
    dists = np.linspace(view_dist['dist']['range'][0], view_dist['dist']['range'][1], num=view_dist['dist']['views']).tolist()


    for output_node in [depthFileOutput, normalFileOutput]:
        output_node.base_path = ''

    T_wo = ob.matrix_basis

    dict_list = []
    for i, r in enumerate(itertools.product(azimuths, elevations, dists)):
        print("Rotation:\nAzimuth: {}, Elevation:{}, Distance: {}".format(r[0], r[1], r[2]))
        cam.location = (0, r[2], 0)
        b_empty.rotation_euler[0] = radians(r[1])
        b_empty.rotation_euler[2] = radians(r[0])
        T_wc = b_empty.matrix_basis
        T_co = T_wc.inverted() @ T_wo
        _, out_quat, _ = T_co.decompose()
        scene.frame_set(i)
        scene.render.filepath = base_fp + '/{}_{}_'.format(class_id, obj_id)
        image_file = scene.render.filepath + "######.png"
        depth_file = scene.render.filepath + "######_d.png"
        depthFileOutput.file_slots[0].path = depth_file
        normalFileOutput.file_slots[0].path = image_file
        #SAMPLE DICTIONARY FOR AN IMAGE
        sample_dict = {"class_id": class_id,
                       "obj_id": obj_id,
                       "image_file": "{}_{}_{}.png".format(class_id, obj_id, str(bpy.context.scene.frame_current).zfill(6)),
                       "depth_file": "{}_{}_{}_d.png".format(class_id, obj_id, str(bpy.context.scene.frame_current).zfill(6)),
                       "azimuth": r[0],
                       "elevation": r[1],
                       "dist": r[2],
                       "pose": list(out_quat)}

        dict_list.append(sample_dict)
        bpy.ops.render.render()  # render still



    return dict_list

import argparse

parser = argparse.ArgumentParser(
    description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=tuple, default=(8, 4),
                    help='number of views to be rendered (azimuth, elevation)')
parser.add_argument('--obj', type=str,
                    help='Path to the obj file to be rendered.',
                    default="/Volumes/Storage/ShapeNetCore.v2/02691156/1ac29674746a0fc6b87697d3904b168b/models/model_normalized.obj")
parser.add_argument('--output_folder', type=str, default='',
                    help='The path the output will be dumped to.')
parser.add_argument('--scale', type=float, default=1,
                    help='Scaling factor applied to model. '
                         'Depends on size of mesh.')
parser.add_argument('--remove_doubles', action='store_true',
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', action='store_true',
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=1,
                    help='Scaling that is applied to depth. '
                         'Depends on size of mesh. '
                         'Try out various values until you get a good '
                         'result.')
parser.add_argument('--shape', type=int, default=[192, 256], nargs=2,
                    help='2D shape of rendered images.')


if __name__ == '__main__':
    category_ids = {
        'Airplane': '02691156',
        'Car': '02958343',
        'Chair': '03001627',
        'Guitar': '03467517',
        'Train': '04468005',
        'Motorbike': '03790512',
        'Sofa': '04256520',
        'Mug': '03797390',
        'Table': '04379243',
        'Lamp': '03636649'
    }
    view_dist = {"azimuth": {"range": (0, 360),
                              "views": 36},
                                               "elevation": {"range": (-30, 30),
                                                             "views": 6},
                                               "dist": {"range": (1, 2.5),
                                                        "views": 3}
                 }
    args = parser.parse_args()
    ll = render(args.obj, 1, True, False, args.output_folder, view_dist, args.shape)
    print(ll)
