"""Blender script to render images of 3D models.
This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.
Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2
Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""
import argparse
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple
import numpy as np
import bpy
from mathutils import Vector, Matrix


parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, required=True,
                    help="Path to the object file")
parser.add_argument("--output_dir", type=str, default="./views")
parser.add_argument("--engine", type=str, default="BLENDER_EEVEE",
                    choices=["CYCLES", "BLENDER_EEVEE"])

parser.add_argument("--distance", type=float, default=1.5)
parser.add_argument("--azimuth", type=float, default=0.0)
parser.add_argument("--elevation", type=float, default=0.0)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

context = bpy.context
scene = context.scene
render = scene.render
render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 256
render.resolution_y = 256
render.resolution_percentage = 100
scene.cycles.device = "GPU"
scene.cycles.samples = 32
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True


def add_lighting() -> None:
    # delete the default light
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 30000
    bpy.data.objects["Area"].location[2] = 0.5
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.object.select_all(action='DESELECT')
        this_obj = bpy.ops.import_scene.obj(filepath=object_path, use_edges=False, use_smooth_groups=False, split_mode='OFF')
        for this_obj in bpy.data.objects:
            if this_obj.type == "MESH":
                this_obj.select_set(True)
                bpy.context.view_layer.objects.active = this_obj
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.split_normals()
        bpy.ops.object.mode_set(mode='OBJECT')
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    return scale, offset


def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    return cam


def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)
    reset_scene()
    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    _scale, _offset = normalize_scene()
    add_lighting()
    cam = setup_camera()
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)

    all_theta = [-np.pi/3]
    all_phi = [np.pi/3]
    all_rot = [0.0]
    for i in range(len(all_theta)):
        cam_constraint = cam.constraints.new(type="TRACK_TO")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
        cam_constraint.up_axis = "UP_Y"
        cam_constraint.target = empty

        camera_dist = args.distance
        theta = all_theta[i] + args.azimuth / 180.0 * np.pi
        phi = all_phi[i] - args.elevation / 180.0 * np.pi
        camera_rotation = all_rot[i]

        phi = phi % (2 * np.pi)
        if phi > np.pi:
            theta = theta + np.pi
            phi = 2 * np.pi - phi
        theta = theta % (2 * np.pi)
        phi = np.clip(phi, 0, np.pi)
        camera_rotation = np.clip(camera_rotation, -np.pi, np.pi)
        if phi == 0.0:
            phi = 1e-5
        # set the camera position
        point = (
            camera_dist * math.sin(phi) * math.cos(theta),
            camera_dist * math.sin(phi) * math.sin(theta),
            camera_dist * math.cos(phi),
        )
        cam.location = point
        bpy.ops.object.select_all(action='DESELECT')
        cam.select_set(True)
        bpy.context.view_layer.objects.active = cam
        bpy.ops.object.visual_transform_apply()
        bpy.ops.object.constraints_clear()
        bpy.ops.object.select_all(action='DESELECT')
        location = cam.location.copy()
        right, up, back = cam.matrix_world.to_3x3().transposed()
        direction = np.cross(up,right)
        rotation_vertical = Matrix.Rotation(camera_rotation, 3, Vector(direction))
        matrix = rotation_vertical @ cam.matrix_world.to_3x3()
        cam.matrix_world = matrix.to_4x4()
        cam.location = location
        # render the image
        render_path = os.path.join(args.output_dir, f"{i:03d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
    # bpy.ops.wm.save_as_mainfile(filepath='/home/wufei/research/diffusion_synthetic_dataset/main.blend')


def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        save_images(local_path)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
