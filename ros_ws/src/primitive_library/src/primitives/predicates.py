import numpy as np
from scipy.spatial import distance

import rospy

from llm_common import helpers as llmh
from llm_simulator.srv import objPos, objMesh
from primitives.action_functions import pick, get_meshes, get_table_mesh, get_shelf_mesh


def can_grasp(object_to_grasp: str, grasp_side: str) -> bool:
    from primitives.action_functions import js_lds
    GRASP_DISTANCE = js_lds.collision_proximity + 0.15

    # Check if the object is in the vicinity of the hand
    object_in_vicinity = True

    dst_to_obj = js_lds.hand_distance_to_obj(object_to_grasp)

    if dst_to_obj > GRASP_DISTANCE:
        object_in_vicinity = False

    # Check if the object is in the workspace by performing a mock pick
    pick(object_to_grasp, 1., 0.0, grasp_side, mock_run=True)
    object_in_workspace = not js_lds._failed_ik

    # Check holding
    (mesh, radii, name) = get_meshes([object_to_grasp], detailed_meshes=True)
    js_lds.set_obstacles(mesh, radii, name)
    is_holding = js_lds.is_holding()

    return (object_in_vicinity and object_in_workspace) or is_holding


def can_reach(object_to_reach: str, grasp_side: str) -> bool:
    from primitives.action_functions import js_lds
    pick(object_to_reach, 1., 0.0, grasp_side, mock_run=True)
    object_reachable = not js_lds._failed_ik
    return object_reachable


def collision_free() -> str:
    from primitives.action_functions import js_lds
    obstacle_collided = ''

    if js_lds._in_collision:
        if js_lds.obj_grasped != js_lds._obstacle_collided:
            obstacle_collided = js_lds._obstacle_collided

    return obstacle_collided


def timeout() -> bool:
    from primitives.action_functions import js_lds
    return not js_lds.timeout


def check_motion_health() -> bool:
    return True


def get_motion_health() -> float:
    from primitives.action_functions import js_lds

    motion_health = js_lds.compute_motion_health()
    return motion_health


def holding() -> bool:
    from primitives.action_functions import js_lds
    is_holding = False
    if js_lds.obj_grasped != "":
        (mesh, radii, name) = get_meshes([js_lds.obj_grasped], detailed_meshes=True)
        js_lds.set_obstacles(mesh, radii, name)
        is_holding = js_lds.is_holding()

    return is_holding


def at_location(object: str, location: str) -> bool:
    from primitives.action_functions import js_lds

    if location == "robot":
        if object == js_lds.obj_grasped:
            return holding()
        else:
            return False

    # TODO: Not super accurate description without radius
    else:

        # Get meshes
        obj_mesh, obj_radii, _ = get_meshes([object], detailed_meshes=True)
        location_mesh, location_radii, _ = get_meshes([location], detailed_meshes=True)
        distances = distance.cdist(obj_mesh, location_mesh)

        # Adjust for radii
        distances = (distances.T - obj_radii).T
        distances = distances - location_radii

        min_dst = np.min(distances)

        return min_dst < 0.5 # TODO: This extension is not super good but ok for now