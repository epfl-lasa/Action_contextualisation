import typing as tp
import numpy as np
from scipy.spatial.transform import Rotation
import contextlib

import rospy

from primitives.js_lds_oa import JS_LDS_OA
from llm_common import utils as llmu
from llm_common import helpers as llmh
from llm_simulator.srv import objPos, objMesh


static_elements = []
js_lds = None

approach_side_dst = 0.05
approach_ee_offset_side = 0.14
approach_ee_offset_side_additional = 0.02
approach_ee_offset_top = 0.19
approach_top_dst = 0.00
approach_top_dst_offset = 0.15
table_altitude = 0.00


def robot_action(func):
    def wrapper_robot_action(*args, **kwargs):

        # Controller reset
        global js_lds
        if js_lds is None:
            reset_controller()

        # Function execution
        func(js_lds, *args, **kwargs)

    return wrapper_robot_action


def reset_controller():
    global js_lds
    with contextlib.redirect_stdout(None):
        js_lds = JS_LDS_OA(llmu.IIWA_URDF_FOLDER,
                        llmu.IIWA_URDF_PATH,
                        llmu.ALLEGRO_URDF_FOLDER,
                        llmu.ALLEGRO_URDF_PATH)
        js_lds.send_robot_zero_torque()


def get_shelf_mesh() -> np.ndarray:
    x_min = -0.4
    x_max = 0.2
    y_min = -0.65
    y_max = -0.35
    z_min = 0.45  # Only model the top part of the shelf
    z_max = 0.9

    total_nb_pts = 200

    # Generate top obstascle
    nb_pts_per_side = int(np.round(np.sqrt(total_nb_pts)))
    x_range = np.linspace(x_min, x_max, nb_pts_per_side)
    y_range = np.linspace(y_min, y_max, nb_pts_per_side)
    z_range = np.array([z_max])
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
    xx = np.expand_dims(xx.flatten(), axis=1)
    yy = np.expand_dims(yy.flatten(), axis=1)
    zz = np.expand_dims(zz.flatten(), axis=1)
    top_mesh = np.concatenate((xx, yy, zz), axis=1)

    # Generate side pannel 1
    x_range = np.array([x_min])
    y_range = np.linspace(y_min, y_max, nb_pts_per_side)
    z_range = np.linspace(z_min, z_max, nb_pts_per_side)
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
    xx = np.expand_dims(xx.flatten(), axis=1)
    yy = np.expand_dims(yy.flatten(), axis=1)
    zz = np.expand_dims(zz.flatten(), axis=1)
    side_mesh_1 = np.concatenate((xx, yy, zz), axis=1)

    # Generate side pannel 2
    x_range = np.array([x_max])
    y_range = np.linspace(y_min, y_max, nb_pts_per_side)
    z_range = np.linspace(z_min, z_max, nb_pts_per_side)
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
    xx = np.expand_dims(xx.flatten(), axis=1)
    yy = np.expand_dims(yy.flatten(), axis=1)
    zz = np.expand_dims(zz.flatten(), axis=1)
    side_mesh_2 = np.concatenate((xx, yy, zz), axis=1)

    # Generate front pannel
    x_range = np.linspace(x_min, x_max, nb_pts_per_side)
    y_range = np.array([y_max])
    z_range = np.linspace(z_min, z_max, nb_pts_per_side)
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
    xx = np.expand_dims(xx.flatten(), axis=1)
    yy = np.expand_dims(yy.flatten(), axis=1)
    zz = np.expand_dims(zz.flatten(), axis=1)
    front_mesh = np.concatenate((xx, yy, zz), axis=1)

    return np.concatenate((front_mesh, side_mesh_1, side_mesh_2, top_mesh), axis=0)


def get_table_mesh() -> np.ndarray:
    nb_table_points = 1000
    nb_legs_points = 50
    table_x_min = 0.35
    table_x_max = 1.05
    table_y_min = -0.5
    table_y_max = 0.7

    nb_pts_per_side = int(np.round(np.sqrt(nb_table_points)))
    x_range = np.linspace(table_x_min, table_x_max, nb_pts_per_side)
    y_range = np.linspace(table_y_min, table_y_max, nb_pts_per_side)
    z_range = np.array([0.99])

    xx, yy, zz = np.meshgrid(x_range, y_range, z_range)

    xx = np.expand_dims(xx.flatten(), axis=1)
    yy = np.expand_dims(yy.flatten(), axis=1)
    zz = np.expand_dims(zz.flatten(), axis=1)

    mesh = np.concatenate((xx, yy, zz), axis=1)

    return mesh


def get_meshes(obj_names: tp.List[str], detailed_meshes: bool = True, use_robot_frame: bool = True):
    rospy.wait_for_service('objMesh')
    rospy.wait_for_service('objPos')

    obj_frame_service = rospy.ServiceProxy('objPos', objPos, persistent=True)
    mesh_service = rospy.ServiceProxy('objMesh', objMesh, persistent=True)

    iiwa_pos = obj_frame_service('kuka_base').object_position
    iiwa_base_pos = llmh.mujoco_pos_quat_to_se3(iiwa_pos[:3], iiwa_pos[3:])

    meshes = None
    radius = None
    names = None
    for name in obj_names:

        if (name in llmu.ROUND_OBJECTS and not detailed_meshes) and False:
            new_mesh, new_radius = get_mesh_for_round_object(name)
        else:

            if name == "table" and False:
                new_mesh = get_table_mesh()
                if detailed_meshes:
                    new_radius = np.ones([new_mesh.shape[0]]) * 0.006  # from trigo
                else:
                    new_radius = np.ones([new_mesh.shape[0]]) * 0.01
            elif name == "shelf" and False:
                new_mesh = get_shelf_mesh()
                if detailed_meshes:
                    new_radius = np.ones([new_mesh.shape[0]]) * 0.007
                else:
                    new_radius = np.ones([new_mesh.shape[0]]) * 0.01
            else:
                mesh_details = mesh_service(name)
                new_mesh = np.reshape(mesh_details.object_vertices, (-1, 3))
                new_radius = np.asarray(mesh_details.object_radii)
                if detailed_meshes:
                    new_radius -= 0.001
                else:
                    new_radius += 0.02

        if use_robot_frame:
            new_mesh = (np.linalg.inv(iiwa_base_pos) @ np.concatenate((new_mesh, np.ones((new_mesh.shape[0], 1))), axis=1).T).T[:, :3]
        new_names = [name] * new_mesh.shape[0]
        if meshes is None:
            meshes = new_mesh
            radius = new_radius
            names = new_names
        else:
            meshes = np.concatenate([meshes, new_mesh], axis=0)
            radius = np.concatenate([radius, new_radius], axis=0)
            names += new_names

    return (meshes, radius, names)


def get_mesh_for_round_object(obj_name: str) -> tp.Tuple[np.ndarray, np.ndarray]:
    rospy.wait_for_service('objMesh')
    rospy.wait_for_service('objComPos')
    com_pos_service = rospy.ServiceProxy('objComPos', objPos, persistent=True)
    mesh_service = rospy.ServiceProxy('objMesh', objMesh, persistent=True)

    obj_com_pos = np.asarray(com_pos_service(obj_name).object_position)
    obj_geom = mesh_service(obj_name)

    obj_mesh = np.reshape(obj_geom.object_vertices, (-1, 3))
    obj_radii = np.asarray(obj_geom.object_radii)
    radius = np.max(np.linalg.norm(obj_mesh - obj_com_pos, axis=1) + obj_radii)

    return (np.array([obj_com_pos]), np.array([radius]))


def update_static_obstacles() -> None:
    global static_elements
    if len(static_elements) == 0:
        static_elements = np.asarray(get_meshes(llmu.STATIC_ELEMENTS))


@robot_action
def approach(js_lds, object_to_grasp: str,
             speed: float,
             obstacle_clearance: tp.Optional[float],
             grasp: str,
             orientation: float = 0.,
             disregard_object_to_grasp: bool = False,
             detailed_obstacles: bool = False,
             apply_offsets: bool = True,
             obstacle_ik: bool = False,
             vertical_clearance_offset: float = 0.,
             force_altitude: tp.Optional[float] = None,
             mock_run: bool = False,
             disregard_table: bool = False,
             drop_side_offset: bool = False,
             disregard_object: tp.Optional[str] = None) -> None:

    # Default grasp to top
    if grasp == '':
        grasp = 'top'
    # Waiting for services
    rospy.wait_for_service('objComPos')
    rospy.wait_for_service('objPos')
    rospy.wait_for_service('objMesh')
    obj_frame_service = rospy.ServiceProxy('objPos', objPos, persistent=True)
    com_pos_service = rospy.ServiceProxy('objComPos', objPos, persistent=True)
    mesh_service = rospy.ServiceProxy('objMesh', objMesh, persistent=True)

    # Getting object positions
    iiwa_pos = obj_frame_service('kuka_base').object_position
    iiwa_base_pos = llmh.mujoco_pos_quat_to_se3(iiwa_pos[:3], iiwa_pos[3:])
    obj_com_pos = np.ones((4, 1))
    obj_com_pos[:3] = np.expand_dims(com_pos_service(object_to_grasp).object_position, axis=1)

    # Here we are doing something cheeky because I am too tired to figure out why the proper way doesn't work. TODO: Do this right
    if object_to_grasp in ['shelf', 'table']:
        (obj_mesh, obj_radii, _) = get_meshes([object_to_grasp], detailed_meshes=True, use_robot_frame=False)
    else:
        obj_mesh = np.reshape(mesh_service(object_to_grasp).object_vertices, (-1, 3))
        obj_radii = np.asarray(mesh_service(object_to_grasp).object_radii)

    base_obj_vec = np.squeeze(obj_com_pos[:3]) - iiwa_base_pos[:3, 3]
    obj_goal = obj_com_pos

    # Find direction of approach
    approach_direction = np.copy(base_obj_vec)
    approach_direction[2] = 0
    approach_direction = approach_direction / np.linalg.norm(approach_direction)
    approach_direction_perp = np.array([-approach_direction[1], approach_direction[0], 0])

    # Handle offset for approach
    grasp_orientation = grasp
    if grasp == "side":
        # Compute object radius
        obj_radii_perp = (obj_mesh.T - obj_com_pos[:3]).T @ np.expand_dims(approach_direction_perp,axis=1)
        obj_radius_perp = -np.min(obj_radii_perp - obj_radii)
        obj_radii_parl = (obj_mesh.T - obj_com_pos[:3]).T @ np.expand_dims(approach_direction,axis=1)
        obj_radius_parl = -np.min(obj_radii_parl - obj_radii)

        # Adjust goal position to object
        obj_goal[:3] -= np.expand_dims(approach_direction_perp * (obj_radius_perp + 0.03), axis=1)   # Adding hand width
        obj_goal[:3] -= np.expand_dims(approach_direction * (obj_radius_parl + approach_ee_offset_side), axis=1)

        if apply_offsets:
            obj_goal[:3] -= np.expand_dims(approach_direction_perp * (approach_side_dst+obstacle_clearance), axis=1)
            obj_goal[:3] -= np.expand_dims(approach_direction * approach_ee_offset_side_additional, axis=1)
        obj_yaw = np.degrees(np.arctan2(base_obj_vec[1], base_obj_vec[0]))

        # Adjust altitude if we are over the table
        if obj_goal[2] < table_altitude:
            obj_goal[2] = table_altitude

        if orientation == 0:
            goal_rot = Rotation.from_euler('xyz', [90, 0, 90 + obj_yaw], degrees=True).as_quat()

    elif grasp == "top":

        obj_goal[2] = np.max(obj_mesh[:, 2]+obj_radii)
        obj_goal[2] += approach_top_dst
        obj_goal[2] += vertical_clearance_offset

        if apply_offsets:
            obj_goal[2] += obstacle_clearance
            obj_goal[2] += approach_top_dst_offset

        if drop_side_offset:
            obj_goal[:3] -= np.expand_dims(approach_direction * 0.15, axis=1)
        else:
            if object_to_grasp == "paper_ball_2":
                obj_goal[:3] -= np.expand_dims(approach_direction_perp * (0.025), axis=1)
            elif object_to_grasp == "paper_ball_1":
                obj_goal[:3] -= np.expand_dims(approach_direction_perp * (0.025), axis=1)
            else:
                obj_goal[:3] -= np.expand_dims(approach_direction_perp * (0.025), axis=1)

            obj_goal[:3] -= np.expand_dims(approach_direction * approach_ee_offset_top, axis=1)

        obj_yaw = np.degrees(np.arctan2(base_obj_vec[1], base_obj_vec[0]))

        if orientation == 0:
            goal_rot = Rotation.from_euler('xyz', [0, 90, obj_yaw], degrees=True).as_quat()

        if force_altitude is not None:
            obj_goal[2] = force_altitude

    else:
        raise ValueError(f"Unknown grasp: {grasp}")

    if orientation != 0:
        goal_rot = np.array([obj_yaw])

    # Compute goal position in IIWA frame
    print(obj_goal)
    obj_pos_in_iiwa = np.squeeze(np.linalg.inv(iiwa_base_pos) @ obj_goal)[:3]
    if goal_rot is not None:
        obj_pos_in_iiwa = np.concatenate([obj_pos_in_iiwa, goal_rot])

    # Initializing controller
    js_lds.joint_speed_scale = speed
    js_lds.orientation_factor = orientation

    if obstacle_clearance is None:
        js_lds.reset_collosion_proximity()
    else:
        js_lds.collision_proximity = obstacle_clearance

    # Set obstacles
    obstacles = ["apple", "eaten_apple",
                 "paper_ball_1", "paper_ball_2",
                 # "paper_ball_3",
                 "champagne_1", "champagne_2",
                 "table", "sink", "shelf", "trash_bin"]
    # obstacles = ["eaten_apple", "champagne_1", "sink", "shelf", "trash_bin", "table"]

    if disregard_table:
        obstacles.remove("table")

    if disregard_object_to_grasp or js_lds.grasping:
        js_lds._obstacle_to_approach = js_lds.obj_grasped
    elif disregard_object is not None:
        js_lds._obstacle_to_approach = disregard_object 
    else:
        js_lds._obstacle_to_approach = ""

    (meshes, radii, names) = get_meshes(obstacles, detailed_meshes=detailed_obstacles)
    js_lds.set_obstacles(meshes, radii, names)
    js_lds._obstacle_ik = obstacle_ik
    js_lds.cartesian_goal = obj_pos_in_iiwa
    if not js_lds._failed_ik and not mock_run:
        js_lds.run_controller()


@robot_action
def pick(js_lds, object_to_grasp: str,
         speed: float,
         obstacle_clearance: tp.Optional[float] = None,
         grasp_orientation: tp.Optional[str] = None,
         mock_run: bool = False) -> None:

    

    # Fine-tuned approach
    approach(object_to_grasp, speed, obstacle_clearance,
                grasp_orientation, 0.,
                disregard_object_to_grasp=False,
                detailed_obstacles=True,
                apply_offsets=False,
                obstacle_ik=True,
                vertical_clearance_offset=0.,
                mock_run=mock_run,
                disregard_table=False,
                disregard_object=object_to_grasp)
    if mock_run:
        return

    if not js_lds._in_collision or js_lds._obstacle_collided == object_to_grasp or True:
        js_lds._in_collision = False
        js_lds._obstacle_collided = ''
        # Perform grasping
        js_lds.grasping = True
        js_lds.obj_grasped = object_to_grasp
        obstacles = ["apple", "eaten_apple",
                    "paper_ball_1", "paper_ball_2",
                    #"paper_ball_3",
                    "champagne_1", "champagne_2", "sink", "shelf",
                    "trash_bin"]
        #obstacles = ["eaten_apple", "champagne_1", "sink", "shelf", "trash_bin"]


        js_lds._obstacle_to_approach = js_lds.obj_grasped

        (meshes, radii, names) = get_meshes(obstacles, detailed_meshes=True)
        js_lds.set_obstacles(meshes, radii, names)
        js_lds._obstacle_ik = True
        try:
            js_lds.run_controller()
        except ValueError as e:
            if not (js_lds._in_collision and js_lds._obstacle_collided == object_to_grasp):
                raise e

    # Flyoff straight up to avoid some collisions
    if not js_lds._in_collision or js_lds._obstacle_collided == object_to_grasp:
        js_lds._in_collision = False
        js_lds._obstacle_collided = ''
        # Fly off as high as possible up to 0.4
        current_hand_pos = js_lds.hand_position[:3, 3]
        for fly_off_offset in [0.2]:
            hand_pos_goal = np.copy(current_hand_pos)
            hand_pos_goal[2] += fly_off_offset
            js_lds.orientation = 1.
            js_lds.cartesian_goal = hand_pos_goal
            if not js_lds._failed_ik and not js_lds._in_collision:
                print(f"Flyoff {fly_off_offset}")
                js_lds.run_controller()
            else:
                break


@robot_action
def place(js_lds, object_to_grasp: str, orientation: float, speed: float, obstacle_clearance: float) -> None:

    for vertical_offset in [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, .0]:
        if js_lds._in_collision:
            break

        approach(object_to_grasp,
                speed, grasp="top",
                orientation=orientation,
                detailed_obstacles=True,
                disregard_object_to_grasp=True,
                vertical_clearance_offset=vertical_offset,
                disregard_table = False,
                apply_offsets=False,
                obstacle_clearance=obstacle_clearance,
                drop_side_offset=True)
        if not js_lds._failed_ik:
            break
        else:
            print("Failed IK")

    if not js_lds._in_collision:
        for vertical_offset in [0.15, 0.15, 0.2]:
            approach(object_to_grasp,
                        speed, grasp="top",
                        orientation=orientation,
                        detailed_obstacles=True,
                        disregard_object_to_grasp=True,
                        vertical_clearance_offset=vertical_offset,
                        disregard_table = False,
                        apply_offsets=False,
                        obstacle_clearance=obstacle_clearance,
                        drop_side_offset=True)
            if not js_lds._failed_ik:
                break
            else:
                print("Failed IK")

    if not js_lds._in_collision or True:
        # Drop the object
        js_lds.let_go = True
        js_lds.run_controller()
        js_lds.grasping = False
        
    if not js_lds._in_collision or True:
        # Flyoff
        print("Flyoff")
        hand_pos_goal = js_lds.hand_position[:3, 3]
        hand_pos_goal[2] += 0.4
        js_lds.orientation = .9
        js_lds.cartesian_goal = hand_pos_goal

        if js_lds._failed_ik:
            hand_pos_goal[2] -= 0.1
            js_lds.cartesian_goal = hand_pos_goal

        if js_lds._failed_ik:
            hand_pos_goal[2] -= 0.1
            js_lds.cartesian_goal = hand_pos_goal

        if not js_lds._failed_ik:
            js_lds.run_controller()
            js_lds.obj_grasped = ""
        js_lds.let_go = False



@robot_action
def drop(js_lds, object_to_grasp: str,
         speed: float = 1.,
         obstacle_clearance: float = 0.05,
         orientation: float = 0) -> None:
    print(orientation)
    approach(object_to_grasp,
             speed, grasp="top",
             orientation=orientation,
             detailed_obstacles=False,
             disregard_object_to_grasp=True,
             disregard_table=False,
             apply_offsets=False,
             force_altitude=0.8,
             obstacle_clearance=obstacle_clearance,
             drop_side_offset=True)
    if not js_lds._in_collision and not js_lds._failed_ik:
        # Drop the object
        js_lds.let_go = True
        js_lds.grasping = False
        js_lds.run_controller()
        js_lds.let_go = False
        js_lds.obj_grasped = ""


@robot_action
def throw(js_lds, object_to_grasp: str,
          speed: float = 1.,
          obstacle_clearance: float = 0.05) -> None:

    approach(object_to_grasp,
             speed, grasp="top",
             orientation=0.,
             disregard_object_to_grasp=True,
             force_altitude=1.3,
             obstacle_clearance=obstacle_clearance,
             drop_side_offset=True)

    if not js_lds._in_collision:
        # Drop the object
        js_lds.let_go = True
        js_lds.grasping = False
        js_lds.run_controller()
        js_lds.let_go = False
        js_lds.obj_grasped = ""
