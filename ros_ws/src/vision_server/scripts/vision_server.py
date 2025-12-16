import numpy as np
import time

import rospy
from geometry_msgs.msg import PoseStamped

from llm_simulator.srv import objPos, objPosResponse, objPosRequest
from llm_simulator.srv import objMesh, objMeshResponse, objMeshRequest
from llm_simulator.srv import inertia, inertiaResponse, inertiaRequest
from iiwa_tools.srv import GetMassMatrix, GetMassMatrixRequest
import llm_common.helpers as llmh

class VisionServer:
    VRPN_TOPICS_NAMES = [
        "/vrpn_client_node/apple/pose",
        "/vrpn_client_node/champagne_1/pose",
        "/vrpn_client_node/champagne_2/pose",
        "/vrpn_client_node/eaten_apple/pose",
        "/vrpn_client_node/iiwa_base7/pose",
        "/vrpn_client_node/paper_ball_1/pose",
        "/vrpn_client_node/paper_ball_2/pose",
        #"/vrpn_client_node/paper_ball_3/pose",
        "/vrpn_client_node/shelf/pose",
        "/vrpn_client_node/sink/pose",
        "/vrpn_client_node/table/pose",
        "/vrpn_client_node/trash_bin/pose",
        "/vrpn_client_node/iiwa_base7/pose"
        ]
    
    COM_Y_OFFSETS = {
        'apple': 0,
        'eaten_apple': -0.3,
        'champagne_1': 0.045,
        'champagne_2': 0.045,
        'paper_ball_1': 0,
        'paper_ball_2': 0,
        'iiwa_base7': 0,
        'shelf': 0,
        'sink': 0,
        'table': 0,
        'trash_bin': 0,
    }


    OPTI_WORLD_TO_WORLD = [0, 0, 0, 
                           0.5, 0.5, 0.5, 0.5]

    IIWA_BASE_IN_IIWA_MARKER = [-3.222253154994657309e-02,
                                -4.178227469017845030e-03,
                                -7.590960376033785428e-02,
                                1.,
                                0.,
                                0.,
                                0.]
    
    INERTIA_SERVER = "/iiwa/iiwa_mass_server"

    def __init__(self):
        self.object_poses = {}
        for topic in self.VRPN_TOPICS_NAMES:
            print(topic)
            rospy.Subscriber(topic, PoseStamped, self.pose_callback, (topic,))
        
        rospy.wait_for_service(self.INERTIA_SERVER)
        self._inertia_iiwa_server = rospy.ServiceProxy(self.INERTIA_SERVER, GetMassMatrix, persistent=True)

        self._inertia_srv = rospy.Service('inertia', inertia, self._inertia_srv_cb)
        self._obj_pos_srv = rospy.Service('objPos', objPos, self._obj_pos_srv_cb)
        self._obj_cop_pos_srv = rospy.Service('objComPos', objPos, self._obj_com_pos_srv_cb)
        self._obj_mesh_srv = rospy.Service('objMesh', objMesh, self._obj_mesh_srv_cb)

    def get_table_mesh(self):
        nb_table_points = 1000
        table_x_min = -0.6
        table_x_max = 0.6
        table_z_min = -0.45
        table_z_max =  0.45

        nb_pts_per_side = int(np.round(np.sqrt(nb_table_points)))
        x_range = np.linspace(table_x_min, table_x_max, nb_pts_per_side)
        y_range = np.array([-0.0401-0.01-0.02])
        z_range = np.linspace(table_z_min, table_z_max, nb_pts_per_side)

        xx, yy, zz = np.meshgrid(x_range, y_range, z_range)

        xx = np.expand_dims(xx.flatten(), axis=1)
        yy = np.expand_dims(yy.flatten(), axis=1)
        zz = np.expand_dims(zz.flatten(), axis=1)

        mesh = np.concatenate((xx, yy, zz), axis=1)
        radii = np.ones((mesh.shape[0])) * 0.04517

        return mesh, radii
    
    def get_sink_mesh(self): 
        nb_table_points = 400
        table_x_min = -0.2
        table_x_max = 0.2
        table_z_min = -0.15
        table_z_max =  0.15

        nb_pts_per_side = int(np.round(np.sqrt(nb_table_points)))
        x_range = np.linspace(table_x_min, table_x_max, nb_pts_per_side)
        y_range = np.array([-0.03])
        z_range = np.linspace(table_z_min, table_z_max, nb_pts_per_side)

        xx, yy, zz = np.meshgrid(x_range, y_range, z_range)

        xx = np.expand_dims(xx.flatten(), axis=1)
        yy = np.expand_dims(yy.flatten(), axis=1)
        zz = np.expand_dims(zz.flatten(), axis=1)

        mesh = np.concatenate((xx, yy, zz), axis=1)
        radii = np.ones((mesh.shape[0])) * 0.015

        return mesh, radii

    def get_shelf_mesh(self):
        nb_table_points = 400
        table_x_min = -0.16
        table_x_max = 0.16
        table_z_min = -0.25
        table_z_max =  0.25

        nb_pts_per_side = int(np.round(np.sqrt(nb_table_points)))
        x_range = np.linspace(table_x_min, table_x_max, nb_pts_per_side)
        y_range = np.array([-0.04])
        z_range = np.linspace(table_z_min, table_z_max, nb_pts_per_side)

        xx, yy, zz = np.meshgrid(x_range, y_range, z_range)

        xx = np.expand_dims(xx.flatten(), axis=1)
        yy = np.expand_dims(yy.flatten(), axis=1)
        zz = np.expand_dims(zz.flatten(), axis=1)

        mesh = np.concatenate((xx, yy, zz), axis=1)
        radii = np.ones((mesh.shape[0])) * 0.02
        return mesh, radii
    
    def get_bin_mesh(self):
        nb_table_points = 400
        table_x_min = -0.18
        table_x_max = 0.18
        table_z_min = -0.18
        table_z_max =  0.18

        nb_pts_per_side = int(np.round(np.sqrt(nb_table_points)))
        x_range = np.linspace(table_x_min, table_x_max, nb_pts_per_side)
        y_range = np.array([0])
        z_range = np.linspace(table_z_min, table_z_max, nb_pts_per_side)

        xx, yy, zz = np.meshgrid(x_range, y_range, z_range)

        xx = np.expand_dims(xx.flatten(), axis=1)
        yy = np.expand_dims(yy.flatten(), axis=1)
        zz = np.expand_dims(zz.flatten(), axis=1)

        mesh = np.concatenate((xx, yy, zz), axis=1)
        radii = np.ones((mesh.shape[0])) * 0.02
        return mesh, radii

    def get_paper_ball_mesh(self):
        return np.array([[0., 0., .0]]), np.array([0.04])
    
    def get_apple_mesh(self):
        return np.array([[0., 0., .0]]), np.array([0.07])
    
    def get_eaten_apple_mesh(self):
        return np.array([[0., 0., .0]]), np.array([0.06])
    
    def get_champagne_mesh(self):

        mesh = np.array([[0., -.10, 0.],
                         [0., -.05, 0.],
                         [0., -.01, 0.],
                         [0.,  .02, 0.]])
        radii = np.array([.04, .04, .04, 0.04])

        return mesh, radii

    def get_mesh(self, mesh_name):

        if mesh_name == 'sink':
            mesh, radii = self.get_sink_mesh()
        elif mesh_name == 'table':
            mesh, radii = self.get_table_mesh()
        elif mesh_name == 'shelf':
            mesh, radii = self.get_shelf_mesh()
        elif mesh_name == 'trash_bin':
            mesh, radii = self.get_bin_mesh()
        elif mesh_name in ['paper_ball_1', 'paper_ball_2']:
            mesh, radii = self.get_paper_ball_mesh()
        elif mesh_name == 'apple':
            mesh, radii = self.get_apple_mesh()
        elif mesh_name == 'eaten_apple':
            mesh, radii = self.get_eaten_apple_mesh()
        elif mesh_name in ['champagne_1', 'champagne_2']:
            mesh, radii = self.get_champagne_mesh()
        else:
            raise ValueError(f"Mesh {mesh_name} not found")

        return mesh, radii

    def pose_callback(self, data, args):
        topic = args[0]  # Extract topic from arguments
        object_name = topic.split('/')[-2]
        # Update pose of object in object_poses dictionary
        pose = data.pose
        quat = pose.orientation
        position = pose.position
        full_cat_pose = np.asarray([position.x, position.y, position.z, quat.w, quat.x, quat.y, quat.z])
        self.object_poses[object_name] = full_cat_pose

    def _obj_pos_srv_cb(self, req: objPosRequest):

        # Get object position
        obj_name = req.object_name
        if obj_name == 'kuka_base':
            obj_name = 'iiwa_base7'
            
        # Wait for object position
        while not obj_name in self.object_poses.keys():
            time.sleep(0.1)

        # Get object position
        obj_pos = self.object_poses[obj_name]

        # Apply transformation to put Z up
        obj_se3 = llmh.mujoco_pos_quat_to_se3(obj_pos[:3], obj_pos[3:])
        iiwa_frame_se3 = llmh.mujoco_pos_quat_to_se3(self.OPTI_WORLD_TO_WORLD[:3],
                                                    self.OPTI_WORLD_TO_WORLD[3:])
        obj_full_se3 = iiwa_frame_se3 @ obj_se3
        obj_pos = llmh.se3_to_mujoco_cartesian(obj_full_se3)

        # Account for marker-urdf offset with the iiwa_base
        if obj_name == 'iiwa_base7':
            
            obj_se3 = llmh.mujoco_pos_quat_to_se3(obj_pos[:3], obj_pos[3:])
            iiwa_frame_se3 = llmh.mujoco_pos_quat_to_se3(self.IIWA_BASE_IN_IIWA_MARKER[:3],
                                                         self.IIWA_BASE_IN_IIWA_MARKER[3:])
            obj_full_se3 = obj_se3 @ iiwa_frame_se3
            obj_pos = llmh.se3_to_mujoco_cartesian(obj_full_se3)

        # Send obj pos
        reply = objPosResponse()
        
        reply.object_position = obj_pos.tolist()
        return reply

    def _obj_com_pos_srv_cb(self, req: objPosRequest):

        # Get object position
        obj_name = req.object_name

        # Wait for object position
        while not obj_name in self.object_poses.keys():
            time.sleep(0.1)

        # Get object position
        obj_pos = self.object_poses[obj_name]
        #obj_pos[1] += self.COM_Y_OFFSETS[obj_name]  # TODO: Wrong axis

        # Apply transformation to put Z up
        obj_se3 = llmh.mujoco_pos_quat_to_se3(obj_pos[:3], obj_pos[3:])
        iiwa_frame_se3 = llmh.mujoco_pos_quat_to_se3(self.OPTI_WORLD_TO_WORLD[:3],
                                                    self.OPTI_WORLD_TO_WORLD[3:])
        obj_full_se3 =  iiwa_frame_se3 @ obj_se3
        obj_pos = llmh.se3_to_mujoco_cartesian(obj_full_se3)

        if obj_name == 'iiwa_base7':
            obj_se3 = llmh.mujoco_pos_quat_to_se3(obj_pos[:3], obj_pos[3:])
            iiwa_frame_se3 = llmh.mujoco_pos_quat_to_se3(self.IIWA_BASE_IN_IIWA_MARKER[:3],
                                                         self.IIWA_BASE_IN_IIWA_MARKER[3:])
            obj_full_se3 = obj_se3 @ iiwa_frame_se3
            obj_pos = llmh.se3_to_mujoco_cartesian(obj_full_se3)

        # Only keep tranlation
        obj_pos = obj_pos[:3]

        # Send obj pos
        reply = objPosResponse()
        reply.object_position = obj_pos.tolist()
        return reply

    def _obj_mesh_srv_cb(self, req: objMeshRequest):
        # Wait for object position
        obj_name = req.object_name
        print(f"Answering to mesh request {obj_name}")
        while not obj_name in self.object_poses.keys():
            breakpoint()
            time.sleep(0.1)

        # Get mesh position
        obj_name = req.object_name
        obj_vertices, obj_radii = self.get_mesh(obj_name)

        # Adjust mesh position to world
        obj_pos = self.object_poses[obj_name]

        # Apply transformation to put Z up
        obj_se3 = llmh.mujoco_pos_quat_to_se3(obj_pos[:3], obj_pos[3:])
        iiwa_frame_se3 = llmh.mujoco_pos_quat_to_se3(self.OPTI_WORLD_TO_WORLD[:3],
                                                    self.OPTI_WORLD_TO_WORLD[3:])
        obj_full_se3 = iiwa_frame_se3 @ obj_se3
        obj_pos = llmh.se3_to_mujoco_cartesian(obj_full_se3)

        obj_pos_se3 = llmh.mujoco_pos_quat_to_se3(obj_pos[:3], obj_pos[3:])
        obj_vert_ext = np.ones((obj_vertices.shape[0], obj_vertices.shape[1]+1))
        obj_vert_ext[:, :-1] = obj_vertices

        obj_vert_inplace = (obj_pos_se3 @ obj_vert_ext.T).T
        obj_vertices = obj_vert_inplace[:, :-1]

        # Send obj pos
        reply = objMeshResponse()
        reply.object_vertices = obj_vertices.flatten().tolist()
        reply.object_radii = obj_radii.flatten().tolist()
        return reply

    def _inertia_srv_cb(self, req: inertiaRequest) -> None:
        
        inertia_req = GetMassMatrixRequest()
        inertia_req.joint_angles = req.jnt
        iiwa_inertia = np.reshape(self._inertia_iiwa_server(inertia_req).mass_matrix.data, (7, 7))
        inertia = np.eye(23)   # Neglect hand inertia
        inertia[:7, :7] = iiwa_inertia
        reply = inertiaResponse()
        reply.inertia = inertia.flatten().tolist()
        return reply


def main():
    rospy.init_node("vision_server")
    vision_server = VisionServer()
    rospy.loginfo("Vision server initialized")
    rospy.spin()


if __name__ == "__main__":
    main()