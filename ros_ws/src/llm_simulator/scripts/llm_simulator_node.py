"""Main node responsible for running the MuJoCo simulation and setting up topics the same way the real robot would."""
import sys
import time
import threading
import signal
from controller_utils import controller_utils
from llm_simulator.srv import inertia, inertiaResponse, inertiaRequest
from llm_simulator.srv import objPos, objPosResponse, objPosRequest
from llm_simulator.srv import objMesh, objMeshResponse, objMeshRequest
from llm_common import utils as llmu
import mujoco
from mujoco import viewer
import numpy as np

import rospy
from std_msgs.msg import Float64MultiArray, Header, Bool
from sensor_msgs.msg import JointState


class LLMSimulator:

    PUBLISHER_RATE = 200  # Hz

    def __init__(self) -> None:

        # Generic flags on the state of the simulation (released=false, locked=true)
        self._sim_initialized = threading.Lock()
        self._sim_running = threading.Lock()

        # Mujoco simulation
        self._model = None
        self._data = None
        self._view = None
        self._controller = None
        self._setup_MuJoco()

        # Parameters for grav comp
        self._latest_iiwa_cmd = np.zeros(len(llmu.IIWA_JOINT_NAMES))
        self._latest_allegro_cmd = np.zeros(len(llmu.ALLEGRO_JOINT_NAMES))

        # Locks for multithreading
        self._controller_lock = threading.Lock()  # Locks view and controller

        # Subs and publishers
        self._iiwa_torque_sub = rospy.Subscriber(llmu.IIWA_TORQUE_CMD_TOPIC, Float64MultiArray, self._iiwa_torque_cmd_cb)
        self._allegro_torque_sub = rospy.Subscriber(llmu.ALLEGRO_TORQUE_CMD_TOPIC, Float64MultiArray, self._allegro_torque_cmd_cb)
        self._joint_setter_sub = rospy.Subscriber(llmu.JOINT_SETTER_TOPIC, Float64MultiArray, self._joint_setter_cb)
        self._sim_reset_sub = rospy.Subscriber(llmu.SIM_RESET_TOPIC, Bool, self._reset_sim_cb)
        self._iiwa_joint_state_pub = rospy.Publisher(llmu.IIWA_JOINT_STATES_TOPIC, JointState, queue_size=10)
        self._allegro_joint_state_pub = rospy.Publisher(llmu.ALLEGRO_JOINT_STATES_TOPIC, JointState, queue_size=10)
        self._joint_state_id = 0

        # Services
        self._inertia_srv = rospy.Service('inertia', inertia, self._inertia_srv_cb)
        self._obj_pos_srv = rospy.Service('objPos', objPos, self._obj_pos_srv_cb)
        self._obj_cop_pos_srv = rospy.Service('objComPos', objPos, self._obj_com_pos_srv_cb)
        self._obj_mesh_srv = rospy.Service('objMesh', objMesh, self._obj_mesh_srv_cb)

        # Secondary threads
        self._publisher_thread = threading.Thread(target=self._publisher_thread_cb)
        self._publisher_thread.start()

    def __del__(self) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        if self._sim_initialized.locked():
            self._sim_initialized.release()
        if self._sim_running.locked():
            self._sim_running.release()

        if not rospy.is_shutdown():
            rospy.signal_shutdown("User requested shutdown")

    def _setup_MuJoco(self) -> None:

        # Setup simulation
        self._model = mujoco.MjModel.from_xml_path(str(llmu.MUJOCO_MODEL_PATH))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_step(self._model, self._data)
        self._view = viewer.launch_passive(self._model, self._data)

        self._controller = controller_utils.Robot(self._model,
                                                  self._data,
                                                  self._view,
                                                  auto_sync=True,
                                                  obj_names=llmu.SIM_OBJECT_LIST,
                                                  description_folder=str(llmu.MUJOCO_DESCRIPTION_FOLDER),
                                                  kinematics_folder=str(llmu.MUJOCO_KINEMATICS_FOLDER))

        self._view.cam.distance = 3
        self._view.cam.azimuth = 100
        self._view.cam.elevation = -40
        self._view.opt.flags[14] = 1   # Activate contact points visualisation

        self._sim_initialized.acquire()
        self._sim_running.acquire()

    def _iiwa_torque_cmd_cb(self, msg: Float64MultiArray) -> None:

        # Check layout
        if len(msg.layout.dim) != 1 and msg.layout.dim[0].size != len(llmu.IIWA_JOINT_NAMES) and msg.layout.dim[0].stride != 1:
            rospy.logerr("Unexpected layout of torque command")
            return

        if msg.layout.dim[0].label != "joints":
            rospy.logwarn(f"Unexpected torque command label: {msg.layout.dim[0].label}")

        # Apply torque
        self._controller_lock.acquire()
        if self._view.is_running() and self._controller is not None:
            compensed_torque = msg.data + self._controller.C # Compensating gravity and coriolis
            self._controller.send_torque(compensed_torque)  # Works only if IIWA joints are the first 7 joints
            self._latest_iiwa_cmd = np.asarray(msg.data)
            # self._controller.step()  # TODO: make a dedicated thread for this
            # self._view.sync()
        elif self._sim_initialized.locked():
            self._sim_running.release()
        self._controller_lock.release()

    def _reset_sim_cb(self, msg: Bool):
        if msg.data:
            self._controller_lock.acquire()
            mujoco.mj_resetData(self._model, self._data)
            time.sleep(1)
            self._controller.modify_joint(self._controller.q0)
            self._controller.set_zero_speed()
            self._controller_lock.release()


    def _allegro_torque_cmd_cb(self, msg: Float64MultiArray) -> None:

        # Check layout
        if len(msg.layout.dim) != 1 or \
           msg.layout.dim[0].size != len(llmu.ALLEGRO_JOINT_NAMES) or \
           msg.layout.dim[0].stride != 1:
            rospy.logerr("Unexpeted layout of torque command")
            return

        if msg.layout.dim[0].label != "joints":
            rospy.logwarn(f"Unexpected torque command label: {msg.layout.dim[0].label}")

        # Apply torque
        self._controller_lock.acquire()
        if self._view.is_running() and self._controller is not None:
            compensed_torque = msg.data + self._controller.C_allegro  # Compensating gravity and coriolis
            self._controller.send_torque_h(compensed_torque)  # Works only if IIWA joints are the first 7 joints
            self._latest_allegro_cmd = np.asarray(msg.data)
        elif self._sim_initialized.locked():
            self._sim_running.release()
        self._controller_lock.release()

    def _joint_setter_cb(self, msg: Float64MultiArray) -> None:

        # Check joint dimensions
        nb_allegro = len(llmu.ALLEGRO_JOINT_NAMES)
        nb_iiwa = len(llmu.IIWA_JOINT_NAMES)
        nb_total = nb_allegro + nb_iiwa
        if len(msg.layout.dim) != 1 or \
           msg.layout.dim[0].size not in [nb_allegro, nb_iiwa, nb_total] or \
           msg.layout.dim[0].stride != 1:
            rospy.logerr("Unexpected layout of joint setter command")
            return

        # Check label
        if msg.layout.dim[0].label != "joints":
            rospy.logwarn(f"Unexpected joint setter command label: {msg.layout.dim[0].label}")

        # Set joint position
        self._controller_lock.acquire()
        if self._view.is_running() and self._controller is not None:
            self._controller.modify_joint(msg.data)
        elif self._sim_initialized.locked():
            self._sim_running.release()
        self._controller_lock.release()

    def _publisher_thread_cb(self) -> None:

        # Wait of init
        while not self._sim_initialized.locked():
            time.sleep(0.1)
            rospy.loginfo_once("Secondary thread: Waiting for simulation intialization")

        # Main publication loop
        publication_rate = rospy.Rate(self.PUBLISHER_RATE)
        rospy.loginfo_once("Starting main loop on secondary thread")
        while self._sim_initialized.locked() and self._sim_running.locked() and not rospy.is_shutdown():
            self._publish_joint_state()

            # We're gonna use the opportunity to step our simulation at this frequency as well
            self._controller_lock.acquire()
            compensed_iiwa_torque = self._latest_iiwa_cmd + self._controller.C
            compensed_allegro_torque = self._latest_allegro_cmd + self._controller.C_allegro
            self._controller.send_torque(compensed_iiwa_torque)
            self._controller.send_torque_h(compensed_allegro_torque)
            self._controller.step()  # TODO: make a dedicated thread for this
            self._view.sync()
            self._controller_lock.release()

            publication_rate.sleep()

        # Make sure all locks are open
        self.shutdown()

    def _publish_joint_state(self) -> None:

        self._controller_lock.acquire()
        if self._controller is not None and self._view.is_running():

            # IIWA Joint state
            msg_iiwa = JointState()
            msg_iiwa.name = llmu.IIWA_JOINT_NAMES
            msg_iiwa.position = self._controller.q
            msg_iiwa.velocity = self._controller.dq
            msg_iiwa.effort = self._controller.torque

            # Allegro Joint state
            msg_allegro = JointState()
            msg_allegro.name = llmu.ALLEGRO_JOINT_NAMES
            msg_allegro.position = self._controller.qh
            msg_allegro.velocity = self._controller.dqh
            msg_allegro.effort = self._controller.torque_h

            # Header
            header = Header()
            header.seq = self._joint_state_id
            self._joint_state_id += 1
            header.stamp = rospy.Time.now()
            header.frame_id = ''
            msg_iiwa.header = header
            msg_allegro.header = header

            # Publication
            self._iiwa_joint_state_pub.publish(msg_iiwa)
            self._allegro_joint_state_pub.publish(msg_allegro)

        # Error
        elif self._sim_initialized.locked():
            self._sim_running.release()
        self._controller_lock.release()

    def _inertia_srv_cb(self, req: inertiaRequest) -> None:

        # Wait for init
        while not self._sim_initialized.locked():
            time.sleep(0.1)

        # Get inertia
        inertia = np.array([])
        self._controller_lock.acquire()
        if self._controller is not None:
            inertia = self._controller.M_all
        self._controller_lock.release()

        reply = inertiaResponse()
        reply.inertia = inertia.flatten().tolist()
        return reply

    def _obj_pos_srv_cb(self, req: objPosRequest):

        # Wait for initialization
        while not self._sim_initialized.locked():
            time.sleep(0.1)

        # Get object position
        obj_name = req.object_name
        obj_pos = np.array([])
        self._controller_lock.acquire()
        if self._controller is not None:
            try:
                obj_pos = self._controller.obj_pos(obj_name)
            except:
                rospy.logerr(f"No object named {obj_name}")
        self._controller_lock.release()

        # Send obj pos
        reply = objPosResponse()
        reply.object_position = obj_pos.tolist()
        return reply

    def _obj_com_pos_srv_cb(self, req: objPosRequest):

        # Wait for initialization
        while not self._sim_initialized.locked():
            time.sleep(0.1)

        # Get object position
        obj_name = req.object_name
        obj_pos = np.array([])
        self._controller_lock.acquire()
        if self._controller is not None:
            try:
                obj_pos = self._controller.obj_com_pos(obj_name)
            except:
                rospy.logerr(f"No object named {obj_name}")
        self._controller_lock.release()

        # Send obj pos
        reply = objPosResponse()
        reply.object_position = obj_pos.tolist()
        return reply

    def _obj_mesh_srv_cb(self, req: objMeshRequest):

        # Wait for initialization
        while not self._sim_initialized.locked():
            time.sleep(0.1)

        # Get mesh position
        obj_name = req.object_name
        obj_vertices = np.array([])
        self._controller_lock.acquire()
        if self._controller is not None:
            obj_vertices, obj_radii = self._controller.get_object_vertices(obj_name)
        self._controller_lock.release()

        # Send obj pos
        reply = objMeshResponse()
        reply.object_vertices = obj_vertices.flatten().tolist()
        reply.object_radii = obj_radii.flatten().tolist()
        return reply


def main():
    rospy.init_node("llm_simulator")
    sim = LLMSimulator()
    rospy.loginfo("Simulator node initialized")

    # Properly catch signint in our secondary thread
    def clean_up(signum, frame):
        if sim is not None:
            sim.shutdown()
        if not rospy.is_shutdown():
            rospy.signal_shutdown("User requested shutdown")
        sys.exit()
    signal.signal(signal.SIGINT, clean_up)

    # Setup custom cleanup to mage sure we always exit properly
    rospy.spin()
    sim.shutdown()


if __name__ == "__main__":
    main()
