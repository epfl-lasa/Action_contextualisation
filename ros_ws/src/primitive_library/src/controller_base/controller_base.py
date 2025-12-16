from abc import ABC, abstractmethod
import typing as tp
import sys
import pathlib
import numpy as np
from threading import Lock, Event
import rospy
import signal
from math import pi
import pinocchio as pin

from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension

from roboticstoolbox.robot.ERobot import ERobot

import rospy
from sensor_msgs.msg import JointState

from llm_simulator.srv import inertia
from llm_common import utils as llmu


class ControllerBase(ERobot, ABC):

    IIWA_MAX_ANGLE_SPEED = 0.1 * pi  # rad/s
    IIWA_MAX_TORQUE = 300  # Nm

    ALLEGRO_MAX_ANGLE_SPEED = pi  # rad/s
    ALLEGRO_MAX_TORQUE = 2  # Nm

    TIMEOUT = 180  # s

    @staticmethod
    def clean_up(signum, frame):
        if not rospy.is_shutdown():
            rospy.signal_shutdown("User requested shutdown")
        sys.exit()

    def __init__(self,
                 description_folder: tp.Union[str, pathlib.Path],
                 description_file: tp.Union[str, pathlib.Path],
                 end_effector_folder: tp.Optional[tp.Union[str, pathlib.Path]] = None,
                 end_effector_file: tp.Optional[tp.Union[str, pathlib.Path]] = None) -> None:

        # Handle typing and path resolution
        if isinstance(description_folder, str):
            description_folder = pathlib.Path(description_folder)
        if isinstance(description_file, str):
            description_file = pathlib.Path(description_file)
        description_folder = description_folder.resolve()
        description_file = description_file.resolve()

        # Create rtb robot model
        (e_links, name, _, _) = ERobot.URDF_read(description_file, tld=description_folder)
        self._robot_ee_link = None
        self._n_rbt = ERobot.URDF(description_file).n

        # Add end effector
        if end_effector_folder is not None and end_effector_file is not None:
            (ee_links, ee_name, _, _) = ERobot.URDF_read(end_effector_file, tld=end_effector_folder)
            self._n_ee = ERobot.URDF(end_effector_file).n

            # Find the last robot link
            last_robot_link_idx = 0
            i = 0
            while True:
                if i == len(e_links):
                    break
                if e_links[i].parent == e_links[last_robot_link_idx]:
                    last_robot_link_idx = i
                    i = 0
                    continue
                i = i+1

            # Find the first end effector link
            first_ee_link_idx = 0
            while ee_links[first_ee_link_idx].parent:
                first_ee_link_idx = first_ee_link_idx + 1

            self._robot_ee_link = ee_links[first_ee_link_idx].name

            # Link robot and end effector
            ee_links[first_ee_link_idx].parent = e_links[last_robot_link_idx]

            e_links = e_links + ee_links
            name = name + '_' + ee_name

        ERobot.__init__(self, e_links, name=name)
        if self._robot_ee_link is None:
            self._robot_ee_link = self.ee_links[0].name

        ABC.__init__(self)
        self._define_qz()  # define a zero position

        self._grasping = False

        self._previous_js_iiwa_ts = 0
        self._previous_js_allegro_ts = 0

        # IIWA joint state
        self._iiwa_js_lock = Lock()
        self._new_iiwa_js_event = Event()
        self._x_iiwa = np.zeros([len(llmu.IIWA_JOINT_NAMES)]).tolist()
        self._xd_iiwa = np.zeros([len(llmu.IIWA_JOINT_NAMES)]).tolist()
        self._xdd_iiwa = np.zeros([len(llmu.IIWA_JOINT_NAMES)]).tolist()
        self._torque_iiwa = np.zeros([len(llmu.IIWA_JOINT_NAMES)]).tolist()

        # Allegro joint state
        self._allegro_js_lock = Lock()
        self._new_allegro_js_event = Event()
        self._x_allegro = np.zeros([len(llmu.ALLEGRO_JOINT_NAMES)]).tolist()
        self._xd_allegro = np.zeros([len(llmu.ALLEGRO_JOINT_NAMES)]).tolist()
        self._xdd_allegro = np.zeros([len(llmu.ALLEGRO_JOINT_NAMES)]).tolist()
        self._torque_allegro = np.zeros([len(llmu.ALLEGRO_JOINT_NAMES)]).tolist()

        # Init subscriber
        self._iiwa_js_sub = rospy.Subscriber(llmu.IIWA_JOINT_STATES_TOPIC, JointState, self._iiwa_joint_state_cb)
        self._allegro_js_sub = rospy.Subscriber(llmu.ALLEGRO_JOINT_STATES_TOPIC, JointState, self._allegro_joint_state_cb)
        self._iiwa_torque_pub = rospy.Publisher(llmu.IIWA_TORQUE_CMD_TOPIC, Float64MultiArray, queue_size=10)
        self._allegro_torque_pub = rospy.Publisher(llmu.ALLEGRO_TORQUE_CMD_TOPIC, JointState, queue_size=10)
        self._joint_setter_pub = rospy.Publisher(llmu.JOINT_SETTER_TOPIC, Float64MultiArray, queue_size=10)
        self._iiwa_position_pub = rospy.Publisher(llmu.IIWA_POSITION_CMD_TOPIC, Float64MultiArray, queue_size=10)

        # Init service
        rospy.wait_for_service('inertia')
        self._inertia_service = rospy.ServiceProxy('inertia', inertia, persistent=True)

        # Setup a standard clean up function on ctr-c
        signal.signal(signal.SIGINT, ControllerBase.clean_up)

    @abstractmethod
    def run_controller(self) -> None:
        raise NotImplementedError("Must be implemented in derived class")

    @property
    def x_iiwa(self) -> tp.List[float]:
        self._iiwa_js_lock.acquire()
        x = self._x_iiwa.copy()
        self._iiwa_js_lock.release()
        return x

    @x_iiwa.setter
    def x_iiwa(self, x: tp.List[float]):

        if len(x) != len(llmu.IIWA_JOINT_NAMES):
            raise ValueError("Length of x must be equal to the number of joints")

        self._iiwa_js_lock.acquire()
        self._x_iiwa = x
        self._iiwa_js_lock.release()

    @property
    def x_allegro(self) -> tp.List[float]:
        self._allegro_js_lock.acquire()
        x = self._x_allegro.copy()
        self._allegro_js_lock.release()
        return x

    @x_allegro.setter
    def x_allegro(self, x: tp.List[float]):

        if len(x) != len(llmu.ALLEGRO_JOINT_NAMES):
            raise ValueError("Length of x must be equal to the number of joints")

        self._allegro_js_lock.acquire()
        self._x_allegro = x
        self._allegro_js_lock.release()

    @property
    def xd_iiwa(self) -> tp.List[float]:
        self._iiwa_js_lock.acquire()
        xd = self._xd_iiwa.copy()
        self._iiwa_js_lock.release()
        return xd

    @xd_iiwa.setter
    def xd_iiwa(self, xd: tp.List[float]):

        if len(xd) != len(llmu.IIWA_JOINT_NAMES):
            raise ValueError("Length of x must be equal to the number of joints")

        self._iiwa_js_lock.acquire()
        self._xd_iiwa = xd
        self._iiwa_js_lock.release()

    @property
    def xdd_iiwa(self) -> tp.List[float]:
        self._iiwa_js_lock.acquire()
        xdd = self._xdd_iiwa.copy()
        self._iiwa_js_lock.release()
        return xdd

    @property
    def xd_allegro(self) -> tp.List[float]:
        self._allegro_js_lock.acquire()
        xd = self._xd_allegro.copy()
        self._allegro_js_lock.release()
        return xd

    @xd_allegro.setter
    def xd_allegro(self, xd: tp.List[float]):

        if len(xd) != len(llmu.ALLEGRO_JOINT_NAMES):
            raise ValueError("Length of x must be equal to the number of joints")

        self._allegro_js_lock.acquire()
        self._xd_allegro = xd
        self._allegro_js_lock.release()

    @property
    def xdd_allegro(self) -> tp.List[float]:
        self._allegro_js_lock.acquire()
        xdd = self._xdd_allegro.copy()
        self._allegro_js_lock.release()
        return xdd

    @property
    def torque_iiwa(self) -> tp.List[float]:
        self._iiwa_js_lock.acquire()
        torque = self._torque_iiwa.copy()
        self._iiwa_js_lock.release()
        return torque

    @torque_iiwa.setter
    def torque_iiwa(self, torque: tp.List[float]):

        if len(torque) != len(llmu.IIWA_JOINT_NAMES):
            raise ValueError("Length of x must be equal to the number of joints")

        self._iiwa_js_lock.acquire()
        self._torque_iiwa = torque
        self._iiwa_js_lock.release()

    @property
    def torque_allegro(self) -> tp.List[float]:
        self._allegro_js_lock.acquire()
        torque = self._torque_allegro.copy()
        self._allegro_js_lock.release()
        return torque

    @torque_allegro.setter
    def torque_allegro(self, torque: tp.List[float]):

        if len(torque) != len(llmu.ALLEGRO_JOINT_NAMES):
            raise ValueError("Length of x must be equal to the number of joints")

        self._allegro_js_lock.acquire()
        self._torque_allegro = torque
        self._allegro_js_lock.release()

    @property
    def grasping(self) -> bool:
        return self._grasping

    @grasping.setter
    def grasping(self, value: bool):
        self._grasping = value

    @property
    def qz(self) -> np.ndarray:
        """Shortcut to get qz form the robot model
        Returns:
            np.ndarray: q zero
        """
        if 'qz' not in self.configs:
            self._define_qz()
        return self.configs['qz'].copy()

    def get_inertia(self, q: tp.Union[np.ndarray, tp.List[float]]) -> np.ndarray:
        q = np.asarray(q)

        # Generate pinocchio data
        full_q = pin.neutral(self._pin_model)

        # Fill in joint data in the pinocchio neutralconfiguration
        q_idx = 0
        active_q_idx = np.zeros_like(q, dtype=int)
        for name in self._pin_model.names:
            if name not in ['universe', 'root_joint']:
                jnt_idx = self._pin_model.getJointId(name)
                jnt_q_idx = self._pin_model.idx_qs[jnt_idx]
                full_q[jnt_q_idx] = q[q_idx]
                active_q_idx[q_idx] = jnt_idx
                q_idx = q_idx + 1

                # Break if we don't have any more q the rest will be neutral
                if q_idx >= len(q):
                    break

        full_inertia = pin.crba(self._pin_model, self._pin_data, full_q)
        active_inertia = full_inertia[np.ix_(active_q_idx, active_q_idx)]

        return active_inertia

    def get_inertia_from_srv(self) -> np.ndarray:
        try:
            response = self._inertia_service(self.x_iiwa)
        except (rospy.exceptions.TransportTerminated, rospy.service.ServiceException):
            self._inertia_service.close()
            rospy.wait_for_service('inertia')
            self._inertia_service = rospy.ServiceProxy('inertia', inertia, persistent=True)
            response = self._inertia_service(self.x_iiwa)

        return np.asarray(response.inertia).reshape((self.n, self.n))

    def _iiwa_joint_state_cb(self, msg: JointState) -> None:
        current_time = msg.header.stamp.secs + msg.header.stamp.nsecs / 1000000000.
        if current_time > self._previous_js_iiwa_ts:

            # Approximate joint acceleration
            delta_time = current_time-self._previous_js_iiwa_ts
            if self._previous_js_iiwa_ts > 0:
                self._iiwa_js_lock.acquire()
                self._xdd_iiwa = ((msg.velocity - np.asarray(self._xd_iiwa)) / delta_time).tolist()
                self._iiwa_js_lock.release()
            self._previous_js_iiwa_ts = current_time

            self.x_iiwa = list(msg.position)
            self.xd_iiwa = list(msg.velocity)
            self.torque_iiwa = list(msg.effort)
            self._new_iiwa_js_event.set()

    def _allegro_joint_state_cb(self, msg: JointState) -> None:

        current_time = msg.header.stamp.secs + msg.header.stamp.nsecs / 1000000000.
        if current_time > self._previous_js_allegro_ts:

            # Approximate joint acceleration
            delta_time = current_time-self._previous_js_allegro_ts
            if self._previous_js_allegro_ts > 0:
                self._allegro_js_lock.acquire()
                self._xdd_allegro = ((msg.velocity - np.asarray(self._xd_allegro)) / delta_time).tolist()
                self._allegro_js_lock.release()
            self._previous_js_allegro_ts = current_time

            self.x_allegro = list(msg.position)
            self.xd_allegro = list(msg.velocity)
            self.torque_allegro = list(msg.effort)
            self._new_allegro_js_event.set()

    def _send_allegro_torque(self, torques: np.ndarray) -> None:

        allegro_torque_cmd = JointState()

        # Fill out layout information
        # layout = MultiArrayLayout()
        # layout.dim.append(MultiArrayDimension())
        # layout.data_offset = 0
        # layout.dim[0].size = len(torques)
        # layout.dim[0].stride = 1
        # layout.dim[0].label = "joints"
        # allegro_torque_cmd.layout = layout

        # Fill torques
        # grav_torque = self.gravload(self.x_iiwa+self.x_allegro)
        # allegro_torque_cmd.data = torques # + grav_torque[7:]
        # allegro_torque_cmd.position = self.x_allegro
        # allegro_torque_cmd.velocity = self.xd_allegro
        allegro_torque_cmd.effort = torques

        # Send
        self._allegro_torque_pub.publish(allegro_torque_cmd)

    def _send_iiwa_torque(self, torques: np.ndarray) -> None:

        iiwa_torque_cmd = Float64MultiArray()

        # Fill out layout information
        layout = MultiArrayLayout()
        layout.dim.append(MultiArrayDimension())
        layout.data_offset = 0
        layout.dim[0].size = len(torques)
        layout.dim[0].stride = 1
        layout.dim[0].label = "joints"
        iiwa_torque_cmd.layout = layout

        # Fill torques
        # import time
        # tic = time.time()
        # grav_torque = self.gravload(self.x_iiwa+self.x_allegro)
        iiwa_torque_cmd.data = torques # + grav_torque[:7]
        # print(time.time() - tic)

        # Send
        self._iiwa_torque_pub.publish(iiwa_torque_cmd)

    def _send_iiwa_position(self, position: np.ndarray) -> None:

        iiwa_pos_cmd = Float64MultiArray()

        # Fill out layout information
        layout = MultiArrayLayout()
        layout.dim.append(MultiArrayDimension())
        layout.data_offset = 0
        layout.dim[0].size = len(position)
        layout.dim[0].stride = 1
        layout.dim[0].label = "joints"
        iiwa_pos_cmd.layout = layout

        # Fill torques
        # import time
        # tic = time.time()
        # grav_torque = self.gravload(self.x_iiwa+self.x_allegro)
        iiwa_pos_cmd.data = position # + grav_torque[:7]
        # print(time.time() - tic)

        # Send
        self._iiwa_position_pub.publish(iiwa_pos_cmd)

    def _set_robot_joints(self, joints: np.ndarray) -> None:
        joints_msg = Float64MultiArray()

        # Fill out layout information
        layout = MultiArrayLayout()
        layout.dim.append(MultiArrayDimension())
        layout.data_offset = 0
        layout.dim[0].size = len(joints)
        layout.dim[0].stride = 1
        layout.dim[0].label = "joints"
        joints_msg.layout = layout

        # Fill torques
        joints_msg.data = joints

        # Send
        self._joint_setter_pub.publish(joints_msg)

    def _define_qz(self) -> None:
        qz = np.zeros((self.n))

        # Adjust qz if not within joint limits
        for i in range(self.n):

            if qz[i] < self.qlim[0, i] or qz[i] > self.qlim[1, i]:
                qz[i] = self.qlim[0, i] + (self.qlim[1, i] - self.qlim[0, i])/2

        # Fix first thumb joint in a better grasping pose
        qz[self._n_rbt+12] = self.qlim[1][self._n_rbt+12] + 0.5
        qz[self._n_rbt+12] = self.qlim[1][self._n_rbt+12] + 0.5
        qz[self._n_rbt + 1] = self.qlim[1][self._n_rbt + 1] + 0.5
        qz[self._n_rbt+11] = self.qlim[1][self._n_rbt+11]
        #qz[self._n_rbt+14] += 0.3

        self.addconfiguration('qz', qz)
