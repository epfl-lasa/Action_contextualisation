from pathlib import Path
import typing as tp
from controller_base.controller_base import ControllerBase
import numpy as np
import time
import rospy
from scipy.spatial.transform import Rotation
from threading import Lock, Thread

from llm_common import helpers as llmh


class JS_LDS(ControllerBase):
    Kp = np.array([50, 80, 50, 70, 15, 40., 20.,  # IIWA
                   0.4, 0.7, 0.4, 0.4,   # Index
                   0.4, 0.7, 0.4, 0.4,   # Middle
                   0.4, 0.5, 0.3, 0.4,   # Ring
                   0.7, 0.3, 0.45, 0.4]) * 0.15 #  Thumb

    Kd = np.array([5, 6, 4, 4, 5, 3, 1.5,   # IIWA
                   .0, 0., .0, .0,   # Index
                   .0, 0., .0, .0,   # Middle
                   .0, 0., .0, .0,   # Ring
                   .0, .0, .0, .0]) * 0.005 # Thumb

    Ki = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,   # IIWA
                   .0, .0, .0, .0,   # Index
                   .0, .0, .0, .0,   # Middle
                   .0, .0, .0, .0,   # Ring
                   .0, .0, .0, .0]) * 0. # Thumb

    # TODO: Build these arrays better
    JOINT_TOL = np.array([0.01] * 7 + [0.1] * 16)
    SPEED_TOL = np.array([0.01] * 7 + [0.1] * 16)
    CONTROLLER_SLEEP_PRD = 0.001

    TASK_SPACE_TRANS_TOL = 0.001   # m
    TASK_SPACE_ANGLE_TOL = 0.5       # deg
    TASK_SPACE_SPEED_TOL = 0.001   # m/s
    TASK_SPACE_OMEGA_TOL = .1      # deg/s

    def __init__(self,
                 description_folder: tp.Union[str, Path],
                 description_file: tp.Union[str, Path],
                 end_effector_folder: tp.Optional[tp.Union[str, Path]] = None,
                 end_effector_file: tp.Optional[tp.Union[str, Path]] = None) -> None:
        self._cartesian_goal = None     # IIWA Robot frame
        self._joint_goal = None
        self._joint_speed_scale = 1
        super().__init__(description_folder, description_file,
                         end_effector_folder, end_effector_file)
        self._q = self.qz  # Calling parent function, need init before
        self.grasping = False
        self.let_go = False
        self.obj_grasped = ""
        self.timeout = False
        self._qdd_integrator = None
        self._failed_ik = False

        self._controller_prds = []
        # Idle thread
        self._idle_thread_run_signal = Lock()
        self._idle_thread = Thread(target=self.idle_controller)

        # Start Idle thread
        self._idle_thread.start()


    def __del__(self):

        if not self._idle_thread_run_signal.locked():
            self._idle_thread_run_signal.acquire()
        if self._idle_thread.run():
            self._idle_thread.join()

    @property
    def cartesian_goal(self):
        return self._cartesian_goal

    @property
    def hand_position(self):
        q_iiwa = self.x_iiwa
        hand_position = self.fkine(q_iiwa, end=self._robot_ee_link).A
        return hand_position

    @cartesian_goal.setter
    def cartesian_goal(self, goal):
        # Goal is x, y, z, rx, ry, rz, rw (quaternions for rotation)

        if len(goal) == 7:

            # Wait until we are sure we have an accurate joint position
            while not self._new_iiwa_js_event.is_set() and not self._new_allegro_js_event.is_set():
                time.sleep(self.CONTROLLER_SLEEP_PRD)

            # Set IIWA goal (go to point)
            q_iiwa = self.x_iiwa
            goal_se3 = llmh.cartesian_to_se3(goal)
            self._failed_ik = False
            (iiwa_joint_goal, success) = self.compute_ik_iiwa(goal_se3, q_iiwa, verbose=True)
            if success:
                self._cartesian_goal = goal
            else:
                self._failed_ik = True
                return

            # Set allegro goal (remain the same)
            q_allegro = self.qz[self._n_rbt:]
            self._joint_goal = np.concatenate((iiwa_joint_goal, q_allegro))

        else:
            raise ValueError("Cartesian goal is expected in 7 dimensions (x, y, z, rx, ry, rz, rw)")

    @property
    def joint_speed_scale(self):
        return self._joint_speed_scale

    @joint_speed_scale.setter
    def joint_speed_scale(self, scale):
        if 0 < scale <= 1:
            self._joint_speed_scale = scale
        else:
            raise ValueError("Joint speed scale must be between 0 and 1")


    def idle_controller(self) -> None:
        print("Starting Idle thread")

        self._joint_speed_scale = 1
        idle_goal = self.x_iiwa + self.x_allegro
        desired_qdd_prev = np.zeros(self.n)
        start_time = time.time()
        time_prev = start_time
        self._qdd_integrator = np.zeros(self.n)
        idle_thread_rate = rospy.Rate(50)
        while not self._idle_thread_run_signal.locked():
            while not self._new_iiwa_js_event.is_set() and not self._new_allegro_js_event.is_set():
                time.sleep(self.CONTROLLER_SLEEP_PRD)

            # Get robot joint state
            q = np.asarray(self.x_iiwa + self.x_allegro)
            current_qd = self.xd_iiwa + self.xd_allegro
            self._new_iiwa_js_event.clear()
            self._new_allegro_js_event.clear()

            desired_qd = self._compute_lds_speed(np.asarray(q), np.asarray(idle_goal))
            desired_qdd = desired_qd - current_qd
            (torque_cmd, time_prev) = self._compute_torque_from_js(time_prev, desired_qdd, desired_qdd_prev)

            desired_qdd_prev = desired_qdd

            if self.let_go:
                torque_cmd = self._compute_letgo_torques(torque_cmd)
            elif self.grasping: 
                torque_cmd = self._compute_grasping_torques(torque_cmd)                
            else:
                torque_cmd = self._compute_open_torques(torque_cmd)

            # Send torques to robots
            self._send_iiwa_position(idle_goal[:7])
            # self._send_iiwa_torque(torque_cmd[:self._n_rbt])
            self._send_allegro_torque(torque_cmd[self._n_rbt:])
            idle_thread_rate.sleep()

        print("Getting out of idle thread")



    def compute_ik_iiwa(self,
                        goal_se3: np.ndarray,
                        q_iiwa: np.ndarray,
                        verbose: bool = True) -> tp.Tuple[np.ndarray, bool]:

        solution_list = [None] * 3
        success_list = [None] * 3
        (solution_list[0], success_list[0], _, _, _) = self.ik_GN(goal_se3, end=self._robot_ee_link, q0=q_iiwa, ilimit=300, slimit=200)
        (solution_list[1], success_list[1], _, _, _) = self.ik_LM(goal_se3, end=self._robot_ee_link, q0=q_iiwa, ilimit=300, slimit=200)
        (solution_list[2], success_list[2], _, _, _) = self.ik_NR(goal_se3, end=self._robot_ee_link, q0=q_iiwa, ilimit=300, slimit=200)

        # Iterate over all valid solutions to find the one closer to the current position
        iiwa_joint_goal = np.ones_like(solution_list[0]) * np.inf
        solution_id = -1
        for id, (solution, success) in enumerate(zip(solution_list, success_list)):
            if success and np.linalg.norm(q_iiwa[1:]-solution[1:]) < np.linalg.norm(q_iiwa[1:]-iiwa_joint_goal[1:]):
                iiwa_joint_goal = solution
                solution_id = id

        # Report on the solution
        if solution_id >= 0:
            solution_names = ["Gauss-Newton", "Levenberg-Marquardt", "Newton-Raphson"]
            if verbose:
                rospy.loginfo(f"Found best IK solution using {solution_names[solution_id]}")

        return (iiwa_joint_goal, solution_id >= 0)

    def run_controller(self) -> None:

        self.timeout = False

        if self._joint_goal is None:
            raise RuntimeError("Must set a cartesian goal before running the controller")
        
        self._idle_thread_run_signal.acquire()
        self._idle_thread.join()
        desired_qdd_prev = np.zeros(self.n)
        self._qdd_integrator = np.zeros(self.n)
        start_time = time.time()
        time_prev = start_time
        if self.let_go:
            self._joint_goal[7:] = self.qlim[0][7:]
            self._joint_goal[7] = 0
            self._joint_goal[11] = 0
            self._joint_goal[15] = 0
            self._joint_goal[19] = self.qz[19]
        elif self.grasping:
            self._joint_goal[7:] = self.qlim[1][7:]
            self._joint_goal[7] = 0
            self._joint_goal[11] = 0
            self._joint_goal[15] = 0
            self._joint_goal[19] = self.qz[19]
            self._joint_goal[20] = self.qz[20]

        else:
            self._joint_goal[7:] = self.qz[self._n_rbt:]

        print(np.rad2deg(self._joint_goal))
        i = 0
        i_interp = 5
        controller_rate = rospy.Rate(200)
        desired_qd = None
        while True:

            # Check for time out
            if time.time() - start_time > self.TIMEOUT:
                self._send_iiwa_torque(np.zeros(self._n_rbt))
                self.timeout = True
                break

            q = self.x_iiwa + self.x_allegro
            current_qd = self.xd_iiwa + self.xd_allegro
            if i_interp == 5:
                # Compute joint state from LDS
                (_, q, current_qd, desired_qd) = self._compute_current_lds_speed()

                # Compute desired qd
                desired_qd, stop_request = self._compute_desired_qd(q, desired_qd)
                if stop_request:
                    break

                # Compute desired torques
                desired_qdd = desired_qd - current_qd
                # print(np.rad2deg(np.max(np.absolute(desired_qdd[:7]))))
                
                time_pp = time_prev
                (torque_cmd, time_prev) = self._compute_torque_from_js(time_prev, desired_qdd, desired_qdd_prev)
                
                d_time = time_prev - time_pp
                if len(self._controller_prds) >= 8:
                    self._controller_prds[:-1] = self._controller_prds[1:]
                    self._controller_prds[1] = d_time
                else:
                    self._controller_prds.append(d_time)
                desired_qdd_prev = desired_qdd
                i_interp = 0

                # i += 1
                # if i == 2:
                #     print("Cmd:", np.rad2deg(self._joint_goal[:7]-q[:7]))
                #     i = 0
            
            i_interp += 1
            

            iiwa_postion_cmd = self._compute_postion_ctrl(q[:7], desired_qd[:7], current_qd[:7])
            
            
            # if self.grasping:
            #     torque_cmd = self._compute_grasping_torques(torque_cmd)

            # Apply high outwards torque for letting go
            # if self.let_go and 5 < (time.time() - start_time) < 7:
            #     torque_cmd = self._compute_letgo_torques(torque_cmd)

            if self.let_go:
                torque_cmd = self._compute_letgo_torques(torque_cmd)
            elif self.grasping: 
                torque_cmd = self._compute_grasping_torques(torque_cmd)                
            else:
                torque_cmd = self._compute_open_torques(torque_cmd)

            # Send torques to robots
            # self._send_iiwa_torque(torque_cmd[:self._n_rbt])
            # self._send_iiwa_position(self._joint_goal[:7])
            self._send_iiwa_position(iiwa_postion_cmd)
            self._send_allegro_torque(torque_cmd[self._n_rbt:])

            
            # Check task space convergence
            ts_converged = self._check_task_space_convergence(q, current_qd, desired_qd, time.time() - start_time)
            if ts_converged:
                break
            
            controller_rate.sleep()

        self._idle_thread_run_signal.release()
        self._idle_thread = Thread(target=self.idle_controller)
        self._idle_thread.start()


    def send_robot_zero_torque(self):
        self._send_iiwa_torque(np.zeros(self._n_rbt))

    def _check_task_space_convergence(self, q: np.ndarray, qd: np.ndarray, desired_qd: np.ndarray, delta_time: float) -> bool:

        # Compute end effector postion and orientation
        ee_position = self.fkine(q, end=self._robot_ee_link).A
        ee_pos_trans = ee_position[:3, 3]
        ee_rot = Rotation.from_matrix(ee_position[:3, :3])

        # Check end effector position
        goal_trans = self.cartesian_goal[:3]
        ee_trans_conv = np.linalg.norm(ee_pos_trans - goal_trans) <= self.TASK_SPACE_TRANS_TOL

        # Check end effector rotation
        if len(self.cartesian_goal[3:]) > 0:

            # Quaternion representation of the goal
            if len(self.cartesian_goal[3:]) == 4:
                goal_rot = Rotation.from_quat(self.cartesian_goal[3:])

            # Only yaw in deg for when rotation in constrained
            elif len(self.cartesian_goal[3:] == 1):
                xy_rot_proj = ee_rot.as_matrix()[:, :2].transpose()
                xy_rot_proj[:, -1] = 0
                (inverse_yaw_rot, _) = Rotation.align_vectors(np.array([[1, 0, 0], [0, 1, 0]]) * np.linalg.norm(xy_rot_proj), xy_rot_proj)
                goal_rot_no_yaw = inverse_yaw_rot * ee_rot
                goal_rot = Rotation.from_rotvec(np.array([0, 0, self.cartesian_goal[-1]]), degrees=True) * goal_rot_no_yaw
            else:
                raise ValueError("Unknown rotation representation in catesian goal")

            rot_diff = (ee_rot.inv() * goal_rot)
            ee_rot_conv = np.all(np.absolute(np.rad2deg(rot_diff.magnitude())) <= self.TASK_SPACE_ANGLE_TOL)
        else:
            ee_rot_conv = True

        # Compute end effector speed
        jacob = self.jacobe(q, end=self._robot_ee_link)
        ee_speed = jacob @ qd[:self._n_rbt]
        ee_trans_speed = ee_speed[:3]
        ee_omega_deg = np.rad2deg(ee_speed[3:])

        # Check end effector speed convTASK_SPACE_OMEGA_TOL
        ee_speed_conv = np.linalg.norm(ee_trans_speed) < self.TASK_SPACE_SPEED_TOL
        ee_omega_conv = np.all(np.absolute(ee_omega_deg) < self.TASK_SPACE_OMEGA_TOL)

        # Compute end effector desired speed
        jacob = self.jacobe(q, end=self._robot_ee_link)
        ee_desired_speed = jacob @ desired_qd[:self._n_rbt]
        ee_desired_trans_speed = ee_desired_speed[:3]
        ee_desire_omega_deg = np.rad2deg(ee_desired_speed[3:])

        # Check end effector speed conv
        ee_desired_speed_conv = np.linalg.norm(ee_desired_trans_speed) < self.TASK_SPACE_SPEED_TOL
        ee_desired_omega_conv = np.all(np.absolute(ee_desire_omega_deg) < self.TASK_SPACE_OMEGA_TOL)

        goal_reached = ee_trans_conv and ee_rot_conv and ee_speed_conv and ee_omega_conv
        has_converged = ee_speed_conv and ee_omega_conv and ee_desired_speed_conv and ee_desired_omega_conv
        is_locked = delta_time >=10 and ee_speed_conv and ee_omega_conv

        if self.grasping:
            grasping_has_converged = np.all((np.absolute(qd) < self.SPEED_TOL)[self._n_rbt:]) and delta_time >= 5
            #print(goal_reached, has_converged, is_locked, grasping_has_converged)
            goal_reached = goal_reached and grasping_has_converged
            has_converged = has_converged and grasping_has_converged
            is_locked = is_locked and grasping_has_converged

        if self.let_go:
            let_go_has_converged = np.all(np.absolute(q[7:] - self._joint_goal[7:]) <= self.JOINT_TOL[7:]) or delta_time >= 5
            goal_reached = goal_reached and let_go_has_converged
            has_converged = has_converged and let_go_has_converged
            is_locked = is_locked or let_go_has_converged   # Doesn't make sense to compute lock on let go

        if not (self.grasping or self.let_go):
            has_converged = False

        #print(goal_reached, has_converged, is_locked)
        if (goal_reached or has_converged or is_locked):
            print(goal_reached, has_converged, is_locked)
        return goal_reached or has_converged or is_locked

    def _compute_desired_qd(self, q: np.ndarray, desired_qd: np.ndarray) -> np.ndarray:
        return desired_qd, False

    def _compute_lds_speed(self, q: np.ndarray, q_attractor: np.ndarray) -> np.ndarray:

        scaled_speeds = np.zeros_like(q)

        # IIWA speeds
        speeds_iiwa = (q_attractor[:self._n_rbt] - q[:self._n_rbt]) * self._joint_speed_scale * 10
        max_speed_iiwa = np.max(np.abs(speeds_iiwa))
        if max_speed_iiwa > self.IIWA_MAX_ANGLE_SPEED:
            speeds_iiwa = speeds_iiwa / max_speed_iiwa * self.IIWA_MAX_ANGLE_SPEED
        scaled_speeds[:self._n_rbt] = speeds_iiwa
    
        # Allegro speeds
        scaled_speeds[self._n_rbt:] = \
            (q_attractor[self._n_rbt:] - q[self._n_rbt:]) / \
            (self.qlim[1][self._n_rbt:]-self.qlim[0][self._n_rbt:]).max() \
            * self.ALLEGRO_MAX_ANGLE_SPEED * 1

        return scaled_speeds

    def _compute_current_lds_speed(self):
        # Wait to get new joint states
        while not self._new_iiwa_js_event.is_set() and not self._new_allegro_js_event.is_set():
            time.sleep(self.CONTROLLER_SLEEP_PRD)

        # Get robot joint state
        q = np.asarray(self.x_iiwa + self.x_allegro)
        current_qd = self.xd_iiwa + self.xd_allegro
        self._new_iiwa_js_event.clear()
        self._new_allegro_js_event.clear()

        # Check if we have reached the goal
        has_converged = False
        if (np.all(np.absolute(self._joint_goal-q) <= self.JOINT_TOL) and np.all(np.absolute(current_qd) <= self.SPEED_TOL)):
            has_converged = True

        # Compute ds command
        desired_qd = self._compute_lds_speed(q, self._joint_goal)

        return (has_converged, q, current_qd, desired_qd)

    def _compute_torque_from_js(self,
                                time_prev: float,
                                desired_qdd: np.ndarray,
                                desired_qdd_prev: np.ndarray) -> tp.Tuple[np.ndarray, float]:

        inertia = self.get_inertia_from_srv()

        # Handle time variance for Kd
        current_time = time.time()
        delta_time = current_time - time_prev
        self._qdd_integrator += desired_qdd * delta_time
        torque_cmd = inertia @ (desired_qdd * self.Kp +
                                (desired_qdd_prev - desired_qdd) / delta_time * self.Kd +
                                self._qdd_integrator * self.Ki)
        # torque_cmd_inertia = inertia @ desired_qdd
        time_prev = current_time

        # Clip inertial torques
        # torque_cmd_inertia[:self._n_rbt] = np.clip(torque_cmd_inertia[:self._n_rbt],
        #                                            -self.IIWA_MAX_TORQUE,
        #                                            self.IIWA_MAX_TORQUE)
        # torque_cmd_inertia[self._n_rbt:] = np.clip(torque_cmd_inertia[self._n_rbt:],
        #                                            -self.ALLEGRO_MAX_TORQUE,
        #                                            self.ALLEGRO_MAX_TORQUE)

        # Scale down iiwa torque
        torque_cmd_iiwa = torque_cmd[:self._n_rbt]
        total_iiwa_torque = torque_cmd_iiwa # + torque_cmd_inertia[:self._n_rbt]
        if np.any(np.absolute(total_iiwa_torque) > self.IIWA_MAX_TORQUE):
            highest_torque_idx = np.argmax(np.absolute(total_iiwa_torque))
            max_mvt_toque = np.absolute(torque_cmd_iiwa[highest_torque_idx])
            torque_cmd[:self._n_rbt] = torque_cmd_iiwa / max_mvt_toque * self.IIWA_MAX_TORQUE

        # Scale down allegro torque
        torque_cmd_allegro = torque_cmd[self._n_rbt:]
        total_allegro_torque = torque_cmd_allegro # + torque_cmd_inertia[self._n_rbt:]
        if np.any(np.absolute(total_allegro_torque) > self.ALLEGRO_MAX_TORQUE):
            highest_torque_idx = np.argmax(np.absolute(total_allegro_torque))
            max_mvt_toque = np.absolute(torque_cmd_allegro[highest_torque_idx])
            torque_cmd[self._n_rbt:] = torque_cmd_allegro / max_mvt_toque * self.ALLEGRO_MAX_TORQUE

        return torque_cmd, time_prev

    def _compute_grasping_torques(self, desired_torques: np.ndarray) -> np.ndarray:
        grasping_torque = 0.2

        desired_torques[8] = grasping_torque
        desired_torques[9] = grasping_torque
        #desired_torques[10] = grasping_torque

        desired_torques[12] = grasping_torque
        desired_torques[13] = grasping_torque
        #desired_torques[14] = grasping_torque

        desired_torques[16] = grasping_torque
        desired_torques[17] = grasping_torque
        #desired_torques[18] = grasping_torque

        #desired_torques[20] = grasping_torque
        desired_torques[21] = grasping_torque
        desired_torques[22] = grasping_torque

        return desired_torques

    def _compute_letgo_torques(self, desired_torques: np.ndarray) -> np.ndarray:
        grasping_torque = -0.5

        desired_torques[8] = grasping_torque
        desired_torques[9] = grasping_torque
        desired_torques[10] = grasping_torque

        desired_torques[12] = grasping_torque
        desired_torques[13] = grasping_torque
        desired_torques[14] = grasping_torque

        desired_torques[16] = grasping_torque
        desired_torques[17] = grasping_torque
        desired_torques[18] = grasping_torque

        desired_torques[19] = grasping_torque
        desired_torques[20] = grasping_torque
        desired_torques[21] = grasping_torque
        desired_torques[22] = grasping_torque

        return desired_torques
    
    def _compute_open_torques(self, desired_torques: np.ndarray) -> np.ndarray:
        grasping_torque = -0.5

        desired_torques[8] = grasping_torque
        desired_torques[9] = grasping_torque
        desired_torques[10] = grasping_torque

        desired_torques[12] = grasping_torque
        desired_torques[13] = grasping_torque
        desired_torques[14] = grasping_torque

        desired_torques[16] = grasping_torque
        desired_torques[17] = grasping_torque
        desired_torques[18] = grasping_torque

        desired_torques[20] = grasping_torque
        desired_torques[21] = grasping_torque
        desired_torques[22] = grasping_torque

        return desired_torques

    def _compute_postion_ctrl(self, q: np.ndarray, desired_speed: np.ndarray, current_speed: np.ndarray) -> np.ndarray:

        transmision_time = np.average(self._controller_prds)
        if np.rad2deg(np.max(np.abs(self._joint_goal[:7] - q))) <= -np.inf:
            goal_pos = self._joint_goal[:7]
        else:
            delta_speed = desired_speed-current_speed
            goal_pos = q + delta_speed * 1/30
        return goal_pos

