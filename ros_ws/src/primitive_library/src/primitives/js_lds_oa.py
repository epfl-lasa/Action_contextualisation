import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import typing as tp
from pathlib import Path

from neural_jsdf.JSDF import arm_hand_JSDF
from .js_lds_oriented import JS_LDS_ORIENTED
from llm_common import utils as llmu

import torch


class JS_LDS_OA(JS_LDS_ORIENTED):

    AVOIDANCE_FACTOR = 0.001
    COLLISION_PROXIMITY = 0.02
    COLLISION_THRESHOLD = 0.0005

    def __init__(self,
                 description_folder: tp.Union[str, Path],
                 description_file: tp.Union[str, Path],
                 end_effector_folder: tp.Optional[tp.Union[str, Path]] = None,
                 end_effector_file: tp.Optional[tp.Union[str, Path]] = None) -> None:

        super().__init__(description_folder,
                         description_file,
                         end_effector_folder,
                         end_effector_file)
        self._jsdf_model = arm_hand_JSDF(grid_size=[110] * 3)
        self._obstacles = None
        self._obstacles_radii = None
        self._obstacles_names = []
        self.collision_proximity = self.COLLISION_PROXIMITY
        self._in_collision = False
        self._obstacle_collided = ""
        self._obstacle_to_approach = ""
        self._obstacle_ik = False
        self._proximity_history = []
        self._max_prox = -np.inf
        self._condition_numbers = []
        self._time_arr = []

    def reset_collosion_proximity(self):
        self.collision_proximity = self.COLLISION_PROXIMITY

    # TODO: Proper obstacle handling framwork
    def set_obstacles(self,
                      obstacles_points: np.ndarray,
                      obstacle_radii: np.ndarray,
                      obstacle_name: tp.List[str]) -> None:
        self._obstacles = obstacles_points
        self._obstacles_radii = obstacle_radii
        self._obstacles_names = obstacle_name

    def add_obstacle(self,
                     obstacle_point: np.ndarray,
                     obstacle_radius: float,
                     obstacle_name: tp.List[str]) -> None:
        self._obstacles = np.concatenate([self._obstacles, obstacle_point])
        self._obstacles_radii = np.concatenate([self._obstacles_radii, obstacle_radius])
        self._obstacles_names += obstacle_name

    def _compute_desired_qd(self, q: np.ndarray, desired_qd: np.ndarray) -> tp.Tuple[np.ndarray, bool]:

        # Modulate ds for obstacle avoidance
        if self._obstacles is not None and len(self._obstacles) > 0:
            (self._in_collision, self._obstacle_collided, desired_qd) = \
                self._modulate_ds_with_obstacle(q, desired_qd)
            if self._in_collision:
                self._send_iiwa_torque(np.zeros(self._n_rbt))

        jacob = self.jacobe(q, end=self._robot_ee_link)
        self._condition_numbers.append(1/np.linalg.cond(jacob))

        # Apply orientation constraints
        desired_qd = self._project_in_null_rot_space(q, desired_qd)
        return desired_qd, self._in_collision

    def _modulate_ds_with_obstacle(self,
                                   q: np.ndarray,
                                   desired_qd: np.ndarray) -> tp.Tuple[bool, str, np.ndarray]:

        ### New stuff
        self._jsdf_model.arm_hand_robot.set_system_config(q, syn_type=0)
        distances = self._jsdf_model.whole_arm_inference_with_gradient(self._obstacles, return_grad=False)
        adjusted_distances = ((-distances.cpu().detach().numpy()).T - self._obstacles_radii.T).T
        min_distances = np.min(adjusted_distances, axis=1)

        obstacles = self._obstacles
        radii = self._obstacles_radii
        obstacle_names = np.asarray(self._obstacles_names)

        # Update proximity history
        self._max_prox = np.max([self._max_prox, np.max(adjusted_distances)])

        # Disregard collision with grasped object
        if self.grasping and len(self.obj_grasped) > 0:
            valid_obstacles_idx = np.where(np.asarray(obstacle_names) != self.obj_grasped)[0]
            min_distances = min_distances[valid_obstacles_idx]
            obstacles = obstacles[valid_obstacles_idx]
            obstacle_names = obstacle_names[valid_obstacles_idx]
            radii = radii[valid_obstacles_idx]

        # Check for collisions in general
        # If we are realeasing an object we're gonna disregard collision in order to make sure we complete the release
        # The collision will be caught by the next action and make real tests smoother
        is_collided = np.min(min_distances) < self.COLLISION_THRESHOLD and not self.let_go
        obstacle_collided = ""
        if is_collided:
            if len(self.obj_grasped) > 0:
                non_grasped_obj_idx = (np.asarray(obstacle_names) != self.obj_grasped)
                new_min_dst = np.min(min_distances[non_grasped_obj_idx])
                is_collided = np.min(new_min_dst) < self.COLLISION_THRESHOLD
                if is_collided:
                    collided_idx = np.argmin(new_min_dst)
                    potential_collision_names = np.asarray(obstacle_names)[non_grasped_obj_idx]
                    obstacle_collided = potential_collision_names[collided_idx]
            else:
                collided_idx = np.argmin(min_distances)
                potential_collision_names = np.asarray(obstacle_names)
                obstacle_collided = potential_collision_names[collided_idx]
        
        # TODO: ADD comment here
        prox_obs = np.copy(min_distances)
        if len(self.obj_grasped) > 0:
            non_grasped_obj_idx = (np.asarray(obstacle_names) != self.obj_grasped)
            prox_obs = prox_obs[non_grasped_obj_idx]
        self._proximity_history.append(np.min(prox_obs))

        # Postential object to disregard
        if len(self._obstacle_to_approach) > 0:
            valid_obstacles_idx = np.where(np.asarray(obstacle_names) != self._obstacle_to_approach)[0]
            min_distances = min_distances[valid_obstacles_idx]
            obstacles = obstacles[valid_obstacles_idx]
            obstacle_names = obstacle_names[valid_obstacles_idx]
            radii = radii[valid_obstacles_idx]

        close_idx = np.where(min_distances < self.collision_proximity * 2)[0]
        # close_idx = np.where(min_distances < np.inf)[0]
        if len(close_idx) > 0:
            obstacle_names = obstacle_names[close_idx]
            (_, grad) = self._jsdf_model.whole_arm_inference_with_gradient(obstacles[close_idx])
            grad = -grad.cpu().numpy()
            min_distances = min_distances[close_idx]
            # omega_hat = (self.collision_proximity / min_distances) ** 2
            omega_hat = (self.collision_proximity / min_distances) ** 4
            omega_sum = np.min([np.sum(omega_hat), (1/0.001) ** 4])
            omega = omega_hat / np.max([1, omega_sum])
            grad_norm = np.linalg.norm(grad, axis=1)
            weighted_ref_dir = np.sum((grad.T * (omega/grad_norm)).T, axis=0)
            basis_matrix = np.eye(len(weighted_ref_dir))
            basis_matrix[:, 0] = weighted_ref_dir / np.linalg.norm(weighted_ref_dir)
            for i in range(1, len(weighted_ref_dir)):
                ortho_vec = basis_matrix[:, i] - sum(np.dot(basis_matrix[:, i], np.asarray(u).T) * u for u in basis_matrix[:, :i].T)
                ortho_vec = ortho_vec / np.linalg.norm(ortho_vec)
                basis_matrix[:, i] = ortho_vec
            ref_dir_amp = np.linalg.norm(weighted_ref_dir)

            lambda_r = np.cos(np.pi/2 * ref_dir_amp) if ref_dir_amp < 2 else -1
            if np.dot(weighted_ref_dir, desired_qd) < 0 and ref_dir_amp > 1:
                lambda_r *= -1
            lambda_e = 1 + np.sin(np.pi/2 * ref_dir_amp) if ref_dir_amp < 1 else 2 * np.sin(np.pi/(2 * ref_dir_amp))
            eigen_vals = np.eye(len(weighted_ref_dir)) * lambda_e
            eigen_vals[0, 0] = lambda_r

            mod_mat = basis_matrix @ eigen_vals @ basis_matrix.T
            desired_qd = mod_mat @ desired_qd

        output = (is_collided, obstacle_collided, desired_qd)

        return output

    def compute_ik_iiwa(self, goal: np.ndarray,
                        q: np.ndarray,
                        verbose: bool = True) -> np.ndarray:
        print("IN IK")
        if not self._obstacle_ik or True:
            return super().compute_ik_iiwa(goal, q, verbose)

        x_eval = None
        x_eval_cons = None
        diff = None
        distances = None
        grad = None

        def objective_function(x) -> float:
            nonlocal goal
            nonlocal x_eval
            nonlocal diff

            if x_eval is None or np.any(x_eval != x) or diff is None:
                x_eval = np.copy(x)
                current_position = self.fkine(x, end=self._robot_ee_link).A
                diff = np.linalg.inv(current_position) @ goal
            dst_diff = np.sum(np.square(diff[:3, 3]))

            rot_diff = Rotation.from_matrix(diff[:3, :3]).magnitude()

            normalized_dst = (dst_diff + rot_diff / 2.) / 2.

            return normalized_dst

        def constraint_function(x: np.ndarray, i: tp.Optional[int] = None) -> float:
            nonlocal x_eval_cons
            nonlocal distances
            nonlocal grad

            if x_eval_cons is None or \
               np.any(x_eval_cons != x) or \
               distances is None or \
               grad is None \
               or True:
                x_eval_cons = np.copy(x)
                full_q = self.qz
                full_q[:self._n_rbt] = x
                self._jsdf_model.arm_hand_robot.set_system_config(full_q, syn_type=0)
                distances_out = \
                    self._jsdf_model.whole_arm_inference_with_gradient(self._obstacles[i, :], return_grad=False)

                distances = (-distances_out.cpu().detach().numpy().T - self._obstacles_radii.T).T
                # grad = -grad_out.cpu().detach().numpy()

            if i is not None:
                dst = np.min(distances[i, :])
            else:
                dst = np.min(distances)

            return dst

        def constraint_function_grad(x, i: int = 0) -> float:
            nonlocal x_eval_cons
            nonlocal distances
            nonlocal grad

            if x_eval_cons is None or \
               np.any(x_eval_cons != x) or \
               distances is None or \
               grad is None \
               or True:
                x_eval_cons = np.copy(x)
                full_q = self.qz
                full_q[:self._n_rbt] = x
                self._jsdf_model.arm_hand_robot.set_system_config(full_q, syn_type=0)
                (_, grad_out) = \
                    self._jsdf_model.whole_arm_inference_with_gradient(self._obstacles[:, i])
                # distances = (-distances_out.cpu().detach().numpy().T - self._obstacles_radii.T).T
                grad = -grad_out.cpu().detach().numpy()

            return grad[i, :self._n_rbt]

        # Build constraints dict
        constraints = []
        # constraints.append({'type': 'ineq',
        #                     'fun': lambda x: constraint_function(x)})
        print("Building constraint dict")
        for i in range(self._obstacles.shape[0]):
            constraints.append({'type': 'ineq',
                                'fun': lambda x, i=i: constraint_function(x, i),
                                'jac': lambda x, i=i: constraint_function_grad(x, i)})

        # Find the best x0
        print("Doing initial IK")
        q_iiwa = self.x_iiwa
        (sol_gn, success_gn, _, _, _) = self.ik_GN(goal, end=self._robot_ee_link, q0=q_iiwa)
        (sol_lm, success_lm, _, _, _) = self.ik_LM(goal, end=self._robot_ee_link, q0=q_iiwa)
        (sol_nr, success_nr, _, _, _) = self.ik_NR(goal, end=self._robot_ee_link, q0=q_iiwa)

        min_col_gn = np.inf
        min_col_lm = np.inf
        min_col_nr = np.inf

        if not success_gn and not success_lm and not success_nr:
            return None, False
        print("Checking for collisions")
        if success_gn:
            for cons in constraints:
                if (cons['fun'](sol_gn)) < min_col_gn:
                    min_col_gn = cons['fun'](sol_gn)
                    if min_col_gn < 0:
                        break
        if success_lm:
            for cons in constraints:
                if (cons['fun'](sol_lm)) < min_col_lm:
                    min_col_lm = cons['fun'](sol_lm)
                    if min_col_lm < 0:
                        break
        if success_nr:
            for cons in constraints:
                if (cons['fun'](sol_nr)) < min_col_nr:
                    min_col_nr = cons['fun'](sol_nr)
                    if min_col_nr < 0:
                        break

        # Check for valid solutions
        valid_solutions = []
        if min_col_gn > 0:
            valid_solutions.append(sol_gn)
        if min_col_lm > 0:
            valid_solutions.append(sol_lm)
        if min_col_nr > 0:
            valid_solutions.append(sol_nr)

        valid_sol_found = True
        # If no valid solution select the one with the less collisions
        if len(valid_solutions) == 0:
            valid_sol_found = False
            valid_solutions = [sol_gn, sol_lm, sol_nr]
            print("No obstacle free solution... Proceeding")

        angle_dst = np.inf
        for sol in valid_solutions:
            current_angle_dst = np.linalg.norm(np.asarray(sol) - np.asarray(q_iiwa))
            if angle_dst > current_angle_dst:
                x0 = sol
                angle_dst = current_angle_dst

        if not valid_sol_found:
            optim_options = {'disp': True,
                            'maxiter': 30,
                            'ftol': 0.0005}
            bounds = [(q_min, q_max) for q_min, q_max in zip(self.qlim[0, :self._n_rbt],
                                                            self.qlim[1, :self._n_rbt])]

            print("Starting IK optim")
            import time
            tic = time.time()
            ik_res = minimize(objective_function,
                            x0,
                            bounds=bounds,
                            options=optim_options,
                            constraints=constraints,
                            method='SLSQP')
            toc = time.time()
            print(f"IK optim done {toc - tic}")
            ik_res = ik_res.x

            # Additional constaint check.
            # violated_contraints = []
            # collieded_obtacles = []
            # for i, cons in enumerate(constraints):
            #     if cons['fun'](ik_res) < 0:
            #         print(cons['fun'](ik_res))
            #         violated_contraints.append(i)
            #         if self._obstacles_names[i] not in collieded_obtacles:
            #             collieded_obtacles.append(self._obstacles_names[i])
        else:
            ik_res = x0
        # import rospy
        # from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
        # joint_setter_pub = rospy.Publisher(llmu.JOINT_SETTER_TOPIC, Float64MultiArray, queue_size=10)
        # print(len(violated_contraints), set(collieded_obtacles))
        # while True:
        #     msg = Float64MultiArray()
        #     layout = MultiArrayLayout()
        #     layout.dim.append(MultiArrayDimension())
        #     layout.data_offset = 0
        #     layout.dim[0].size = len(self.qz)
        #     layout.dim[0].stride = 1
        #     layout.dim[0].label = "joints"
        #     msg.layout = layout

        #     full_q = self.qz
        #     full_q[:self._n_rbt] = ik_res
        #     msg.data = full_q
        #     joint_setter_pub.publish(msg)

        return ik_res, True

    def compute_motion_health(self) -> float:

        distance_health = np.min(self._proximity_history) / self._max_prox
        condition_health = np.min(self._condition_numbers)

        self._proximity_history = []
        self._condition_numbers = []

        return distance_health + condition_health

    # TODO: Not effective but ok because it's not called often
    def is_holding(self) -> bool:
        holding = False

        if self.obj_grasped != "" and self.grasping:

            jnt = np.asarray(self.x_iiwa + self.x_allegro)
            self._jsdf_model.arm_hand_robot.set_system_config(jnt, syn_type=0)
            distances = self._jsdf_model.whole_arm_inference_with_gradient(self._obstacles, return_grad=False)
            adjusted_distances = ((-distances.cpu().detach().numpy()).T - self._obstacles_radii.T).T

            obj_distances = adjusted_distances[np.asarray(self._obstacles_names) == self.obj_grasped]

            if len(obj_distances) > 0:
                contact_points = obj_distances[obj_distances < self.COLLISION_THRESHOLD]
                if len(contact_points) >= 2:
                    holding = True

        return holding

    def hand_distance_to_obj(self, obj: str) -> float:

        dst = np.inf

        jnt = np.asarray(self.x_iiwa + self.x_allegro)
        self._jsdf_model.arm_hand_robot.set_system_config(jnt, syn_type=0)
        distances = self._jsdf_model.whole_arm_inference_with_gradient(self._obstacles, return_grad=False)
        adjusted_distances = ((-distances.cpu().detach().numpy()).T - self._obstacles_radii.T).T

        obj_distances = adjusted_distances[np.asarray(self._obstacles_names) == obj]

        if len(obj_distances):
            hand_distances = obj_distances[:, 7:]
            dst = np.min(hand_distances)

        return dst