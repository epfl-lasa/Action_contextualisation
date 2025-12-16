from primitives.action_functions import pick, place, drop, approach, throw, reset_controller
from primitives.predicates import at_location, can_grasp, holding, collision_free, timeout, check_motion_health, get_motion_health, can_reach
import contextlib
import roslaunch
import time
import rospy
from std_msgs.msg import Bool
from llm_common import utils as llmu
import time
import typing as tp
import os
import pickle
import numpy as np

SIMU=False
class TaskPlanExecutor:

    def __init__(self, metadata_file: tp.Optional[str] = None) -> None:
        self.function_matching_dict = {'approach': approach,
                                       'place': place,
                                       'pick': pick,
                                       'drop': drop,
                                       'throw': throw}

        self.object_matching_dict = {'crumpled paper ball 1': 'paper_ball_1',
                                     'crumpled paper ball 2': 'paper_ball_2',
                                     'crumpled paper ball 3': 'paper_ball_3',
                                     'crumpled paper ball 4': 'paper_ball_4',
                                     'crumpled paper ball 5': 'paper_ball_5',
                                     'whole apple': 'apple',
                                     'half-eaten apple': 'eaten_apple',
                                     'empty glass 1': 'champagne_2',
                                     'empty glass 2': 'champagne_3',
                                     'glass with yellowish liquid': 'champagne_1',
                                     'large red trash can': 'trash_bin',
                                     "large white sink": 'sink',
                                     'robot': 'robot',
                                     'storage shelf': 'shelf',
                                     "white table": "table",
                                     '': ''}

        self.predicate_matching_dict = {'at_location': at_location,
                                        'can_grasp': can_grasp,
                                        'holding': holding,
                                        'collision_free': collision_free,
                                        'timeout': timeout,
                                        'check_motion_health': check_motion_health,
                                        'can_reach': can_reach}

        # Starting simulator
        print("About to start simulator")
        self._first_run = True
        self._meta_data_file = metadata_file

        if SIMU:
            with contextlib.redirect_stdout(None):
                self.node = roslaunch.core.Node('llm_simulator', 'llm_simulator_node.py')
                self.launch = roslaunch.scriptapi.ROSLaunch()
                self.launch.start()
                self.process = self.launch.launch(self.node)
            print("Please wait as the simulator is starting...")
            time.sleep(10)

        self.inv_obj_matching_dict = {v: k for k, v in self.object_matching_dict.items()}

        self._sim_reset_pub = rospy.Publisher(llmu.SIM_RESET_TOPIC, Bool, queue_size=10)
        self._action_plan_number = 0
        self.full_execution_time = 0
        self.average_score = 0

    def __del__(self) -> None:
        if SIMU:
            self.process.stop()
            print("Please wait as the simulator is shutting down...")
            time.sleep(10)

    def reset_simulator(self):
        msg = Bool()
        msg.data = True
        # if js_lds is not None:
        #     js_lds.send_robot_zero_torque()
        self._sim_reset_pub.publish(msg)
        time.sleep(5)
        # if js_lds is not None:
        #     js_lds.send_robot_zero_torque()
        reset_controller()
        time.sleep(5)
        # if js_lds is not None:
        #     js_lds.send_robot_zero_torque()


    def execute_task_plan(self, action_plan, evaluation_plan):
        self.full_execution_time = 0
        self.average_score = 0
        total_score_history = []

        if self._meta_data_file is not None:
            root, ext = os.path.splitext(self._meta_data_file)
            current_meta_file_name = f"{root}_{self._action_plan_number}{ext}"
        else:
            current_meta_file_name = None
        self._action_plan_number += 1
        evaluation_log = []

        if not self._first_run:
            self.reset_simulator()
        else:
            self._first_run = False
        #time.sleep(1)
        reset_controller()
        action_timings = []
        action_health = []
        for action, evaluation in zip(action_plan, evaluation_plan):
            print(f"New action {action}")
            # ====== Act
            action_id = action[0]
            action_function_name = action[1]
            action_arguments = action[2]

            # Check indices
            if action_id != evaluation[0]:
                raise RuntimeError(f"Plan generated indices do no match\n "
                                   f"action: {action}\n"
                                   f"evaluation: {evaluation}")

            # Approach function
            try:
                action_function = self.function_matching_dict[action_function_name]

            except KeyError:
                print(f"Not implemented action for {action_function_name}")
                continue

            # Account for wired pythonic stuff
            if type(action_arguments) == str:
                action_arguments = [action_arguments]

            object_to_approach = self.object_matching_dict[action_arguments[0]]
            print(action_function_name, object_to_approach)

            # Execute and time action
            print(action_function_name, object_to_approach, action_arguments[1:])
            tic = time.time()
            action_function(object_to_approach, *action_arguments[1:])
            toc = time.time()
            print("Time: ", toc-tic)

            from primitives.action_functions import js_lds
            action_timings.append((action_function_name, object_to_approach, action_arguments, toc-tic))
            action_health.append((action_function_name, object_to_approach, action_arguments, js_lds._condition_numbers, js_lds._proximity_history))
            self.full_execution_time += toc-tic

            if len(js_lds._condition_numbers) and len(js_lds._proximity_history):
                total_score_history.append(np.min(js_lds._condition_numbers) + np.min(js_lds._proximity_history)/js_lds._max_prox)
            else:
                total_score_history.append(-np.inf)
            self.average_score = np.average(total_score_history)

            # ======  Evaluate
            # if action_function_name in ['drop', 'place']:
            #     time.sleep(7)  # Give a bit of time for things in the simulator to settle
            evaluation_id = evaluation[0]
            evaluation_predicates = evaluation[1]
            evaluation_expected = evaluation[2]
            predicate_results = []

            for key, values in evaluation_predicates.items():

                evaluation_function = self.predicate_matching_dict[key]

                corrected_arguments = []

                for val in values:
                    if isinstance(val, str) and val not in ["top", "side"]:
                        corrected_arguments.append(self.object_matching_dict[val])
                    else:
                        corrected_arguments.append(val)

                predicate_results.append(evaluation_function(*corrected_arguments))

            print(f"Predicate_check {evaluation_id}:")
            predicate_result = []

            predicate_ok = True
            for result, expected, predicate in zip(predicate_results, evaluation_expected, evaluation_predicates.keys()):

                # Revert obstacle name for LLM to understand output
                if predicate == 'collision_free':
                    result = self.inv_obj_matching_dict[result]
                    # Fix an issue that happens sometimes
                    if expected == True:
                        expected = ""

                predicate_result.append(result)
                if result == expected:
                    print(f"{predicate}  OK ", result, "==", expected)
                else:
                    predicate_ok = False
                    print(f"{predicate}  NO ", result, "!=", expected)

            evaluation_log.append((evaluation_id, tuple(predicate_result), get_motion_health()))
            print("Iteration over")
            if False:
                # Save meta data
                if current_meta_file_name is not None:
                    with open(current_meta_file_name, 'wb') as f:
                        pickle.dump({'timings': action_timings,
                                     'health': action_health,
                                     'evaluation_log': evaluation_log}, f)

                return False, evaluation_log, evaluation_id

        # Save meta data
        if current_meta_file_name is not None:
            with open(current_meta_file_name, 'wb') as f:
                pickle.dump({'timings': action_timings,
                              'health': action_health,
                              'evaluation_log': evaluation_log}, f)
        return True, evaluation_log, -1

