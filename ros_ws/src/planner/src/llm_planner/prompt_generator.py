# Import task_plan and evaluation_plan from 'param_server'
# from .param_server import task_plan, evaluation_plan

# Sample import for action functions and checking functions libraries
from .action_functions import *
from .checking_functions import *

from copy import deepcopy

import typing as tp



def image_description_gen():
    """Generates the prompt to generate the image description

    Returns:
        str: Prompt
    """
    prompt0 = "Generate a description of the scene and objects in it. Output as a Python string (named 'scene_description')"

    return prompt0


def label_gen():
    """Generates the prompt to generate labels for the objects

    Returns:
        str: Prompt
    """
    prompt0_5 = """
    Identify all the objects and generate relevant labels for them. Objects also includes potential the locations. When there are multiple objects of the same type, differentiate the labels by their respective counts. Output a list named 'object_labels', where each list element is a python string
    """

    return prompt0_5


def modify_labels(labels: tp.List[str]):
    """Adds the robot label to labels

    Args:
        labels (dict[str, str]): Existing labels and their descriptions
    """
    mod_labels = deepcopy(labels)
    mod_labels.append("robot_1")
    return mod_labels


def task_plan_gen(task: str, prompt0: str, objects: tp.List[str], locations: tp.List[str]):
    """Prepares the prompt to fetch the task plan

    Args:
        task (str): The task for the robot
        prompt0 (str): Image desription
        objects (list[str]): Object labels inside the scene
        locations (list[str]): Location labels inside the scene

    Returns:
        str: The composed prompt
    """

    # object_labels = modify_labels(objects) # Adding the robot_1 label to the list
    object_labels = deepcopy(objects) # Adding the robot_1 label to the list

    str_0 = prompt0
    str_1_1 = f"""The list of recognised objects is:

    objects = {object_labels}

    """
    str_1_2 = f"""The list of recognised locations is:

    locations = {locations}
    """

    str_2 = f"""

    There is a robot, labeled 'robot', that can only manipulate ONE object at a time. The robot accepts commands in the form 'action functions' written in python. These action functions, which can be imported from the 'action_functions' library are:

    drop(location: str, speed: float, obstacle_clearance: float) -> None: # Goes to "loaction" and drops the grasped object over it. It is not advised to use the approach function directly before this one.

    approach(object_to_grasp: str, speed: float, obstacle_clearance: float, grasp: str) -> None:  # Moves the robot close to "object_to_grasp" so that the object is in robot's reach

    place(location: str, orientation: float, speed: float, obstacle_clearance: float) -> None: #Positions the "grasped_object" on/at/in the "location" and release the grasp. It is not advised to use the approach function directly before this one.

    pick(object_to_grasp: str, speed: float, obstacle_clearance: float, grasp: str) -> None: # Instructs the robot to pick up the "object_to_grasp", if it is close enough

    The "speed" argument for 'approach', 'pick', 'drop' and 'place' functions, assumes value in [0,1] and regulates how fast the robot moves. The closer the the value is to 1 the faster the robot moves. moving with higher speed is faster but might result in a jerky and less precise motion.

    The "orientation" argument for the 'place' and 'drop' function, regulates how crucial it is for the robot to maintain the original orientation of object that the robot is holding. A value closer to 1, instructs the robot to strictly maintain the orientation, but may result in difficulty to avoid external perturbations or obstacles .

    The "grasp" argument for 'approach' and 'pick' assumes one of the two values {"top", "side"}, where "top" instructs the robot to approach or pick the object from the top and selecting "side" instructs the robot to approach or pick the object from the side.

    The "obstacle_clearance" for 'drop', 'approach', 'place' and 'pick' functions defines how close the robot can get from an object (including the one it is trying to grasp in the pick action) before starting to avoid it. The distance is in meter. Small values allow the robot to get closer to obstacles and usually give a better chance of reaching the object, picking it and holding it. Typically values are between 0.005 and 0.05 although values out of this range are possible.

    The actions described in these functions are the only motions known to the robot. The task for the robot is: "{task}". First explain how you are going to solve the task, why each step is executed and how it makes sense to a human that you do it this way.

    Then, using the actions functions, 'objects' and 'locations', define a task plan as a Python list of tuples (named 'task_plan'), for the robot to follow. The action_functions from the task_plan will be directly run on the robot. Each element of the list is a tuple of the form (action number, action function name string, (arguments)). For each action, use object and task specific arguments.

    The first index of the plan should be 0.

    Take into account that some locations might be outside of the robot reach and you might only be able to use "drop" to put an object there.

    Do not make any assumptions. For the task plan, output a single code block. With each action include as the comment the reasoning behind the action and its parameters.

    Assume the necessary modules containing action functions have already been loaded. Write only a single assignment statement creating the full 'task_plan'

    Do not abbreviate anything in the code you are writing. No matter how long it is, write it in full.
    """

    prompt1 = str_0 + str_1_1 + str_1_2 + str_2

    return prompt1


def eval_plan_gen():
    """Generates the prompt to fetch the evaluation plan
    """
    prompt2 = """
    The robot may not be able to execute an action function, or encounter object collision during execution. Thus, it is required to check for completion of each action function after they have been performed.

    For this, we define some 'checking functions' written in python. These checking functions, which can be imported from the 'checking_functions' library are:

    can_grasp(object_to_grasp: str, grasp: str) -> bool: # Returns True if robot is close-enough to the "object_to_grasp" to securley grasp it with the determined grasp "side" or "top"

    holding() -> bool # Returns True if the robot is holding an object

    at_location(object: str, location: str) -> bool: # Returns True if the "object" is at the "location"

    collision_free() -> str: # If the robot encounters a collision while executing the preceding action, returns the object label string. Otherwise, ''.

    timeout() -> Bool: # Returns True if the preceding action was executed in a timely fashion

    check_motion_health() ->  bool: # Returns True if the robot's motion during the preceding action was safe for its hardware

    can_reach(goal: str, grasp: str) -> bool: # Returns True if it is feasible for the robot to reach the "goal" object or location from the current state from the side determined by the grasp argument "side" or "top". Objects that are out of the workspace will always return false.

    The grasp argument is the same as the one in the "approach" and "pick" functions. It assumes one of the two values {"top", "side"}

    Using the checking_functions, locations and objects, define an evaluation plan (named 'evaluation_plan') to verify the succesful execution of each action. Additionally, for each action verify without fail:
    - collision free
    - timely motion
    - motion health

    Output this plan as a Python list of tuples, where each tuple is of the form (action number int, dictionary with check_function names as keys and a tuple of arguments as value, tuple of expected outputs). Do not assume any other object or location, beyond those in object_labels.

    Each tuple is meant to be checked after the acton with the corresponding number.

    Generate the entire plan. No reasoning, direct output.
    """

    return prompt2


def object_type():
    """Segregates the object_labels into locations and objects"""
    prompt2_0_5 = """
    Segregate the labels present in object_labels into two lists - 'objects' and 'locations'. The 'objects' list should contain the labels of objects which the robot interacts with. The 'locations' list should contains labels of locations
    """

    return prompt2_0_5


def start_state_gen():
    """Generates the prompt to fetch the start state description
    """
    prompt2_1 = """
    Using the relevant checking functions, 'objects' and 'locations',  generate an abstract description of the location of objects for the initial scene.

    Output this description, by defining a Python list of tuples (named 'initial_description'), where each list element is a tuple of the form (check_function, (arguments), expected output).
    """
    return prompt2_1


def goal_state_gen():
    """Generates the prompt to fetch the goal state description
    """
    prompt2_2 = """
    Using the relevant checking functions, 'objects' and 'locations',  generate an abstract description of the location of objects at the end of succesful task completion.

    Output this description, by defining a Python list of tuples (named 'final_description'), where each list element is a tuple of the form (check_function, (arguments), expected output).
    """
    return prompt2_2


def replan_correction_gen(history_log, task_plan, parameter_history):
    """For use on the real robot to handle intermediate failure"""

    failure_index = history_log[-1][0]
    failure_score = history_log[-1][2]
    failure_reason = history_log[-1][-1]
    action_name, action_args = task_plan[int(failure_index)][1], task_plan[int(failure_index)][2]

    str_0 = f"""
    The robot failed during the task_plan's execution. The details are:
    failure index: {failure_index}
    failed action: {action_name}
    current_arguments: {action_args}
    motion score: {failure_score}
    failure reason: {failure_reason}
    """

    # List of parameters of where each list element is of the form ((arguments), score)
    past_failed_parameters = []
    past_succesful_parameters = []

    str_past_failures = ""
    if len(parameter_history['failure'].keys()):
        str_past_failures = "Past failures:\n"
        for action in parameter_history['failure'].keys():
            for object_name in parameter_history['failure'][action].keys():
                failed_parameters = parameter_history['failure'][action][object_name]
                for f_param in failed_parameters:
                    f_arg = list(f_param[0])
                    f_score = f_param[1]
                    str_past_failures += f"FAILURE: {action}: {object_name}{tuple(f_arg)} | score = {f_score}\n"

    str_past_successes = ""
    if len(parameter_history['success'].keys()):
        str_past_successes = "Past successes:\n"
        for action in parameter_history['success'].keys():
            for object_name in parameter_history['success'][action].keys():
                success_parameters = parameter_history['success'][action][object_name]
                for s_param in success_parameters:
                    s_arg = list(s_param[0])
                    s_score = s_param[1]
                    str_past_successes += f"SUCCESS: {action}: {object_name}{tuple(s_arg)} | score = {s_score}\n"


    str_3 = """

    Perform replanning by either using alternative action functions or altering the object interaction sequence.
    Output the code as you did at first by assigning the full task plan to the variable 'task_plan' in a single statement.
    Take into account the parameters I have asked you to change during our conversation if any.
    In the new plan, make sure it is withing plus or minus 5 actions of the original task plan.
    """

    return str_0 + str_past_failures + str_past_successes +  str_3


def replan_after_success_gen(parameter_history, execution_time, average_score):
    """For use on the real robot to handle intermediate failure"""

    str_0 = f"""
        The action plan was successfull with a total time of {execution_time} and an average score of {average_score}.
        Based on the this action plan and the information followng suggest a new action plan that increases the score.
        """

    # List of parameters of where each list element is of the form ((arguments), score)
    past_failed_parameters = []
    past_succesful_parameters = []

    str_past_failures = ""
    if len(parameter_history['failure'].keys()):
        str_past_failures = "Past failures:\n"
        for action in parameter_history['failure'].keys():
            for object_name in parameter_history['failure'][action].keys():
                failed_parameters = parameter_history['failure'][action][object_name]
                for f_param in failed_parameters:
                    f_arg = list(f_param[0])
                    f_score = f_param[1]
                    str_past_failures += f"FAILURE: {action}: {object_name}{tuple(f_arg)} | score = {f_score}\n"

    str_past_successes = ""
    if len(parameter_history['success'].keys()):
        str_past_successes = "Past successes:\n"
        for action in parameter_history['success'].keys():
            for object_name in parameter_history['success'][action].keys():
                success_parameters = parameter_history['success'][action][object_name]
                for s_param in success_parameters:
                    s_arg = list(s_param[0])
                    s_score = s_param[1]
                    str_past_successes += f"SUCCESS: {action}: {object_name}{tuple(s_arg)} | score = {s_score}\n"


    str_3 = """

    Perform replanning by either using alternative action functions or altering the object interaction sequence.
    Output the code as you did at first by assigning the full task plan to the variable 'task_plan' in a single statement.
    Take into account the parameters I have asked you to change during our conversation if any.
    In the new plan, make sure it is withing plus or minus 5 actions of the original task plan.
    """

    return str_0 + str_past_failures + str_past_successes +  str_3


def retune_after_success_gen(parameter_history, execution_time, average_score):

    # List of parameters of where each list element is of the form ((arguments), score)
    past_failed_parameters = []
    past_succesful_parameters = []

    str_0 = f"""
        The action plan was successfull with a total time of {execution_time} and an average score of {average_score}.
        Please suggest the retuning of a single action of your choice to impove the score.
        """

    str_past_failures = ""
    if len(parameter_history['failure'].keys()):
        str_past_failures = "Past failures:\n"
        for action_name in parameter_history['failure'].keys():
            for object_name in parameter_history['failure'][action_name].keys():
                failed_parameters = parameter_history['failure'][action_name][object_name]
                for f_param in failed_parameters:
                    f_arg = list(f_param[0])
                    f_score = f_param[1]
                    str_past_failures += f"        FAILURE: {action_name}: {object_name} {tuple(f_arg)} | score = {f_score}\n"

    str_past_successes = ""
    if len(parameter_history['success'].keys()):
        str_past_successes = "Past successes of this action:\n"
        for action_name in parameter_history['success'].keys():
            for object_name in parameter_history['success'][action_name].keys():
                success_parameters = parameter_history['success'][action_name][object_name]
                for s_param in success_parameters:
                    s_arg = list(s_param[0])
                    s_score = s_param[1]
                    str_past_successes += f"        SUCCESS: {action_name}: {object_name} {tuple(s_arg)} | score = {s_score}\n"

    str_3 = f"""

    The score indicates the suitability of a combination. It is a combination of the minimum distane to objects during a movement and how far the robot is from a singluarits.

    A higher score is better. First explain how the changes your are making will improve the chances of success of the task.

    Then alter the arguments of on single action the index of your choice in task_plan, to increase the score and the total time of execution.

    The plan should still be viable-

    Do not change the action. Make in-place change in task_plan.

    """

    return str_0 + str_past_failures + str_past_successes + str_3


def retune_gen(history_log, parameter_history, task_plan):
    """Generates the prompt to perform parameter tuning for the failed action"""

    failure_index = history_log[-1][0]
    failure_score = history_log[-1][2]
    failure_reason = history_log[-1][-1]
    action_name, action_args = task_plan[int(failure_index)][1], task_plan[int(failure_index)][2]
    object_label = action_args[0]

    # List of parameters of where each list element is of the form ((arguments), score)
    past_failed_parameters = []
    past_succesful_parameters = []
    str_0 = f"""
    The robot failed during the task_plan's execution. The details are:
    failure index: {failure_index}
    failed action: {action_name}
    current_arguments: {action_args}
    motion score: {failure_score}
    failure reason: {failure_reason}
    """

    str_past_failures = ""
    if action_name in parameter_history['failure'].keys():
        str_past_failures = "Past failures of this action:\n"
        for object_name in parameter_history['failure'][action_name].keys():
            failed_parameters = parameter_history['failure'][action_name][object_name]
            for f_param in failed_parameters:
                f_arg = list(f_param[0])
                f_score = f_param[1]
                str_past_failures += f"        FAILURE: {action_name}: {object_name} {tuple(f_arg)} | score = {f_score}\n"

    str_past_successes = ""
    if action_name in parameter_history['success'].keys():
        str_past_successes = "Past successes of this action:\n"
        for object_name in parameter_history['success'][action_name].keys():
            success_parameters = parameter_history['success'][action_name][object_name]
            for s_param in success_parameters:
                s_arg = list(s_param[0])
                s_score = s_param[1]
                str_past_successes += f"        SUCCESS: {action_name}: {object_name} {tuple(s_arg)} | score = {s_score}\n"

    str_3 = f"""

    The score indicates the suitability of a combination. A higher score is better. First explain how the changes your are making will improve the chances of success of the task.

    Then alter the arguments of the failed action at index {failure_index} in task_plan, to overcome the failure.

    Do not use other action. Make in-place change in task_plan.

    """

    return str_0 + str_past_failures + str_past_successes + str_3

