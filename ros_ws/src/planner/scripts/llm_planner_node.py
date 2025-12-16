from llm_planner.chatbots import GptChatBot
import llm_planner.prompt_generator as pg
from llm_planner.utils import create_experiment_log, get_log_folder
from llm_planner.problem_interpreter import ProblemInterpreter
import llm_planner.helper_functions as helper
import pickle
from primitives.execute_task_plan import TaskPlanExecutor
import rospy
from copy import deepcopy
import typing as tp
import os
import numpy as np

action_plan_ask = """
The image shows an indoor scene with a white table at the center, scattered with various objects. On the table, there are crumpled paper balls, a whole apple, a half-eaten apple, and two different types of glasses, one of which appear to be empty and one partially filled with a yellowish liquid. To the left, there is a large white sink with a faucet against a tiled wall. A storage shelf is also kept near the table.In the background, there are two doors, one closed and one ajar, and a folded white chair next to the table. On the right side, a large red trash can is visible, filled with discarded items including colorful plastic wrappers. The room has a clinical or institutional feel, possibly a break room or a workshop space.The list of recognised objects is:

    objects = ['white table', 'crumpled paper ball 1', 'crumpled paper ball 2', 'whole apple', 'half-eaten apple', 'empty glass 1', 'glass with yellowish liquid', 'large red trash can']

    The list of recognised locations is:

    locations = ['large white sink', 'faucet', 'tiled wall', 'closed door', 'ajar door', 'folded white chair', 'clinical room', 'storage shelf']


    There is a robot, labeled 'robot', that can only manipulate ONE object at a time. The robot accepts commands in the form 'action functions' written in python. These action functions, which can be imported from the 'action_functions' library are:

    drop(location: str, speed: float, obstacle_clearance: float) -> None: # Goes to "loaction" and drops the grasped object over it. It is not advised to use the approach function directly before this one.

    approach(object_to_grasp: str, speed: float, obstacle_clearance: float, grasp: str) -> None:  # Moves the robot close to "object_to_grasp" so that the object is in robot's reach

    place(location: str, orientation: float, speed: float, obstacle_clearance: float) -> None: #Positions the "grasped_object" on/at/in the "location" and release the grasp. It is not advised to use the approach function directly before this one.

    pick(object_to_grasp: str, speed: float, obstacle_clearance: float, grasp: str) -> None: # Instructs the robot to pick up the "object_to_grasp", if it is close enough

    The "speed" argument for 'approach', 'pick', 'drop' and 'place' functions, assumes value in [0,1] and regulates how fast the robot moves. The closer the the value is to 1 the faster the robot moves. moving with higher speed is faster but might result in a jerky and less precise motion.

    The "orientation" argument for the 'place' and 'drop' function, regulates how crucial it is for the robot to maintain the original orientation of object that the robot is holding. A value closer to 1, instructs the robot to strictly maintain the orientation, but may result in difficulty to avoid external perturbations or obstacles .

    The "grasp" argument for 'approach' and 'pick' assumes one of the two values ('top', 'side'), where "top" instructs the robot to approach or pick the object from the top and selecting "side" instructs the robot to approach or pick the object from the side.

    The "obstacle_clearance" for 'drop', 'approach', 'place' and 'pick' functions defines how close the robot can get from an object (including the one it is trying to grasp in the pick action) before starting to avoid it. The distance is in meter. Small values allow the robot to get closer to obstacles and usually give a better chance of reaching the object, picking it and holding it. Typically values are between 0.01 and 0.1 although values out of this range are possible.

    The actions described in these functions are the only motions known to the robot. The task for the robot is: "clear the table efficiently". First explain how you are going to solve the task, why each step is executed and how it makes sense to a human that you do it this way.

    Then, using the actions functions, 'objects' and 'locations', define a task plan as a Python list of tuples (named 'task_plan'), for the robot to follow. The action_functions from the task_plan will be directly run on the robot. Each element of the list is a tuple of the form (action number, action function name string, (arguments)). For each action, use object and task specific arguments.

    The first index of the plan should be 0.

    Take into account that some locations might be outside of the robot reach and you might only be able to use "drop" to put an object there.

    Do not make any assumptions. For the task plan, output a single code block. With each action include as the comment the reasoning behind the action and its parameters.

    Assume the necessary modules containing action functions have already been loaded. Write only a single assignment statement creating the full 'task_plan'

    Do not abbreviate anything in the code you are writing. No matter how long it is, write it in full.
"""

action_plan_answer = """
 To efficiently clear the table, we should follow a plan that minimizes the amount of robot movement and sorts objects by type to facilitate future actions (like recycling or disposal). The steps should be:

1. **Pick up crumpled paper balls**: They are likely lightweight and can be picked and placed into the trash can together if the robot can hold multiple after each approach without moving the base. Otherwise, do one at a time.
2. **Pick up the apple remains**: The half-eaten apple should be discarded as waste, while the whole apple might be placed elsewhere like a storage shelf if it's to be kept.
3. **Handle the glasses**: The empty glass should be placed in the sink, assuming it needs to be washed, and the glass with liquid could be poured out, then placed in the sink.

While choosing grasp types, keep in mind that items like paper balls and apples can be picked from the top, whereas glasses may require a side grasp. Use moderate speed to ensure precision and minimize collision risks. Here is the detailed task plan represented as a Python list of tuples named `task_plan`:

```python
task_plan = [
    # Pick up the first crumpled paper ball and drop it in the large red trash can
    (0, 'approach', ('crumpled paper ball 1', 0.5, 0.05, 'top')),
    (1, 'pick', ('crumpled paper ball 1', 0.5, 0.05, 'top')),
    # assuming the robot can move to the trash can without dropping the object
    (2, 'drop', ('large red trash can', 0.5, 0.05)),

    # Pick up the second crumpled paper ball and drop it in the trash can
    (3, 'approach', ('crumpled paper ball 2', 0.5, 0.05, 'top')),
    (4, 'pick', ('crumpled paper ball 2', 0.5, 0.05, 'top')),
    (5, 'drop', ('large red trash can', 0.5, 0.05)),

    # Pick up the half-eaten apple and drop it in the trash can
    (6, 'approach', ('half-eaten apple', 0.5, 0.05, 'top')),
    (7, 'pick', ('half-eaten apple', 0.5, 0.05, 'top')),
    (8, 'drop', ('large red trash can', 0.5, 0.05)),

    # Pick up the whole apple and place it on the storage shelf
    (9, 'approach', ('whole apple', 0.5, 0.05, 'top')),
    (10, 'pick', ('whole apple', 0.5, 0.05, 'top')),
    (11, 'place', ('storage shelf', 0.5, 0.5, 0.05)),

    # Approach the empty glass and pick it from the side, then place it in the sink
    (12, 'approach', ('empty glass 1', 0.5, 0.05, 'side')),
    (13, 'pick', ('empty glass 1', 0.5, 0.05, 'side')),
    (14, 'place', ('large white sink', 0.5, 0.5, 0.05)),

    # Approach the glass with yellowish liquid, pick it, and pour it out before placing in the sink
    # Assuming pouring action is a part of the 'place' function before releasing the grasp
    (15, 'approach', ('glass with yellowish liquid', 0.5, 0.05, 'side')),
    (16, 'pick', ('glass with yellowish liquid', 0.5, 0.05, 'side')),
    (17, 'place', ('large white sink', 0.5, 0.5, 0.05))
]
```

This task plan assumes that:
- The 'pick' function allows the robot to hold the object until the 'drop' or 'place' function is executed.
- The 'place' function can also involve actions like pouring out the contents of a glass before placing it.
- Objects like apples are placed on a storage shelf if they are to be kept, while waste is put in the trash can.
- After executing a 'pick', the robot does not necessarily have to execute an 'approach' again before 'drop' or 'place' if the object is already held.

The task plan aims for both efficiency in motion and proper handling of the objects in accordance with their disposability or reusability.

"""

eval_plan_ask = """
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

eval_plan_answer = """
```python
evaluation_plan = [
    # Check after approaching the first crumpled paper ball
    (0, {'can_grasp': ('crumpled paper ball 1', 'top'),
         'collision_free': (),
         'timeout': (),
         'check_motion_health': (),
         'can_reach': ('crumpled paper ball 1', 'top')},
         (True, '', True, True, True)),

    # Check after picking the first crumpled paper ball
    (1, {'holding': (),
         'collision_free': (),
         'timeout': (),
         'check_motion_health': ()},
         (True, '', True, True)),

    # Check after dropping the first crumpled paper ball into the trash can
    (2, {'at_location': ('crumpled paper ball 1', 'large red trash can'),
         'collision_free': (),
         'timeout': (),
         'check_motion_health': ()},
         (True, '', True, True)),

    # Check after approaching the second crumpled paper ball
    (3, {'can_grasp': ('crumpled paper ball 2', 'top'),
         'collision_free': (),
         'timeout': (),
         'check_motion_health': (),
         'can_reach': ('crumpled paper ball 2', 'top')},
         (True, '', True, True, True)),

    # Check after picking the second crumpled paper ball
    (4, {'holding': (),
         'collision_free': (),
         'timeout': (),
         'check_motion_health': ()},
         (True, '', True, True)),

    # Check after dropping the second crumpled paper ball into the trash can
    (5, {'at_location': ('crumpled paper ball 2', 'large red trash can'),
         'collision_free': (),
         'timeout': (),
         'check_motion_health': ()},
         (True, '', True, True)),

    # Check after approaching the half-eaten apple
    (6, {'can_grasp': ('half-eaten apple', 'top'),
         'collision_free': (),
         'timeout': (),
         'check_motion_health': (),
         'can_reach': ('half-eaten apple', 'top')},
         (True, '', True, True, True)),

    # Check after picking the half-eaten apple
    (7, {'holding': (),
         'collision_free': (),
         'timeout': (),
         'check_motion_health': ()},
         (True, '', True, True)),

    # Check after dropping the half-eaten apple into the trash can
    (8, {'at_location': ('half-eaten apple', 'large red trash can'),
         'collision_free': (),
         'timeout': (),
         'check_motion_health': ()},
         (True, '', True, True)),

    # Check after approaching the whole apple
    (9, {'can_grasp': ('whole apple', 'top'),
         'collision_free': (),
         'timeout': (),
         'check_motion_health': (),
         'can_reach': ('whole apple', 'top')},
         (True, '', True, True, True)),

    # Check after picking the whole apple
    (10, {'holding': (),
          'collision_free': (),
          'timeout': (),
          'check_motion_health': ()},
          (True, '', True, True)),

    # Check after placing the whole apple on the storage shelf
    (11, {'at_location': ('whole apple', 'storage shelf'),
          'collision_free': (),
          'timeout': (),
          'check_motion_health': ()},
          (True, '', True, True)),

    # Check after approaching the empty glass
    (12, {'can_grasp': ('empty glass 1', 'side'),
          'collision_free': (),
          'timeout': (),
          'check_motion_health': (),
          'can_reach': ('empty glass 1', 'side')},
          (True, '', True, True, True)),

    # Check after picking the empty glass
    (13, {'holding': (),
          'collision_free': (),
          'timeout': (),
          'check_motion_health': ()},
          (True, '', True, True)),

    # Check after placing the empty glass in the sink
    (14, {'at_location': ('empty glass 1', 'large white sink'),
          'collision_free': (),
          'timeout': (),
          'check_motion_health': ()},
          (True, '', True, True)),

    # Check after approaching the glass with yellowish liquid
    (15, {'can_grasp': ('glass with yellowish liquid', 'side'),
          'collision_free': (),
          'timeout': (),
          'check_motion_health': (),
          'can_reach': ('glass with yellowish liquid', 'side')},
          (True, '', True, True, True)),

    # Check after picking the glass with yellowish liquid
    (16, {'holding': (),
          'collision_free': (),
          'timeout': (),
          'check_motion_health': ()},
          (True, '', True, True)),

    # Check after placing the glass with yellowish liquid in the sink
    (17, {'at_location': ('glass with yellowish liquid', 'large white sink'),
          'collision_free': (),
          'timeout': (),
          'check_motion_health': ()},
          (True, '', True, True))
]
```

"""

warm_start_history = [action_plan_ask, action_plan_answer, eval_plan_ask, eval_plan_answer]


scene_description = (
    "The image shows an indoor scene with a white table at the center, scattered with various objects. "
    "On the table, there are crumpled paper balls, a whole apple, a half-eaten apple, and two different types of glasses, "
    "one of which appear to be empty and one partially filled with a yellowish liquid. "
    "To the left, there is a large white sink with a faucet against a tiled wall. A storage shelf is also kept near the table."
    "In the background, there are two doors, one closed and one ajar, and a folded white chair next to the table. "
    "On the right side, a large red trash can is visible, filled with discarded items including colorful plastic wrappers. "
    "The room has a clinical or institutional feel, possibly a break room or a workshop space."
)


# Labels that are considered objects which a robot may interact with
objects = [
    "white table",
    "crumpled paper ball 1",
    "crumpled paper ball 2",
    #"crumpled paper ball 3",
    #"crumpled paper ball 4",
    #"crumpled paper ball 5",
    "whole apple",
    "half-eaten apple",
    "empty glass 1",
    #"empty glass 2",
    "glass with yellowish liquid",
    "large red trash can",
    # "discarded plastic wrapper 1",
    # "discarded plastic wrapper 2",
    # "discarded plastic wrapper 3",
]

# Labels that are considered locations
locations = [
    "large white sink",
    "faucet",
    "tiled wall",
    "closed door",
    "ajar door",
    "folded white chair",
    "clinical room",
    "storage shelf"
]

task = "clear the table efficiently"

iter = 1
retuned_iter = 1
retune_count = 5
replan_count = 5

log_folder = ""


def extract_plan(full_answer: str) -> tp.Tuple[bool, str, tp.List]:

    error_message = ""
    plan = []

    raw_code = ProblemInterpreter.extract_code_from_md(full_answer, ['python', 'python3', 'python2', ''])

    # Check number of code blocks
    if len(raw_code) == 0:
        error_message = "I expected to find python code in your answer but didn't get any. Please regenerate the plan"
    elif len(raw_code) > 1:
        error_message = "I expected to find only one python code in your answer but found more. Please regenerate your answer with a single python code block"

    # Check if the code can run without error
    if len(error_message) == 0:
        try:
            exec(raw_code[0])
        except Exception as e:
            error_message = f"I tried running your code it output the following error {str(e)} could you please correct it"

    # Check if the code has a correct assignment
    if len(error_message) == 0:
        operands = raw_code[0].split("=")
        if len(operands) == 1:
            error_message = "I expected to find an assignment in your code but didn't get any. Please regenerate your answer with a single assignment"
        elif len(operands) > 2:
            error_message = "I expected ot find a single assignment in your code but found multiple. Please regenerate your answer with a single assignment"

    # Assign plan
    if len(error_message) == 0:
        try:
            plan = eval(operands[1])
        except Exception as e:
            error_message = f"I tried running your code it output the following error {str(e)} could you please correct it"

    # Check that plan starts at 0, is in order and has no gaps
    if len(error_message) == 0:
        plan_indices = [task[0] for task in plan]
        if plan_indices[0] != 0:
            error_message = "The first index of the plan does not start at 0. Please regenerate your plan"
        elif not all(plan_indices[i] < plan_indices[i + 1] for i in range(len(plan_indices) - 1)):
            error_message = "The plan is not in order. Please regenerate your plan"
        elif any(plan_indices[i + 1] - plan_indices[i] != 1 for i in range(len(plan_indices) - 1)):
            error_message = "The plan indices are not consecutive. Please regenerate your plan"

    return len(error_message) == 0, error_message, plan


def extract_retune_action(full_retune_answer: str,
                          previous_task_plan: tp.List) -> tp.Tuple[bool, str, tp.List]:

    error_message = ""
    task_plan = deepcopy(previous_task_plan)

    raw_code = ProblemInterpreter.extract_code_from_md(full_retune_answer, ['python', 'python3', 'python2', ''])

    # Check number of code blocks
    if len(raw_code) == 0:
       error_message = "I expected to find python code in your answer but didn't get any. Please regenerate the plane"
    elif len(raw_code) > 1:
        error_message = "I expected to find only one python code in your answer but found more. Please regenerate your answer with a single python code block"

    # Check if the code can run without error
    if len(error_message) == 0:
        try:
            exec(raw_code[0])
        except Exception as e:
            error_message = f"I tried running your code it output the following error {str(e)} could you please correct it"

    # Check if task plan was modfied
    if len(error_message) == 0:
        task_plan_changed = False
        for task, old_task in zip(task_plan, previous_task_plan):
            if not (task == old_task):
                task_plan_changed = True
                break

        if not task_plan_changed:
            error_message = "After running your code the plan 'task_plan' did not change. You are expected to provide on or more python lines that have the effect of updating the task plan according to the instructions you were give. Pleas regenerate your answer"

    # Check that plan starts at 0, is in order and has no gaps
    if len(error_message) == 0:
        plan_indices = [task[0] for task in task_plan]

        for index in plan_indices:
            if type(index) != int:
                error_message = "After running your code, the plan contains non-integer indices. Please regenerate your answer"
                break

        if len(error_message) == 0:
            if plan_indices[0] != 0:
                error_message = "After running your code, the first index of the plan does not start at 0. Please regenerate your answer"
            elif not all(plan_indices[i] < plan_indices[i + 1] for i in range(len(plan_indices) - 1)):
                error_message = "After running your code, the plan is not in order. Please regenerate your answer"
            elif any(plan_indices[i + 1] - plan_indices[i] != 1 for i in range(len(plan_indices) - 1)):
                error_message = "After running your code, the plan indices are not consecutive. Please regenerate your answer"

    # Reset task plan if error
    if len(error_message) != 0:
        task_plan = deepcopy(previous_task_plan)

    return len(error_message) == 0, error_message, task_plan


def ask_for_task_plan(llm_bot: GptChatBot,
                      task: str,
                      scene_description: str,
                      objects: tp.List[str],
                      locations=tp.List[str]) -> tp.List:

    task_plan_prompt = pg.task_plan_gen(task=task,
                                         prompt0=scene_description,
                                         objects=objects,
                                         locations=locations)
    print("Asking: ", task_plan_prompt)
    task_plan_answer = llm_bot.ask(task_plan_prompt, show_output=True)

    for _ in range(3):
        plan_is_correct, error_message, task_plan = extract_plan(task_plan_answer)

        if plan_is_correct:
            break
        else:
            print("Asking: ", error_message)
            task_plan_answer = llm_bot.ask(error_message, show_output=True)

    if not plan_is_correct:
        raise Exception(f"Unable to generate valid task plan from LLM. The latest error message was {error_message}")

    with open(os.path.join(log_folder, 'task_plan' + str(iter) + '.pkl'), 'wb') as file:
        pickle.dump(f"task_plan = {str(task_plan)}", file)

    return task_plan


def ask_for_evaluation_plan(llm_bot: GptChatBot):

    eval_plan_prompt = pg.eval_plan_gen()
    print("Asking: ", eval_plan_prompt)
    eval_plan_answer = llm_bot.ask(eval_plan_prompt, show_output=True)

    for _ in range(5):
        plan_is_correct, error_message, eval_plan = extract_plan(eval_plan_answer)

        if plan_is_correct:
            break
        else:
            print("Asking: ", error_message)
            eval_plan_answer = llm_bot.ask(error_message, show_output=True)

    plan_is_correct, error_message, eval_plan = extract_plan(eval_plan_answer)
    if not plan_is_correct:
        raise Exception(f"Unable to generate valid evaluation plan from LLM. The latest error message was {error_message}")

    with open(os.path.join(log_folder, 'evaluation_plan' + str(iter) + '.pkl'), 'wb') as file:
        pickle.dump(f"task_plan = {str(eval_plan)}", file)

    return eval_plan


def ask_for_action_plan_retune(llm_bot: GptChatBot,
                               history_log,
                               parameter_history,
                               previous_task_plan: tp.List,
                               retune_idx: int) -> tp.List:

    retune_prompt = pg.retune_gen(history_log, parameter_history, previous_task_plan)
    print(f"Asking: {retune_prompt}")
    retune_answer = llm_bot.ask(retune_prompt, show_output=True)

    for _ in range(5):
        plan_is_correct, error_message, retuned_plan = extract_retune_action(retune_answer, previous_task_plan)

        if plan_is_correct:
            break
        else:
            print("Asking: ", error_message)
            retune_answer = llm_bot.ask(error_message, show_output=True)
    plan_is_correct, error_message, retuned_plan = extract_retune_action(retune_answer, previous_task_plan)

    if not plan_is_correct:
        raise Exception(f"Unable to generate valid retune plan from LLM. The latest error message was {error_message}")
    global retuned_iter
    with open(os.path.join(log_folder, 'retuned_plan' + str(iter) + "_" + str(retuned_iter) + '.pkl'), 'wb') as file:
        pickle.dump(f"task_plan = {str(retuned_plan)}", file)


    retuned_iter += 1

    return retuned_plan


def ask_for_action_replanned(llm_bot: GptChatBot, history_log, previous_task_plan, parameter_history):

    new_plan_prompt = pg.replan_correction_gen(history_log, previous_task_plan, parameter_history)
    print("Asking: ", new_plan_prompt)
    new_plan_answer = llm_bot.ask(new_plan_prompt, show_output=True)

    for _ in range(3):
        plan_is_correct, error_message, task_plan = extract_plan(new_plan_answer)

        if plan_is_correct:
            # Check that the has the expected number of indices
            if np.absolute(len(task_plan) - len(previous_task_plan)) > 5:
                error_message = "We expected the new plan to be withing 5 actions of the old plan. After running your code it is not. Please regenerate your answer"
            else:
                break
        new_plan_answer = llm_bot.ask(error_message, show_output=True)

    plan_is_correct, error_message, task_plan = extract_plan(new_plan_answer)
    if not plan_is_correct:
        raise Exception(f"Unable to generate valid task plan from LLM. The latest error message was {error_message}")

    global iter
    iter += 1
    global retuned_iter
    retuned_iter = 1

    with open(os.path.join(log_folder, 'task_plan' + str(iter) + '.pkl'), 'wb') as file:
        pickle.dump(f"task_plan = {str(task_plan)}", file)

    return task_plan


def execute_and_log_plans(task_plan: tp.List, evaluation_plan: tp.List, tpu: TaskPlanExecutor):
    # Log of what happened
    task_success, evaluation_log, failure_id = tpu.execute_task_plan(task_plan, evaluation_plan)
    print("evaluation_log")
    print(evaluation_log)

    # A detailed log
    history_log = helper.performance_logger(evaluation_log, evaluation_plan)
    print("history_log")
    print(history_log)

    return task_success, history_log, failure_id


def plan_and_retune(parameter_history, domain_history, warm_start=False):
    global log_folder
    log_folder = create_experiment_log()
    chat_log_file = os.path.join(log_folder, 'chat_log.pckl')
    task_plan_meta_data_file = os.path.join(log_folder, 'task_plan_log.pkl')
    llm_bot = GptChatBot(auto_save_file_name=chat_log_file)
    tpu = TaskPlanExecutor(metadata_file=task_plan_meta_data_file)

    if warm_start:
        llm_bot.set_history(warm_start_history)
        llm_bot.print_history()
        _, _, task_plan = extract_plan(action_plan_answer)  # We know this to be correct
        _, _, evaluation_plan = extract_plan(eval_plan_answer)  # We know this to be correct
    else:
        task_plan = ask_for_task_plan(llm_bot, task, scene_description, objects, locations)
        evaluation_plan = ask_for_evaluation_plan(llm_bot)

    # TODO: Check task plan content
    # TODO: Check evaluation plan content
    # TODO: Check task-evaluation congruence

    # Test plan
    print("=============== EXECUTING INITIAL PLAN =================")
    task_success, history_log, failure_id = execute_and_log_plans(task_plan, evaluation_plan, tpu)

    # Update explored space
    parameter_history = helper.explored_parameter_space(task_plan, history_log, parameter_history)

    for i_replan in range(replan_count):

        if task_success:
            print("=============== TASK SUCCEEDED =================")
            break

        else:

            # Retuning plan
            i_retune = 0
            while i_retune < retune_count:
                # Successful plan: finish
                if task_success:
                    print("=============== TASK SUCCEEDED =================")
                    break

                # Retune plan
                else:
                    task_plan = ask_for_action_plan_retune(llm_bot, history_log, parameter_history, task_plan, i_retune)

                print(f"=============== EXECUTING RETUNED PLAN {i_retune} =================")
                # Test plan
                task_success, history_log, new_failure_id = execute_and_log_plans(task_plan, evaluation_plan, tpu)

                if new_failure_id <= failure_id:
                    i_retune += 1
                else:
                    failure_id = new_failure_id
                    # i_retune += 1

                # Update explored space
                parameter_history = helper.explored_parameter_space(task_plan, history_log, parameter_history)

            # Replan if needed
            if not task_success:
                print("=============== TASK RETUNING FAILED =================")

                task_plan = ask_for_action_replanned(llm_bot, history_log, task_plan, parameter_history)
                evaluation_plan = ask_for_evaluation_plan(llm_bot)
                print(f"=============== EXECUTING REPLANNED PLAN {i_replan} =================")
                task_success, history_log, failure_id = execute_and_log_plans(task_plan, evaluation_plan, tpu)

    # Save parameter_history
    with open(os.path.join(log_folder, 'parameter_history.pkl'), 'wb') as file:
        pickle.dump(parameter_history, file)

    # Logging the domain information
    domain_history = helper.explored_domain(task_plan=task_plan,
                                            evaluation_plan=evaluation_plan,
                                            success_state=task_success,
                                            domain_history=domain_history)

    if not task_success:
        print("=====TOTAL_FAILURE============")
    return task_success, parameter_history, domain_history


if __name__ == "__main__":
    rospy.init_node("llm_planner")
    parameter_history = parameter_history = {'success': {}, 'failure': {}}
    domain_history = None
    success, parameter_history, domain_history =  plan_and_retune(parameter_history, domain_history, warm_start=True)

if False:
    action_plan_ask = """
    The image shows an indoor scene with a white table at the center, scattered with various objects. On the table, there are crumpled paper balls, a whole apple, a half-eaten apple, and two different types of glasses, one of which appear to be empty and one partially filled with a yellowish liquid. To the left, there is a large white sink with a faucet against a tiled wall. A storage shelf is also kept near the table.In the background, there are two doors, one closed and one ajar, and a folded white chair next to the table. On the right side, a large red trash can is visible, filled with discarded items including colorful plastic wrappers. The room has a clinical or institutional feel, possibly a break room or a workshop space.The list of recognised objects is:

    objects = ['white table', 'crumpled paper ball 1', 'crumpled paper ball 2', 'crumpled paper ball 3', 'whole apple', 'half-eaten apple', 'empty glass 1', 'glass with yellowish liquid', 'large red trash can']

    The list of recognised locations is:

    locations = ['large white sink', 'faucet', 'tiled wall', 'closed door', 'ajar door', 'folded white chair', 'clinical room', 'storage shelf']


    There is a robot, labeled 'robot', that can only manipulate ONE object at a time. The robot accepts commands in the form 'action functions' written in python. These action functions, which can be imported from the 'action_functions' library are:

    drop(location: str, obstacle_clearance: float) -> None: # Goes to "loaction" and drops the grasped object over it

    approach(object_to_grasp: str, speed: float, obstacle_clearance: float, grasp: str) -> None:  # Moves the robot close to "object_to_grasp" so that the object is in robot's reach

    place(location: str, orientation: float, speed: float, obstacle_clearance: float) -> None: #Positions the "grasped_object" on/at/in the "location" and release the grasp.

    pick(object_to_grasp: str, orientation: float, speed: float, obstacle_clearance: float, grasp: str) -> None: # Instructs the robot to pick up the "object_to_grasp", if it is close enough

    The "speed" argument for 'approach', 'pick' and 'place' functions, assumes value in [0,1] and regulates how fast the robot moves. The closer the the value is to 1 the faster the robot moves. moving with higher speed is faster but might result in a jerky and less precise motion.

    The "orientation" argument for 'pick', and 'place' functions, regulates how crucial it is for the robot to maintain the original orientation of object that the robot is holding. A value closer to 1, instructs the robot to strictly maintain the orientation, but may result in difficulty to avoid external perturbations or obstacles .

    The "grasp" argument for 'approach' and 'pick' assumes one of the two values ('top', 'side'), where "top" instructs the robot to approach or pick the object from the top and selecting "side" instructs the robot to approach or pick the object from the side.

    The "obstacle_clearance" for 'drop', 'approach', 'place' and 'pick' functions defines how close the robot can get from an object (including the one it is trying to grasp in the pick action) before starting to avoid it. The distance is in meter. Small values allow the robot to get closer to obstacles and usually give a better chance of reaching the object, picking it and holding it. However, small values also increase the chance of getting into a collision. Typicall values are between 0.005 and 0.3 although values out of this range are possible.

    The actions described in these functions are the only motions known to the robot. The task for the robot is - clear the table efficiently. First explain how you are going to solve the task, why each step is executed and how it makes sense to a human that you do it this way.

    Then, using the actions functions, 'objects' and 'locations', define a task plan as a Python list of tuples (named 'task_plan'), for the robot to follow. The action_functions from the task_plan will be directly run on the robot. Each element of the list is a tuple of the form (action number, action function name string, (arguments)). For each action, use object and task specific arguments.

    The first index of the plan should be 0.

    Do not make any assumptions. For the task plan, output a single code block. With each action include as the comment the reasoning behind the action and its parameters.

    Assume the necessary modules containing action functions have already been loaded. Write only a single assignment statement creating the full 'task_plan'

    Do not abbreviate anything in the code you are writing. No matter how long it is, write it in full.
    """

    action_plan_answer = """
    To solve the task of clearing the table efficiently, the robot should prioritize removing larger items and items that might cause mess or safety concerns first. Here’s an outline of the logical sequence of steps:

    1. Crumpled paper balls are picked up and dropped into the trash can - these are lightweight and potentially less stable, making them prone to rolling or blowing off the table.
    2. The half-eaten apple is next, as it is perishable waste - this also goes into the trash can.
    3. The whole apple is an item that can be stored rather than trashed, so it should be placed on the storage shelf.
    4. Empty glass and glass with liquid will be picked up carefully to prevent any spillage, and they should be placed on the sink area, where they can be assumed to be washed or dealt with appropriately.
    5. Finally, the table will be clear, and any small debris remaining on it can be assumed to be cleared off by a separate cleaning method, not included in this plan.

    Given the above plan, here’s the 'task_plan' for the robot to follow:

    ```python
    task_plan = [
        (0, 'approach', ('crumpled paper ball 1', 0.5, 0.05, 'top')), # Approach first paper ball at medium speed with top grasp and reasonable obstacle clearance
        (1, 'pick', ('crumpled paper ball 1', 0.5, 0.5, 0.05, 'top')), # Pick up the first paper ball, orientation semi-important, at medium speed
        (2, 'drop', ('large red trash can', 0.05)), # Drop the first paper ball into the trash can, small clearance to ensure accuracy
        (3, 'approach', ('crumpled paper ball 2', 0.5, 0.05, 'top')), # Repeat approach for second paper ball
        (4, 'pick', ('crumpled paper ball 2', 0.5, 0.5, 0.05, 'top')), # Repeat pick for second paper ball
        (5, 'drop', ('large red trash can', 0.05)), # Repeat drop for second paper ball
        (6, 'approach', ('crumpled paper ball 3', 0.5, 0.05, 'top')), # Repeat approach for third paper ball
        (7, 'pick', ('crumpled paper ball 3', 0.5, 0.5, 0.05, 'top')), # Repeat pick for third paper ball
        (8, 'drop', ('large red trash can', 0.05)), # Drop the third paper ball into the trash can
        (9, 'approach', ('half-eaten apple', 0.4, 0.05, 'side')), # Approach the half-eaten apple at a careful speed with side grasp
        (10, 'pick', ('half-eaten apple', 0.8, 0.4, 0.05, 'side')), # Pick up the half-eaten apple, high orientation importance to avoid spillage
        (11, 'drop', ('large red trash can', 0.05)), # Drop the half-eaten apple into the trash can
        (12, 'approach', ('whole apple', 0.5, 0.05, 'top')), # Approach whole apple at medium speed
        (13, 'pick', ('whole apple', 0.5, 0.5, 0.05, 'top')), # Pick up whole apple, maintaining orientation moderately
        (14, 'place', ('storage shelf', 0.5, 0.5, 0.05)), # Place whole apple on the storage shelf, orientation somewhat important
        (15, 'approach', ('empty glass 1', 0.3, 0.05, 'top')), # Approach empty glass at slower speed to avoid toppling
        (16, 'pick', ('empty glass 1', 1.0, 0.3, 0.05, 'top')), # Pick empty glass, high orientation importance to prevent spills
        (17, 'place', ('large white sink', 1.0, 0.3, 0.05)), # Place empty glass in the sink, maintaining orientation strictly
        (18, 'approach', ('glass with yellowish liquid', 0.3, 0.05, 'top')), # Approach glass with liquid carefully
        (19, 'pick', ('glass with yellowish liquid', 1.0, 0.3, 0.05, 'top')), # Pick glass with liquid, high orientation importance
        (20, 'place', ('large white sink', 1.0, 0.3, 0.05)), # Place glass with liquid in the sink, high orientation to avoid spillage
    ]
    ```

    """

    eval_plan_ask = """
    The robot may not be able to execute an action function, or encounter object collision during execution. Thus, it is required to check for completion of each action function.

    For this, we define some 'checking functions' written in python. These checking functions, which can be imported from the 'checking_functions' library are:

    can_grasp(object_to_grasp: str) -> bool: # Returns True if robot is close-enough to the "object_to_grasp" to securley grasp it

    holding() -> bool # Returns True if the robot is holding an object

    at_location(object: str, location: str) -> bool: # Returns True if the "object" is at the "location"

    collision_free() -> str: # If the robot encounters a collision while executing the preceding action, returns the object label string. Otherwise, ''.

    timeout() -> Bool: # Returns True if the preceding action was executed in a timely fashion

    check_motion_health() ->  bool: # Returns True if the robot's motion during the preceding action was safe for its hardware

    can_reach(goal: str) -> bool: # Returns True if it is feasible for the robot to reach the "goal" object or location from the current state. Objects that are out of the workspace will always return false.

    Using the checking_functions, locations and objects, define an evaluation plan (named 'evaluation_plan') to verify the succesful execution of each action. Additionally, for each action verify without fail:
    - collision free
    - timely motion
    - motion health

    Output this plan as a Python list of tuples, where each tuple is of the form (action number int, dictionary with check_function names as keys and a tuple of arguments as value, tuple of expected outputs). Do not assume any other object or location, beyond those in object_labels.

    Generate the entire plan. No reasoning, direct output.
    """

    eval_plan_answer = """
    ```python
    evaluation_plan = [
        (0, {'can_grasp': ('crumpled paper ball 1', 'top'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (1, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (2, {'at_location': ('crumpled paper ball 1', 'large red trash can'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (3, {'can_grasp': ('crumpled paper ball 2', 'top'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (4, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (5, {'at_location': ('crumpled paper ball 2', 'large red trash can'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (6, {'can_grasp': ('crumpled paper ball 3', 'top'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (7, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (8, {'at_location': ('crumpled paper ball 3', 'large red trash can'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (9, {'can_grasp': ('half-eaten apple', 'side'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (10, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (11, {'at_location': ('half-eaten apple', 'large red trash can'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (12, {'can_grasp': ('whole apple', 'top'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (13, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (14, {'at_location': ('whole apple', 'storage shelf'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (15, {'can_grasp': ('empty glass 1', 'top'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (16, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (17, {'at_location': ('empty glass 1', 'large white sink'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (18, {'can_grasp': ('glass with yellowish liquid', 'top'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (19, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
        (20, {'at_location': ('glass with yellowish liquid', 'large white sink'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True))
    ]
    ```

    """



