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

task = "put away the 'half-eaten apple' and the 'glass with yellowish liquid'"
# task = "put away the 'crumpled paper ball 2', the 'half-eaten apple' and the 'crumpled paper ball 1'"
# task = "put away the 'half-eaten apple', the 'crumpled paper ball 2', the 'empty glass 1' and the 'glass with yellowish liquid'"

iter = 1
retune_count = 3
replan_count = 7

log_folder = ""


def extract_plan(full_answer: str) -> tp.Tuple[bool, str, tp.List]:

    error_message = ""
    plan = []

    raw_code = ProblemInterpreter.extract_code_from_md(full_answer, ['python', 'python3', 'python2', ''])

    # Check number of code blocks
    if len(raw_code) == 0:
        breakpoint()
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

    with open(os.path.join(log_folder, 'retuned_plan' + str(iter) + "_" + str(retune_idx) + '.pkl'), 'wb') as file:
        pickle.dump(f"task_plan = {str(retuned_plan)}", file)

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
    log_folder = create_experiment_log("exp_2_0")
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

    return task_success, parameter_history, domain_history


if __name__ == "__main__":
    rospy.init_node("llm_planner")
    for _ in range(10):
        iter = 1
        parameter_history = parameter_history = {'success': {}, 'failure': {}}
        domain_history = None
        success, parameter_history, domain_history =  plan_and_retune(parameter_history, domain_history, warm_start=False)
