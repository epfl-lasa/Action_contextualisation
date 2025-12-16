import rospy
from primitives.execute_task_plan import TaskPlanExecutor
from llm_planner.utils import create_experiment_log
import os
# Assigning the full task plan as a Python list of tuples. Each tuple represents an action with parameters.

task_plan = [
    (0, 'approach', ('crumpled paper ball 1', 0.5, 0.05, 'top')), 
    (1, 'pick', ('crumpled paper ball 1', 0.3, 0.02, 'top')), 
    (2, 'drop', ('large red trash can', 0.5, 0.05)), 
    (3, 'approach', ('crumpled paper ball 2', 0.5, 0.05, 'top')), 
    (4, 'pick', ('crumpled paper ball 2', 0.3, 0.01, 'top')), 
    (5, 'drop', ('large red trash can', 0.5, 0.05)), 
    (6, 'approach', ('whole apple', 0.5, 0.05, 'top')), 
    (7, 'pick', ('whole apple', 0.3, 0.01, 'top')), 
    (8, 'place', ('storage shelf', 0.5, 0.5, 0.05)), 
    (9, 'approach', ('empty glass 1', 0.5, 0.05, 'side')), 
    (10, 'pick', ('empty glass 1', 0.3, 0.01, 'side')), 
    (11, 'place', ('large white sink', 0., 0.5, 0.05)), 
    (12, 'approach', ('glass with yellowish liquid', 0.5, 0.05, 'side')), 
    (13, 'pick', ('glass with yellowish liquid', 0.3, 0.02, 'side')), 
    (14, 'place', ('large white sink', 0.3, 0.5, 0.05)), 
    (15, 'approach', ('half-eaten apple', 0.5, 0.05, 'top')), 
    (16, 'pick', ('half-eaten apple', 0.3, 0.02, 'top')), 
    (17, 'drop', ('large red trash can', 0.5, 0.05))
]



evaluation_plan  = [
    (0, {'can_grasp': ('crumpled paper ball 1', 'top'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)), 
    (1, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)), 
    (2, {'at_location': ('crumpled paper ball 1', 'large red trash can'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)), 
    (3, {'can_grasp': ('crumpled paper ball 2', 'top'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)), 
    (4, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)), 
    (5, {'at_location': ('crumpled paper ball 2', 'large red trash can'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)), 
    (6, {'can_grasp': ('whole apple', 'top'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)), 
    (7, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)), 
    (8, {'at_location': ('whole apple', 'storage shelf'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)), 
    (9, {'can_grasp': ('empty glass 1', 'side'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)), 
    (10, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)), 
    (11, {'at_location': ('empty glass 1', 'large white sink'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)), 
    (12, {'can_grasp': ('glass with yellowish liquid', 'side'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)), 
    (13, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)), 
    (14, {'at_location': ('glass with yellowish liquid', 'large white sink'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)), 
    (15, {'can_grasp': ('half-eaten apple', 'top'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)), 
    (16, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)), 
    (17, {'at_location': ('half-eaten apple', 'large red trash can'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True))
]





def main():
    rospy.init_node("test_task_plan")
    log_folder = create_experiment_log('paper_ball_high_score')
    task_plan_meta_data_file = os.path.join(log_folder, 'task_plan_log.pkl')
    tpu = TaskPlanExecutor(metadata_file=task_plan_meta_data_file)
    tpu.execute_task_plan(task_plan, evaluation_plan)


if __name__ == "__main__":
    main()