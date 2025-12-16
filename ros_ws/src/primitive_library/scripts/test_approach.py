import rospy
import time
from primitives.action_functions import approach, pick, place, drop
from primitives.predicates import holding, collision_free


def main() -> None:

    object_to_grasp = "paper_ball_1"
    print("Starting main")
    rospy.init_node("test_approach")
    print("Node initialized")
    print("Approach")
    approach(object_to_grasp=object_to_grasp,
             speed=.2,
             grasp="top",
             obstacle_clearance=0.15)
    print(collision_free())

    print("Pick")
    pick(object_to_grasp=object_to_grasp,
         orientation=0., speed=.2,  obstacle_clearance=0.005, grasp_orientation="top")
    time.sleep(4)
    print(collision_free())
    holding()
    print("Place")
    drop(object_to_grasp="sink",  orientation=0., speed=1., obstacle_clearance=0.02)
    print(collision_free())
    rospy.signal_shutdown("Approach test finished")


if __name__ == "__main__":
    main()