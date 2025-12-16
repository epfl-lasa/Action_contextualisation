# LLM robotics with ROS integration

## Branches
This project has 2 branches. The realrobot branch on which you are if you can read this should only be used for real robot. If you want to do some simulation please switch to the sim branch.

## Setup
This implementation uses a docker to run the simulator. This allows to port the environment to any computer without having to do extra setup.
The only thing needed is to build the docker. Note that this operation can take anywhere between 5 minutes and 1 hour depending on your computer and how the docker evolves in this project. That being said, you only need to build the docker once. Volumes are mounter on the docker for the ros_ws so you can develop the packages on your computer or on the docker and the changes are applied at both places. No need to rebuild the docker with each changes but depending on what you do you might still need to redo a `catkin_make`.

### Make sure you have a docker environment
Docker requires a few tools to be installed on your computer. You can find information and a script to install them for you [here](https://github.com/epfl-lasa/wiki/wiki/Docker-install-script)

### Build the docker
```bash
bash build_docker.sh    # Only do once. Operations are cached once it is completed
```

### Start the docker
```bash
bash start_docker.sh interactive   # Doing it this way starts the docker in your terminal. The docker will end once you close the terminal
```

```bash
bash start_docker.sh server   # Here we start the docker in the background. You see nothing in your terminal
bash start_docker.sh connect  # Here we connect to the docker and have terminal output. You can connect simultaneously from as many terminal as you want using this. Even if you kill the terminal, the container will always be running in the background
```

#### NVIDIA drivers
Once the docker has started if you plan on using nvidia drivers you should add
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt install nvidia-driver-545

# Check that is worked
nvidia-smi
```

#### GPT API
If you plan on using the GPT api in the robot you will need to reference you api key. However if you are on this branch it is unexpected that you will need this:
```bash
export GPT_API_KEY=YOU_GPT_API_KEY
```

#### Build and source ros packer
Classic, in the docker do:
```bash
catkin_make
source devel/setup.bash
```

### Starting simulator
Once you are in the docker, the files are arranged in a ros workspace. To start the simulator:
```bash
catkin_make
source devel/setup.bash
roslaunch llm_simulator simulator.launch
```

### Developing with docker
You can simply edit your files on your computer, the changes will synchronize with the docker. Alternatively you can use a docker tool in your favorite IDE and work directly in the docker environment. It doesn't matter if you edit files in the docker or on your computer they are synchronized in real time. VScode's docker extension works very well for example.

### Real robot specifics

#### Prerequisit for real robot
Unfortunately you can't port the allegro hand controller to docker. So natively on your computer you will need to download and install the allegro hand ros package: https://github.com/epfl-lasa/allegro_hand_ros.git. You need to switch to the "ahalya" branch which has the ability to control the hand in torque.


#### Staring

First of all you need to start you hand controller on the host computer:
```bash
roslaunch allegro_hand allegro_hand.launch HAND:=right CONTROLLER:=torque
```

Then you need to start the vision server on the docker:
```bash
roslaunch vision_server full_real_robot.launch 
```

Then you need to start the FRIOverlay_gripper on the IIWA tablet. Choose the gripper you have calibrated, pick position control and 
then on a second docker window start the contrller with:

```bash
rosrun primitive_library test_task_plan.py
```

Then allow the robot to move by selected a gain of 500. 

The test_task_plan.py script is a bit make-shift but works. There you can go and change action and evaluation plans to test different scenarios.

#### Vision
Vision is done through the optitrack system. For the realife scenario to work you need to define the following objects:
- paper_ball_1
- paper_ball_2
- champagne_1 (this will be the full one)
- champagne_2
- apple
- eaten_apple
- table
- sink
- shelf
- iiwa_base7

Make sure that the origin of the objects is as much as possible to the center. Ideally for the champagne glasses you will need to put the center about 12 cm from the base of the glass. All objects are represented as collection of spheres. Those spheres are defined in the object frame. If you need to adjust them you can do so in os_ws_src/vision_server/scripts/vision_server.py you will find a bunch of function called `get_OBJECT_NAME_mesh`. Just check the one corresponding to your object.

You also need to make sure that the iiwa_base7 object has z up, and x towards the windows. If the markers have moved you will need to redefine the transformation from the optitrack base to the urdf base of the iiwa7. This transformation is stored in ros_ws_src/vision_server/scripts/vision_server.py and is called `IIWA_BASE_IN_IIWA_MARKER`. 

#### Caveat of real execution

##### Loosing sight of objects
You should avoid loosing sight of the object in real robot execution. If you happend to lose sight of objects the optitrack my put them at a random place in the workspace and generate unexpected collisions. At the begining if an object is not in sight, it will trigger a breakpoint in the vision server. This gives the experiemnter all the time he wants to go back to the optitrack screen, check which object is not in sight, make sure he makes the object visible and then only type "continue" in the vision server windows to resume the execution.

##### Failed predicates
In real life example, we do not stop for failed predicates. They are sometimes inaccurate since the hand and robot tend to hide markers. The predicates will output their results on screen but will not impact the run.

##### Action continue no matter what
For real life scenario the robot always attemps to do all actions. Which means if he can't find an IK or if a collision is detected it will none the less skip to the next action. Even within action such as pick which is made of an approach of the object, a grasp and a vertical flyoff, the robot will attempt to exectue them all even if one fails. That beeing said, action that have a zero length will trigger and error (which is not very clear when you read it) in the task_plan_executor. So sometimes if an action doesn't start because, for example a collision is detected at the first time stamp an error will be thrown and stop the program. This is a mix between a bug and a feature.



### Interfacing a controller using ros topics
There are 4 ros topics to control the allegro hand and IIWA. Those topics are the same as the ones that will be on the real robots. The topics are as follow:
- `/iiwa7/joint_states`  [**JointStates**](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/JointState.html): The joint state of the IIWA
- `/allegro_hand_left/joint_states` [**JointStates**](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/JointState.html): The joint states of the allegro hand
- `/iiwa7/TorqueController/command` [**Float64MultiArray**](http://docs.ros.org/en/melodic/api/std_msgs/html/msg/MultiArrayLayout.html): The torque command for the IIWA robot. We expect layout.dim to be filled with a single layout, have a stride of 1 and a size of 7. Check the parsing in `llm_simulator.py` in `_iiwa_torque_cmd_cb` if you don't know how to access this field. It is also better if the label of the command is `'joint'` but not mandatory.
- `/allegro_hand_left/TorqueController/command` [**Float64MultiArray**](http://docs.ros.org/en/melodic/api/std_msgs/html/msg/MultiArrayLayout.html): The torque command for the Allegro hand. We expect layout.dim to be filled with a single layout, have a stride of 1 and a size of 16. Check the parsing in `llm_simulator.py` in `_allegro_torque_cmd_cb` if you don't know how to access this field. It is also better if the label of the command is `'joint'` but not mandatory.
