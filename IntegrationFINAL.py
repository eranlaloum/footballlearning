import math
from pyrep.robots.arms.ur5 import UR5
from pyrep.robots.arms.ur5 import Arm
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
import random
from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.backend import sim
import numpy as np
import cv2 as cv
import cv2
import csv

prevX = 0  # global variable
prevY = 0
num_games=10
touch_wall=False
#strikes=[]
initial_positions=[] #x,y,vx,vy,hit/not (hit-1, not-0)

SCENE_FILE = '/home/nitzan/CoppeliaSim/ED_learning.ttt'
# SCENE_FILE = '/home/user/Desktop/robotic_football_learning-main/ED_learning.ttt'
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()
cam = VisionSensor("cam")
ball = Shape('Sphere')
ballHandle=sim.simGetObjectHandle('Sphere')
pi = math.pi
agent = UR5()

## Initialization Mean-Shift Algorithm
# pr.step()
z=0
while z<4:
    img = cam.capture_rgb()
    img = np.uint8(img * 256)
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    z=z+1
    pr.step()
# START
img = cam.capture_rgb()
img = np.uint8(img * 256)
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
img = img[450:962, 0:1024]  # region of interest
frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('frame FIRST' + '.jpeg', frame)
x, y, w, h = 792, 282, 50, 40
track_window = (x, y, w, h)
# create the region of interest
roi = frame[y:y + h, x:x + w]
# converting BGR to HSV format
imgHSV = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
# apply mask on the HSV frame
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
mask = cv.inRange(imgHSV, lower_red, upper_red)
# get histogram for hsv channel
roi_hist = cv.calcHist([imgHSV], [0], mask, [180], [0, 180])
# normalize the retrieved values
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
# Setup the termination criteria, either 25 iteration or move by atleast 2 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 25, 2)
z=0

# calculate x,y, Vx, Vy
prevX= x + (w/2)
prevY= y + (h/2)

def tablecells():  # divide the game board to cells
    table = [[0, 0, 0, 0.30, 0, 0.10], [1, 0, 0.3, 0.6, 0, 0.1], [2, 0, 0.6, 0.9, 0, 0.1], [3, 0, 0.9, 1.2, 0, 0.1],
             [4, 0, 1.2, 1.5, 0, 0.1], [5, 0, 1.5, 1.8, 0, 0.1],
             [0, 1, 0, 0.3, 0.1, 0.2], [1, 1, 0.3, 0.6, 0.1, 0.2], [2, 1, 0.6, 0.9, 0.1, 0.2],
             [3, 1, 0.9, 1.2, 0.1, 0.2], [4, 1, 1.2, 1.5, 0.1, 0.2], [5, 1, 1.5, 1.8, 0.1, 0.2],
             [0, 2, 0, 0.3, 0.2, 0.3], [1, 2, 0.3, 0.6, 0.2, 0.3], [2, 2, 0.6, 0.9, 0.2, 0.3],
             [3, 2, 0.9, 1.2, 0.2, 0.3], [4, 2, 1.2, 1.5, 0.2, 0.3], [5, 2, 1.5, 1.8, 0.2, 0.3],
             [0, 3, 0, 0.3, 0.3, 0.4], [1, 3, 0.3, 0.6, 0.3, 0.4], [2, 3, 0.6, 0.9, 0.3, 0.4],
             [3, 3, 0.9, 1.2, 0.3, 0.4], [4, 3, 1.2, 1.5, 0.3, 0.4], [5, 3, 1.5, 1.8, 0.3, 0.4],
             [0, 4, 0, 0.3, 0.4, 0.5], [1, 4, 0.3, 0.6, 0.4, 0.5], [2, 4, 0.6, 0.9, 0.4, 0.5],
             [3, 4, 0.9, 1.2, 0.4, 0.5], [4, 4, 1.2, 1.5, 0.4, 0.5], [5, 4, 1.5, 1.8, 0.4, 0.5],
             [0, 5, 0, 0.3, 0.5, 0.6], [1, 5, 0.3, 0.6, 0.5, 0.6], [2, 5, 0.6, 0.9, 0.5, 0.6],
             [3, 5, 0.9, 1.2, 0.5, 0.6], [4, 5, 1.2, 1.5, 0.5, 0.6], [5, 5, 1.5, 1.8, 0.5, 0.6],
             [0, 6, 0, 0.3, 0.6, 0.7], [1, 6, 0.3, 0.6, 0.6, 0.7], [2, 6, 0.6, 0.9, 0.6, 0.7],
             [3, 6, 0.9, 1.2, 0.6, 0.7], [4, 6, 1.2, 1.5, 0.6, 0.7], [5, 6, 1.5, 1.8, 0.6, 0.7],
             [0, 7, 0, 0.3, 0.7, 0.8], [1, 7, 0.3, 0.6, 0.7, 0.8], [2, 7, 0.6, 0.9, 0.7, 0.8],
             [3, 7, 0.9, 1.2, 0.7, 0.8], [4, 7, 1.2, 1.5, 0.7, 0.8], [5, 7, 1.5, 1.8, 0.7, 0.8],
             [0, 8, 0, 0.3, 0.8, 0.9], [1, 8, 0.3, 0.6, 0.8, 0.9], [2, 8, 0.6, 0.9, 0.8, 0.9],
             [3, 8, 0.9, 1.2, 0.8, 0.9], [4, 8, 1.2, 1.5, 0.8, 0.9], [5, 8, 1.5, 1.8, 0.8, 0.9]]
    return table


def cellindex(table, x, y):  # return the index of the cell from a given x,y coordinates
    index = 0
    for i in range(len(table)):
        for j in range(len(table[i])):
            if x >= table[i][2] and x < table[i][3] and y >= table[i][4] and y < table[i][5]:
                return [table[i][0], table[i][1]]

game_board = tablecells()


def ballCoordinates(x, y, w, h):  # work with image format without imread
    global prevX
    global prevY
    global touch_wall
    global xpos
    global ypos
    #Originx = 179
    #Originy = 34
    xpos = x + (w / 2)
    ypos = y + (h / 2)
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if prevX == 0 and prevY == 0:  # first image to prev
        prevX = x
        prevY = y
        return [x, y]
    #xpos = xpos - Originx  # centert of coordinate x
    #ypos = ypos - Originy
    #xpos = (xpos / 723) * 1.8
    #ypos = (ypos / 352) * 0.9
    Vx = (xpos - prevX) / 0.05
    Vy = (ypos - prevY) / 0.05
    prevX = xpos
    prevY = ypos
    return [xpos, ypos, Vx, Vy]



def Frame_ballCoordinates():
    global z
    global track_window
    global x,y,w,h
    pr.step()
    z = z + 1
    img = cam.capture_rgb()
    img = np.uint8(img * 256)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = img[450:962, 0:1024]  # region of interest
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert to hsv
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(imgHSV, lower_red, upper_red)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask_red, connectivity=8)
    nb_components = nb_components - 1  # taking out the background which is also considered a componentbut
    if nb_components == 0:
        touch_wall = True
        return "no ball"
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    # apply meanshift to get the new location
    _, track_window = cv.meanShift(dst, track_window, term_crit)
    x, y, w, h = track_window
    return ballCoordinates(x, y, w, h)

def randomposition():
    x = random.uniform(-0.1, 0.5)
    y = -1.71
    ball.set_position([x, y, 0.035])
    vx = 0.55
    vy = random.uniform(-0.2, 0.2)
    sim.simSetObjectFloatParameter(ballHandle, 3001, vx)
    sim.simSetObjectFloatParameter(ballHandle, 3000, vy)
    return [x, y, vx, vy]


def move_arm(position, quaternion, ignore_collisions=False):  # move the robot to direction
    arm_path = agent.get_linear_path(position,
                                     quaternion=quaternion,
                                     ignore_collisions=ignore_collisions)
    arm_path.visualize()
    done = False
    print(num_games)
    while not done:
        done = arm_path.step()
        Frame_ballCoordinates()
        pr.step()
        print('num_games')
    arm_path.clear_visualization()


def rotate_arm(rotation_angle):  # rotate the robot's racket
    joint_positions = agent.get_joint_positions()
    joint_positions[5] = rotation_angle
    joint_positions[4] = 0
    agent.set_joint_positions(
        [joint_positions[0], joint_positions[1], joint_positions[2], joint_positions[3], joint_positions[4],
         joint_positions[5]])


# left dummies
left = Dummy('left')
left_left = Dummy('left_left')
left_center = Dummy('left_center')
left_right = Dummy('left_right')

# center dummies
center = Dummy('center')
center_left = Dummy('center_left')
center_center = Dummy('center_center')
center_right = Dummy('center_right')

# right dummies
right = Dummy('right')
right_left = Dummy('right_left')
right_center = Dummy('right_center')
right_right = Dummy('right_right')

# start position
start_position = agent.get_joint_positions()


def move_main_direction(direction, rotation):  # move the robot: right left or center
    move_arm(direction.get_position(), direction.get_quaternion(), True)
    rotate_arm(rotation)
    return direction


def kick_in_right(ymain):  # kick in right side
    if ymain < 2 and ymain >= 0.8:
        move_arm(right_right.get_position(), right_right.get_quaternion(), True)
        pos = [1, 8]
    elif ymain < 0.8 and ymain >= 0.7:
        move_arm(right_center.get_position(), right_center.get_quaternion(), True)
        pos = [1, 7]
    else:
        move_arm(right_left.get_position(), right_left.get_quaternion(), True)
        pos = [1, 6]
    return pos


def kick_in_center(ymain):  # kick in center side
    if ymain < 0.6 and ymain >= 0.5:
        move_arm(center_right.get_position(), center_right.get_quaternion(), True)
        pos = [1, 5]
    elif ymain < 0.5 and ymain >= 0.4:
        move_arm(center_center.get_position(), center_center.get_quaternion(), True)
        pos = [1, 4]
    else:
        move_arm(center_left.get_position(), center_left.get_quaternion(), True)
        pos = [1, 3]
    return pos


def kick_in_left(ymain):  # kick in left side
    if ymain < 0.3 and ymain >= 0.2:
        move_arm(left_right.get_position(), left_right.get_quaternion(), True)
        pos = [1, 2]
    elif ymain < 0.2 and ymain >= 0.1:
        move_arm(left_center.get_position(), left_center.get_quaternion(), True)
        pos = [1, 1]
    else:
        move_arm(left_left.get_position(), left_left.get_quaternion(), True)
        pos = [1, 0]
    return pos

def moving_to_direction():  # move the robot to direction and perform a kick
    ymain = y1 - m * (x1 - 0.3)

    # move the robot to main direction
    if ymain < 2 and ymain >= 0.6:
        main_direction = move_main_direction(right, -pi / 6)
    elif ymain < 0.6 and ymain >= 0.3:
        main_direction = move_main_direction(center, 0)
    else:
        main_direction = move_main_direction(left, pi / 6)

    # wait until the robot need to performe a kick
    while Frame_ballCoordinates()[0] >= 0.55:
        agent.set_joint_positions(agent.get_joint_positions())

    # robot kicking
    if main_direction == right:
        temp.append(3)
        pos = kick_in_right(ymain)
    elif main_direction == center:
        temp.append(1)
        pos = kick_in_center(ymain)
    else:
        temp.append(3)
        pos = kick_in_left(ymain)
    return pos


for game in range(num_games):

    # start game
    temp = randomposition()
    print(temp)
    touch_wall = False
    agent.set_joint_positions(start_position)

    # calculate the ball line
    x1 = prevX
    y1 = prevY
    pr.step()
    agent.set_joint_positions(start_position)
    x2 = Frame_ballCoordinates()[0]
    y2 = Frame_ballCoordinates()[1]
    m = (float(y2) - float(y1)) / (float(x2) - float(x1))


    # moving the robot
    robot_position = moving_to_direction()
    ball_temp = Frame_ballCoordinates()[0]
    move_arm(center.get_position(), center.get_quaternion(), True)
    rotate_arm(pi / 50)
    temp1 = temp

    # check ball location to know if the robot hit the ball or not
    if isinstance(ball_temp, str):
        print(ball_temp)
        temp1.append(0)
    else:
        ball_position = cellindex(game_board, Frame_ballCoordinates()[0], Frame_ballCoordinates()[1])
        print(ball_position)
        if ball_position is None or touch_wall or (robot_position[0] > ball_position[0]):
            temp1.append(0)
        else:
            temp1.append(1)
    initial_positions.append(temp1)  # add to the hit scores the result
    print(game)
pr.stop()
pr.shutdown()

with open('result.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(initial_positions)
