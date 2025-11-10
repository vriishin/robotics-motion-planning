import argparse 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from utils import *
from component_1 import *

def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='Collision Checker',
                    description='Checks if there is a collision')
    
    parser.add_argument('--robot', required=True)
    parser.add_argument('--map', required=True)

    return parser.parse_args()

def update_robot(frame, env, robot, axis, patches):
    axis.clear()

    env_size = env.get_size()
    axis.set_xlim(env_size[0][0], env_size[0][1])
    axis.set_ylim(env_size[1][0], env_size[1][1])

    robot =  Robot(env, "car") 

    for rect in env.get_obstacles():
        obstacle = create_rec(rect, fc="green")
        axis.add_patch(obstacle)
        patches[rect] = obstacle

    car = car_pose_to_rect(robot.get_pose(), 0.5, 0.3).get_rect("blue")
    axis.add_patch(car)

    collisions = get_collisions(env, car_pose_to_rect(robot.get_pose(), 0.5, 0.3))
    for o in collisions:
        patches[o].set(fc="red")

def visualize_robot(env: Enviornment, save_path="car_animation.mp4"):
    fig, axis = plt.subplots()
    robot = Robot(env, "car")
    patches = {}

    ani = FuncAnimation(
        fig, update_robot, frames=10, fargs=(env, robot, axis, patches), repeat=False
    )

    writer = FFMpegWriter(fps=1)
    ani.save(save_path, writer=writer)

def update_arm(frame, env, arm, axis, patches):
    axis.clear() 

    env_size = env.get_size()
    axis.set_xlim(env_size[0][0], env_size[0][1])
    axis.set_ylim(env_size[1][0], env_size[1][1])

    arm = Robot(env, "arm") 

    for rect in env.get_obstacles():
        obstacle = create_rec(rect, fc="green")
        axis.add_patch(obstacle)
        patches[rect] = obstacle

    collisions = set()
    for link in arm_pose_to_rect(arm.get_pose(), [[0.2, 2], [0.2, 1.5]]):
        axis.add_patch(link.get_rect("blue"))
        collisions |= get_collisions(env, link)

    for collision in collisions:
        patches[collision].set(fc="red")

def visualize_arm(env: Enviornment, save_path="animation.mp4"):
    fig, axis = plt.subplots()
    arm = Robot(env, "arm")
    patches = {}

    ani = FuncAnimation(
        fig, update_arm, frames=10, fargs=(env, arm, axis, patches), repeat=False
    )

    writer = FFMpegWriter(fps=1)  
    ani.save(save_path, writer=writer)

if __name__ == "__main__":
    args = parse_arguments()
    env = scene_from_file(args.map)
    if args.robot == 'arm':
        visualize_arm(env, save_path=f"img/{args.map}_arm.mp4")
    elif args.robot == 'robot':
        visualize_robot(env, save_path=f"img/{args.map}_car.mp4")
