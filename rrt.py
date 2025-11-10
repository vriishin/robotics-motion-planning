import argparse
from collections import defaultdict
import concurrent.futures
import timeit

from utils import *
from component_1 import *

def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='Collision Checker',
                    description='Checks if there is a collision')
    
    parser.add_argument('--robot', required=True)
    parser.add_argument('--map', required=True)
    parser.add_argument('--start', nargs='+', type=float, required=True)
    parser.add_argument('--goal', nargs='+', type=float, required=True)

    return parser.parse_args()

def process_node(env, robot, node, neighbors):
    valid_neighbors = []
    for neighbor in neighbors:
        if collision_free_path(env, robot, node, neighbor):
            valid_neighbors.append(neighbor)
    return (node, valid_neighbors)
    

def RRT(env, robot, N, start) -> Pose:
    G = defaultdict(set)
    G[start] = set()
    nodes = 0    
    history = []
    while nodes < N:
        if robot == "car":
            random_pose = Robot.random_car(env)
            rects = [car_pose_to_rect(random_pose, 0.5, 0.3)]

        if robot == "arm":
            random_pose = Robot.random_arm()
            rects = [rect for rect in arm_pose_to_rect(random_pose, [[0.2, 2], [0.2, 1.5]])]
        
        for rect in rects:
            if not collision_free(env, rect):
                break

        nearest = get_neighbors(robot, G.keys(), random_pose, 1)[0]
        pose = steer(robot, nearest, random_pose, 2)
        if collision_free_path(env, robot, nearest, pose):
            G[pose] = set()
            G[nearest].add((pose, distance(robot, nearest, pose)))
            nodes += 1
        if nodes % (N//10) == 0:
            history.append(G.copy())
        
        print(nodes)
    
    return history

def main():
    args = parse_arguments()
    env = scene_from_file(args.map)

    if args.robot == "robot":
        args.robot = "car"

    if args.robot == "car":
        start = Pose((args.start[0], args.start[1], args.start[2]))
        goal = Pose((args.goal[0], args.goal[1], args.goal[2]))

    if args.robot == "arm":
        start = Pose((args.start[0], args.start[1]))
        goal = Pose((args.goal[0], args.goal[1]))

    # execution_time = timeit.timeit(lambda: PRM(env, robot, 250, 6), number=1)
    # print(f"Function took {execution_time:.4f} seconds")

    # execution_time = timeit.timeit(lambda: PRMT(env, robot, 5000, 6), number=1)
    # print(f"Function took {execution_time:.4f} seconds")

    history = RRT(env, args.robot, 1000, start)
    G = history[-1]
    min_pose = start
    for pose in G.keys():
        if distance(args.robot, pose, goal) <= distance(args.robot, min_pose, goal):
            min_pose = pose
    
    if distance(args.robot, min_pose, goal) <= 3:
        path = uniform_cost_search(G, start, min_pose)
    else:
        path = uniform_cost_search(G, start, goal)
    
    rrt_animation(env, args.robot, history)
    robot_animation(env, args.robot, start, goal, G, path, "rrt.mp4")
    

if __name__ == '__main__':
    main()