import argparse
from collections import defaultdict
import concurrent.futures
import os
import threading
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

def calculate_k(n, d):
    k = math.e * (1 + 1 / d)
    return math.ceil(k * math.log(n))

def PRM_star_multithreaded(env, robot, N, start, goal, d) -> defaultdict:
    G = defaultdict(set)
    G[start] = set()
    G[goal] = set()
    nodes = 0

    executor = concurrent.futures.ProcessPoolExecutor()
    futures = []

    while nodes < N:
        print(nodes)
        if robot == "car":
            random_pose = Robot.random_car(env)
            rects = [car_pose_to_rect(random_pose, 0.5, 0.3)]
        elif robot == "arm":
            random_pose = Robot.random_arm()
            rects = [rect for rect in arm_pose_to_rect(random_pose, [[0.2, 2], [0.2, 1.5]])]
        
        collision_found = False
        for rect in rects:
            if not collision_free(env, rect):
                collision_found = True
                break
        
        if collision_found:
            continue
        
        nodes += 1
        print(nodes)
        G[random_pose] = set()
        k_neighbors = calculate_k(nodes, d)

        neighbors = get_neighbors(robot, G.keys(), random_pose, k_neighbors)
        futures.append(executor.submit(process_node, env, robot, random_pose, neighbors))


    for future in concurrent.futures.as_completed(futures):
        node, valid_neighbors = future.result()
        print(nodes)
        nodes -= 1
        for neighbor in valid_neighbors:
            dist = distance(robot, node, neighbor)
            G[node].add((neighbor, dist))
            G[neighbor].add((node, dist))

    executor.shutdown()

    return G

def main():
    args = parse_arguments()
    env = scene_from_file(args.map)

    if args.robot == "robot":
        args.robot = "car"

    if args.robot == "car":
        start = Pose((args.start[0], args.start[1], args.start[2]))
        goal = Pose((args.goal[0], args.goal[1], args.goal[2]))

        d = 3  #  x, y, and theta 


    if args.robot == "arm":
        start = Pose((args.start[0], args.start[1]))
        goal = Pose((args.goal[0], args.goal[1]))

        d = 2  # joints has 2 DOF
    
    execution_time = timeit.timeit(lambda: PRM_star_multithreaded(env, args.robot, 5000, start, goal, d), number=1)

    G = PRM_star_multithreaded(env, args.robot, 5000, start, goal, d)
    print("finished generating PRM")
    path = uniform_cost_search(G, start, goal)
    print("finished computing path")
    robot_visualization(env, args.robot, start, goal, G, path)
    robot_animation(env, args.robot, start, goal, G, path, "img/prm_star.mp4")
    


if __name__ == '__main__':
    main()