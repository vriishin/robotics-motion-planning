import argparse
from collections import defaultdict
import concurrent.futures
import timeit

from hw3.utils import *
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
    

def PRM_multithreaded(env, robot, N, k, start, goal) -> defaultdict:
    G = defaultdict(set)
    G[start] = set()
    G[goal] = set()
    nodes = 0
    futures = []
    executor = concurrent.futures.ProcessPoolExecutor()
    # Sample N nodes collision-free
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
        
        nodes += 1
        G[random_pose] = set()
    
        neighbors = get_neighbors(robot, G.keys(), random_pose, k)
        futures.append(executor.submit(process_node, env, robot, random_pose, neighbors))
        
    # Collect the results as they complete
    for future in concurrent.futures.as_completed(futures):
        node, valid_neighbors = future.result()
        nodes -=1
        for neighbor in valid_neighbors:
            G[node].add((neighbor, distance(robot, node, neighbor)))
            G[neighbor].add((node, distance(robot, neighbor, node)))

    return G

def PRM(env, robot, N, start, goal) -> Pose:
    G = defaultdict(set)
    G[start] = set()
    G[goal] = set()
    nodes = 0
    
    # Sample N nodes collision-free
    while nodes < N:
        if robot == "robot":
            random_pose = Robot.random_car(env)
            rects = [car_pose_to_rect(random_pose, 0.5, 0.3)]

        if robot == "arm":
            random_pose = Robot.random_arm()
            rects = [rect for rect in arm_pose_to_rect(random_pose, [[0.2, 2], [0.2, 1.5]])]
        
        for rect in rects:
            if not collision_free(env, rect):
                break
        
        nodes += 1
        G[random_pose] = set()

    for node in G.keys():
        # k = 6
        neighbors = get_neighbors(G, node, 6)
        for neighbor in neighbors:
            ## THIS FUNCTION CALL IS VERY SLOW
            if collision_free_path(env, robot, node, neighbor):
                G[node].add((neighbor, node.distance(neighbor)))
                G[neighbor].add((node, neighbor.distance(node)))
    
    return G

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

    G = PRM_multithreaded(env, args.robot, 5000, 6, start, goal)

    path = uniform_cost_search(G, start, goal)

    robot_visualization(env, args.robot, start, goal, G, path)
    robot_animation(env, args.robot, start, goal, G, path, "img/prm.mp4")

if __name__ == '__main__':
    main()