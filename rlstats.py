import argparse
import math
import timeit
import matplotlib.pyplot as plt
from collections import defaultdict

from utils import *
from component_1 import *

def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='PRM* with Metrics and Plots',
                    description='Runs PRM*, computes paths, and plots metrics.')
    
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
    k_prm_star = math.e * (1 + 1 / d)
    return math.ceil(k_prm_star * math.log(n))

def PRM_star_multithreaded(env, robot, N, start, goal, d) -> defaultdict:
    G = defaultdict(set)
    G[start] = set()
    G[goal] = set()
    nodes = 0

    import concurrent.futures
    executor = concurrent.futures.ProcessPoolExecutor()
    futures = []

    while nodes < N:
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
        G[random_pose] = set()
        k_neighbors = calculate_k(nodes, d)
        neighbors = get_neighbors(robot, G.keys(), random_pose, k_neighbors)
        futures.append(executor.submit(process_node, env, robot, random_pose, neighbors))

    for future in concurrent.futures.as_completed(futures):
        node, valid_neighbors = future.result()
        for neighbor in valid_neighbors:
            dist = distance(robot, node, neighbor)
            G[node].add((neighbor, dist))
            G[neighbor].add((node, dist))

    executor.shutdown()
    return G

def path_length(robot, path):
    total_dist = 0.0
    for i in range(len(path)-1):
        total_dist += distance(robot, path[i], path[i+1])
    return total_dist

def run_experiment(env, robot, start, goal, d, trials=5):
    """
    Run multiple trials to gather statistics:
    - Success rate
    - Path lengths
    - Time-to-goal
    """
    success_count = 0
    path_lengths = []
    times = []

    # Increase N or adjust as needed
    N = 5000

    for trial in range(trials):
        print(f"Trial {trial+1}/{trials}")
        start_time = timeit.default_timer()

        G = PRM_star_multithreaded(env, robot, N, start, goal, d)
        path = uniform_cost_search(G, start, goal)
        
        end_time = timeit.default_timer()
        time_elapsed = end_time - start_time

        if path is not None and len(path) > 1:
            success_count += 1
            p_len = path_length(robot, path)
            path_lengths.append(p_len)
            times.append(time_elapsed)
        else:
            # No successful path found in this trial
            times.append(time_elapsed)  # Still record time (could be large)
    
    success_rate = success_count / trials
    return success_rate, path_lengths, times

def plot_metrics(success_rate, path_lengths, times):
    # Plot success rate as a simple bar
    plt.figure()
    plt.bar(['Success Rate'], [success_rate])
    plt.ylim(0,1)
    plt.title("Success Rate")
    plt.ylabel("Rate")
    plt.savefig("success_rate.png", dpi=300)
    plt.close()

    # Plot histogram of path lengths
    if len(path_lengths) > 0:
        plt.figure()
        plt.hist(path_lengths, bins=10, color='blue', edgecolor='black')
        plt.title("Distribution of Path Lengths")
        plt.xlabel("Path Length")
        plt.ylabel("Frequency")
        plt.savefig("path_length_hist.png", dpi=300)
        plt.close()

    # Plot histogram of times
    plt.figure()
    plt.hist(times, bins=10, color='green', edgecolor='black')
    plt.title("Distribution of Time-to-Goal")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.savefig("time_distribution.png", dpi=300)
    plt.close()

def main():
    args = parse_arguments()
    env = scene_from_file(args.map)

    if args.robot == "robot":
        args.robot = "car"

    if args.robot == "car":
        start = Pose((args.start[0], args.start[1], args.start[2]))
        goal = Pose((args.goal[0], args.goal[1], args.goal[2]))
        d = 3  # x, y, and theta 
    elif args.robot == "arm":
        start = Pose((args.start[0], args.start[1]))
        goal = Pose((args.goal[0], args.goal[1]))
        d = 2  # DOF for arm

    # Run multiple trials to collect stats
    success_rate, path_lengths, times = run_experiment(env, args.robot, start, goal, d, trials=10)

    print(f"Success Rate: {success_rate*100:.2f}%")
    if path_lengths:
        avg_path_length = sum(path_lengths) / len(path_lengths)
        print(f"Average Path Length: {avg_path_length:.2f}")
    avg_time = sum(times) / len(times)
    print(f"Average Time-to-Goal: {avg_time:.2f}s")

    # Plot the results
    plot_metrics(success_rate, path_lengths, times)

    G = PRM_star_multithreaded(env, args.robot, 5000, start, goal, d)
    path = uniform_cost_search(G, start, goal)
    robot_visualization(env, args.robot, start, goal, G, path)


if __name__ == '__main__':
    main()
