import argparse
import time
import numpy as np
from component_1 import *
from prm import *
from ao_planner import * 
from rrt import *
from hw3.utils import uniform_cost_search2

# Parsing arguments for input configurations
def parse_arguments():
    parser = argparse.ArgumentParser(description="Planner Comparison")
    parser.add_argument('--map', required=True, help="Path to the map file.")
    parser.add_argument('--robot', required=True, choices=['arm', 'robot'], help="Type of robot: 'arm' or 'car'.")
    parser.add_argument('--start', nargs='+', type=float, required=True, help="Start configuration.")
    parser.add_argument('--goal', nargs='+', type=float, required=True, help="Goal configuration.")
    return parser.parse_args()

# Main evaluation function
def main_evaluation():
    NUM_TIMES = 10
    args = parse_arguments()
    env = scene_from_file(args.map)
    
    start = Pose(args.start)
    goal = Pose(args.goal)
    
    if args.robot == "robot":
        d = 3
        args.robot = 'car'
    else:
        d = 2

    k = 6
    max_iter = 500  # Maximum iterations for each planner

    results = {}

    # List of planners to evaluate
    planners = {
        "PRM*": lambda: PRM_star_multithreaded(env, args.robot, max_iter, start, goal, d),
        "PRM": lambda: PRM_multithreaded(env, args.robot, max_iter, k, start, goal),
        "RRT": lambda: RRT(env, args.robot, max_iter, start)
    }

    # Evaluate each planner directly
    for planner_name, planner_func in planners.items():
        success_count = 0
        path_lengths = []
        times = []

        print(f"Evaluating {planner_name}...")
        
        # Run the planner 10 times
        for i in range(NUM_TIMES):
            print(i)
            start_time = time.time()
            
            # Run the planner to generate graph G
            if planner_name != "RRT":
                G = planner_func()
                
                # Find the shortest path in the graph G
                path, total_cost = uniform_cost_search2(G, start, goal)
                
                end_time = time.time()
                execution_time = end_time - start_time
                times.append(execution_time)

                # Validate and measure the path
                if path:
                    path_lengths.append(total_cost)  # Using total_cost from UCS as the path length
                    success_count += 1
            else:
                history = planner_func()
                G = history[-1]
                min_pose = start
                for pose in G.keys():
                    if distance(args.robot, pose, goal) <= distance(args.robot, min_pose, goal):
                        min_pose = pose
                
                if distance(args.robot, min_pose, goal) <= 2:
                    path, total_cost = uniform_cost_search2(G, start, min_pose)
                    path_lengths.append(total_cost)  # Using total_cost from UCS as the path length
                    success_count += 1
                
                end_time = time.time()
                execution_time = end_time - start_time
                times.append(execution_time)
                    

        # Results summary for the current planner
        print(success_count)
        print(path_lengths)
        success_rate = success_count / NUM_TIMES
        avg_path_length = np.mean(path_lengths) if path_lengths else float('inf')
        avg_time = np.mean(times)

        # Store the results
        results[planner_name] = {
            "Success Rate": success_rate,
            "Average Path Length": avg_path_length,
            "Average Computation Time": avg_time
        }

        print(f"\n{planner_name} Results:")
        print(f"  Success Rate: {success_rate * 100}%")
        print(f"  Average Path Length: {avg_path_length:.4f}")
        print(f"  Average Computation Time: {avg_time:.4f} seconds\n")
    
    # Print all results
    for planner_name, metrics in results.items():
        print(f"{planner_name} Final Summary:")
        print(f"  Success Rate: {metrics['Success Rate'] * 100}%")
        print(f"  Average Path Length: {metrics['Average Path Length']:.4f}")
        print(f"  Average Computation Time: {metrics['Average Computation Time']:.4f} seconds\n")

if __name__ == '__main__':
    main_evaluation()
