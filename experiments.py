import argparse
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import timeit
from collections import defaultdict
import numpy as np
from utils import *
from component_1 import *
from ao_planner import *
from integrated_prm_rl import RLPathPlanner, PRM_star_multithreaded, uniform_cost_search, scene_from_file, Pose

def evaluate_policy(env, robot, start, goal, rl_planner, trials=10):
    successes = 0
    collisions = 0
    path_lengths = []
    times = []

    d = 3 if robot == "car" else 2
    G = PRM_star_multithreaded(env, robot, 250, start, goal, d)
    global_path = uniform_cost_search(G, start, goal)

    # Debug check
    print("Evaluating policy:")
    print("Global path length:", len(global_path))

    for _ in range(trials):
        if len(global_path) < 2:
            # No valid path found
            collisions += 1
            continue

        current_pose = global_path[0]
        total_dist = 0.0
        step_count = 0
        success = True

        for i in range(len(global_path)-1):
            subpath = rl_planner.nav_subgoal(global_path[i], global_path[i+1], max_steps=300)
            if len(subpath) <= 1:
                # either collision or stuck
                collisions += 1
                success = False
                break
            # measure distance traveled in this subpath
            for j in range(len(subpath)-1):
                dist = np.linalg.norm(
                    np.array(subpath[j+1].get_pose()[:2]) - np.array(subpath[j].get_pose()[:2])
                )
                total_dist += dist
                step_count += 1
        if success:
            successes += 1
            path_lengths.append(total_dist)
            times.append(step_count)

    # Handle empty arrays
    avg_path_length = np.mean(path_lengths) if len(path_lengths) > 0 else 0.0
    time_to_goal = np.mean(times) if len(times) > 0 else 0.0

    results = {
        'success_rate': successes / trials if trials > 0 else 0.0,
        'avg_collisions': collisions / trials if trials > 0 else 0.0,
        'avg_path_length': avg_path_length,
        'time_to_goal': time_to_goal
    }
    return results

def train_and_evaluate(env, robot, start_pose, goal_pose, episodes=250, prm_samples=250, trials=5):
    d = 3 if robot == "car" else 2
    G = PRM_star_multithreaded(env, robot, prm_samples, start_pose, goal_pose, d)
    global_path = uniform_cost_search(G, start_pose, goal_pose)

    rl_planner = RLPathPlanner(
        robot=robot,
        env=env,
        start=start_pose.get_pose(),
        goal=goal_pose.get_pose(),
        action_space=4,
        state_space=200 * 200 * 36
    )
    rl_planner.train_agent(episodes=episodes)
    metrics = evaluate_policy(env, robot, start_pose, goal_pose, rl_planner, trials=trials)

    # Debug print metrics
    print(f"Metrics for episodes={episodes}, prm_samples={prm_samples}: {metrics}")

    return metrics

def run_ablation_study(env, robot, start_pose, goal_pose):
    episodes_list = [50, 100, 200, 500]
    prm_samples_list = [10, 25, 50, 100, 250, 500]

    results_episodes = {}
    for ep in episodes_list:
        print(f"Running ablation for episodes={ep}")
        metrics = train_and_evaluate(env, robot, start_pose, goal_pose, episodes=ep, prm_samples=300)
        results_episodes[ep] = metrics

    results_prm = {}
    for ps in prm_samples_list:
        print(f"Running ablation for prm_samples={ps}")
        metrics = train_and_evaluate(env, robot, start_pose, goal_pose, episodes=500, prm_samples=ps)
        results_prm[ps] = metrics

    # Debug print the results before plotting
    print("Results by episodes:", results_episodes)
    print("Results by PRM samples:", results_prm)

    # Figure 1: Success Rate vs. Episodes
    plt.figure()
    x = episodes_list
    y = [results_episodes[e]['success_rate'] for e in episodes_list]
    plt.plot(x, y, marker='o')
    plt.title("Success Rate vs. RL Training Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Success Rate")
    plt.grid(True)
    plt.savefig("success_vs_episodes.png")

    # Figure 2: Collisions vs. PRM Samples
    plt.figure()
    x = prm_samples_list
    y = [results_prm[p]['avg_collisions'] for p in prm_samples_list]
    plt.plot(x, y, marker='o', color='red')
    plt.title("Avg Collisions vs. PRM* Samples")
    plt.xlabel("PRM Samples")
    plt.ylabel("Avg Collisions")
    plt.grid(True)
    plt.savefig("collisions_vs_prm_samples.png")

    # Figure 3: Path Length vs. Episodes
    plt.figure()
    x = episodes_list
    y = [results_episodes[e]['avg_path_length'] for e in episodes_list]
    plt.bar([str(e) for e in episodes_list], y, color='green')
    plt.title("Avg Path Length vs. Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Avg Path Length")
    plt.grid(True)
    plt.savefig("pathlength_vs_episodes.png")

    # Figure 4: Time-to-Goal vs. PRM Samples
    plt.figure()
    x = prm_samples_list
    y = [results_prm[p]['time_to_goal'] for p in prm_samples_list]
    plt.plot(x, y, marker='s', color='purple')
    plt.title("Time-to-Goal vs. PRM Samples")
    plt.xlabel("PRM Samples")
    plt.ylabel("Time-to-Goal (steps)")
    plt.grid(True)
    plt.savefig("time_vs_prm_samples.png")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation studies on integrated PRM* + RL")
    parser.add_argument("--robot", required=True, choices=["car", "arm"], help="Robot type")
    parser.add_argument("--map", required=True, help="Path to environment map")
    parser.add_argument("--start", nargs="+", type=float, required=True, help="Start pose (x y theta for car, x y for arm)")
    parser.add_argument("--goal", nargs="+", type=float, required=True, help="Goal pose (x y theta for car, x y for arm)")
    args = parser.parse_args()

    env = scene_from_file(args.map)

    if args.robot == "car":
        start_pose = Pose((args.start[0], args.start[1], args.start[2]))
        goal_pose = Pose((args.goal[0], args.goal[1], args.goal[2]))
    else:
        start_pose = Pose((args.start[0], args.start[1]))
        goal_pose = Pose((args.goal[0], args.goal[1]))

    run_ablation_study(env, args.robot, start_pose, goal_pose)
