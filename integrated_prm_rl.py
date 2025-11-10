import argparse
import math
import timeit
from collections import defaultdict
import numpy as np
from utils import *
from component_1 import *
from ao_planner import *

# PRM* Code
import concurrent.futures
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
    # parallelizing tasks
    for future in concurrent.futures.as_completed(futures):
        node, valid_neighbors = future.result()
        for neighbor in valid_neighbors:
            dist = distance(robot, node, neighbor)
            G[node].add((neighbor, dist))
            G[neighbor].add((node, dist))

    executor.shutdown()
    return G



# RL Path Planner 
class RLPathPlanner:
    def __init__(self, robot, env, start, goal, action_space, state_space):
        self.robot = robot
        self.env = env
        self.start = Pose(start)
        self.goal = Pose(goal)
        self.action_space = action_space
        self.state_space = state_space
        self.policy = None  

    def train_agent(self, episodes=500, max_steps=500):  
        q_table = np.zeros((self.state_space, self.action_space))
        learning_rate = 0.05
        discount_factor = 0.99
        
        exploration_rate = 1.0
        min_exploration_rate = 0.2
        exploration_decay = 0.0003  # lower number causes more exploration

        for episode in range(episodes):
            curr = self.start
            done = False
            steps = 0
            visited = set()

            while not done and steps < max_steps:
                if steps % 5 == 0:
                    print(f"Episode {episode + 1}, Step {steps}: Pose {curr.get_pose()}")

                # epsilon-greedy
                if np.random.uniform(0, 1) > exploration_rate:
                    action = np.argmax(q_table[self.pose_to_idx(curr)])
                else:
                    action = np.random.choice(range(self.action_space))

                next_pose = self.get_next_pose(curr, action)

                # collision check
                if not collision_free_path(self.env, self.robot, curr, next_pose):
                    reward = -50
                    currstate_idx = self.pose_to_idx(curr)
                    q_table[currstate_idx, action] += learning_rate * (
                        reward - q_table[currstate_idx, action]
                    )
                    done = True
                    print("collided with object")
                else:
                    # partial closeness reward
                    dist_to_goal = np.linalg.norm(
                        np.array(curr.get_pose()[:2]) - np.array(self.goal.get_pose()[:2])
                    )
                    new_dist = np.linalg.norm(
                        np.array(next_pose.get_pose()[:2]) - np.array(self.goal.get_pose()[:2])
                    )
                    # base reward
                    reward = -0.1

                    # partial closeness reward (without immediate termination)
                    reward += 5.0 / (1 + new_dist) 

                    # final threshold for actual termination
                    if new_dist < 0.1: 
                        reward += 100
                        done = True
                    else:
                        # boundary penalty
                        if abs(next_pose.get_pose()[0]) == 10 or abs(next_pose.get_pose()[1]) == 10:
                            reward -= 5
                        # penalty for revisiting states
                        if next_pose.get_pose() in visited:
                            reward -= 1
                        
                        reward += (dist_to_goal - new_dist) * 10

                    currstate_idx = self.pose_to_idx(curr)
                    next_state_idx = self.pose_to_idx(next_pose)
                    q_table[currstate_idx, action] += learning_rate * (
                        reward + discount_factor * np.max(q_table[next_state_idx]) - q_table[currstate_idx, action] )

                    visited.add(curr.get_pose())
                    curr = next_pose
                    steps += 1

            print(f"Episode {episode + 1}/{episodes} completed with {steps} steps.")
            # slower exploration decay
            exploration_rate = max(
                min_exploration_rate,
                exploration_rate * np.exp(-exploration_decay * episode),
            )

        self.policy = q_table

    def get_next_pose(self, pose, action):
        pose_values = pose.get_pose()
        step_size = 0.5

        if self.robot == "car":
            if action == 0:  # Up
                new_x, new_y, theta = pose_values[0], pose_values[1] + step_size, pose_values[2]
            elif action == 1:  # Down
                new_x, new_y, theta = pose_values[0], pose_values[1] - step_size, pose_values[2]
            elif action == 2:  # Left
                new_x, new_y, theta = pose_values[0] - step_size, pose_values[1], pose_values[2]
            elif action == 3:  # Right
                new_x, new_y, theta = pose_values[0] + step_size, pose_values[1], pose_values[2]
            else:
                return pose

            new_x = max(-10, min(10, new_x))
            new_y = max(-10, min(10, new_y))
            return Pose((new_x, new_y, theta))

        return pose

    def pose_to_idx(self, pose):
        pose_values = pose.get_pose()
        x_index = int((pose_values[0] + 10) * 10)
        y_index = int((pose_values[1] + 10) * 10)
        theta_index = int((pose_values[2] / (2 * np.pi)) * 36) % 36
        return x_index * 360 + y_index * 36 + theta_index

    def plan_path(self, max_steps=500):
        if self.policy is None:
            raise ValueError("need to train agent")

        path = [self.start]
        curr = self.start
        visited = []
        steps = 0

        while curr != self.goal and steps < max_steps:
            visited.append(curr.get_pose())
            if len(visited) > 10:
                visited.pop(0)

            action = np.argmax(self.policy[self.pose_to_idx(curr)])
            next_pose = self.get_next_pose(curr, action)

            if not collision_free_path(self.env, self.robot, curr, next_pose):
                print(f"collision at: {next_pose.get_pose()}")
                break

            if visited.count(next_pose.get_pose()) > 3:
                print(f"oscillation at step {steps}.")
                break

            path.append(next_pose)
            curr = next_pose
            steps += 1

        if steps >= max_steps:
            print("path exceeded max steps")

        return path

    def nav_subgoal(self, start, subgoal_pose, max_steps=200):
        if self.policy is None:
            raise ValueError("need to train first ")

        path_segment = [start]
        curr = start
        visited = []
        steps = 0

        while steps < max_steps:
            dist = np.linalg.norm(np.array(curr.get_pose()[:2]) - np.array(subgoal_pose.get_pose()[:2]))
            if dist < 0.5:
                break

            visited.append(curr.get_pose())
            if len(visited) > 10:
                visited.pop(0)

            action = np.argmax(self.policy[self.pose_to_idx(curr)])
            next_pose = self.get_next_pose(curr, action)

            if not collision_free_path(self.env, self.robot, curr, next_pose):
                print("local collision.")
                break

            if visited.count(next_pose.get_pose()) > 3:
                print("oscillation found in subgoal .")
                break

            path_segment.append(next_pose)
            curr = next_pose
            steps += 1

        return path_segment



def main():
    parser = argparse.ArgumentParser(description="QLearning Path Planning PRM* Informed")
    parser.add_argument("--robot", required=True, choices=["car", "arm"], help="Robot type")
    parser.add_argument("--map", required=True, help="Path to environment map")
    parser.add_argument("--start", nargs="+", type=float, required=True, help="Start pose")
    parser.add_argument("--goal", nargs="+", type=float, required=True, help="Goal pose")
    args = parser.parse_args()

    env = scene_from_file(args.map)

    if args.robot == "car":
        start = Pose((args.start[0], args.start[1], args.start[2]))
        goal = Pose((args.goal[0], args.goal[1], args.goal[2]))
        d = 3
    else:
        start = Pose((args.start[0], args.start[1]))
        goal = Pose((args.goal[0], args.goal[1]))
        d = 2

    print("PRM* roadmap:")
    G = PRM_star_multithreaded(env, args.robot, 25, start, goal, d)
    print("uniform cost search for global path:")
    global_path = uniform_cost_search(G, start, goal)
    print("global path (PRM*):", global_path)
    if len(global_path) < 2:
        print("no path found by PRM*")
        return


    print("\nTrain Q-Learning agent:")
    rl_planner = RLPathPlanner(
        robot=args.robot,
        env=env,
        start=start.get_pose(),
        goal=goal.get_pose(),
        action_space=4,
        state_space=200 * 200 * 36
    )
    rl_planner.train_agent(episodes=200)  
    print("training finished\n")

    if len(global_path) < 2:
        print("no path found by PRM*")
        return

    refined_path = [global_path[0]]
    for i in range(len(global_path)-1):
        local_start = refined_path[-1]
        local_subgoal = global_path[i+1]
        print(f"navigating from {local_start.get_pose()} to {local_subgoal.get_pose()} .")

        subpath = rl_planner.nav_subgoal(local_start, local_subgoal, max_steps=300)
        if len(subpath) == 1:
            print(" Breaking.")
            break
        refined_path.extend(subpath[1:])  

    print("\npath using qLearning for local steps:")
    for p in refined_path:
        print(p.get_pose())

    print("showing final path")
    robot_visualization(env, args.robot, start, goal, G, refined_path)


if __name__ == "__main__":
    main()
