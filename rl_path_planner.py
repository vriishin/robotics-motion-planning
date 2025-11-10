import numpy as np
from utils import collision_free_path, robot_visualization, Pose
from component_1 import scene_from_file


class RLPathPlanner:
    def __init__(self, robot, env, start, goal, action_space, state_space):
        self.robot = robot
        self.env = env
        self.start = Pose(start)
        self.goal = Pose(goal)
        self.action_space = action_space
        self.state_space = state_space
        self.policy = None  

    def train_agent(self, episodes=1000, max_steps=500):
        q_table = np.zeros((self.state_space, self.action_space))
        learning_rate = 0.05
        discount_factor = 0.99
        exploration_rate = 1.0
        min_exploration_rate = 0.2
        exploration_decay = 0.0005  

        for episode in range(episodes):
            current_pose = self.start
            done = False
            steps = 0
            visited_states = set()

            while not done and steps < max_steps:
                if steps % 5 == 0:
                    print(f"Episode {episode + 1}, Step {steps}: Pose {current_pose.get_pose()}")

                # Choose action using epsilon-greedy policy
                if np.random.uniform(0, 1) > exploration_rate:
                    action = np.argmax(q_table[self.pose_to_index(current_pose)])
                else:
                    action = np.random.choice(range(self.action_space))

                next_pose = self.get_next_pose(current_pose, action)

                # Check for collision before updating rewards
                if not collision_free_path(self.env, self.robot, current_pose, next_pose):
                    # Large negative reward for collision and terminate episode
                    reward = -50
                    current_state_idx = self.pose_to_index(current_pose)
                    q_table[current_state_idx, action] += learning_rate * (
                        reward - q_table[current_state_idx, action]
                    )
                    done = True
                    print("Collision encountered during training.")
                else:
                    reward = -0.1  # Default small penalty for each step

                    # Distance-based reward shaping
                    distance_to_goal = np.linalg.norm(
                        np.array(current_pose.get_pose()[:2]) - np.array(self.goal.get_pose()[:2])
                    )
                    new_distance_to_goal = np.linalg.norm(
                        np.array(next_pose.get_pose()[:2]) - np.array(self.goal.get_pose()[:2])
                    )

                    # Goal check
                    if next_pose.get_pose() == self.goal.get_pose():
                        reward = 100  # Goal reached
                        done = True
                    else:
                        # Additional penalties and rewards
                        # Boundary penalty
                        if abs(next_pose.get_pose()[0]) == 10 or abs(next_pose.get_pose()[1]) == 10:
                            reward -= 5
                        # Revisiting states penalty
                        if next_pose.get_pose() in visited_states:
                            reward -= 1
                        # Incremental progress toward goal
                        reward += (distance_to_goal - new_distance_to_goal) * 10

                    # Update Q-table
                    current_state_idx = self.pose_to_index(current_pose)
                    next_state_idx = self.pose_to_index(next_pose)
                    q_table[current_state_idx, action] += learning_rate * (
                        reward
                        + discount_factor * np.max(q_table[next_state_idx])
                        - q_table[current_state_idx, action]
                    )

                    # Update state and step count
                    visited_states.add(current_pose.get_pose())
                    current_pose = next_pose
                    steps += 1

            print(f"Episode {episode + 1}/{episodes} completed with {steps} steps.")

            # Decay exploration rate
            exploration_rate = max(
                min_exploration_rate,
                exploration_rate * np.exp(-exploration_decay * episode),
            )

        self.policy = q_table

    def get_next_pose(self, pose, action):
        pose_values = pose.get_pose()
        step_size = 0.5  # Fixed step size for simplicity

        # Determine the new pose based on the action
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
                return pose  # Invalid action, return current pose

            # Enforce boundaries for the map
            new_x = max(-10, min(10, new_x))
            new_y = max(-10, min(10, new_y))

            return Pose((new_x, new_y, theta))

        # Extend for arm logic if needed
        return pose

    def pose_to_index(self, pose):
        pose_values = pose.get_pose()
        # Discretize pose values
        x_index = int((pose_values[0] + 10) * 10)  # Adjust based on environment bounds
        y_index = int((pose_values[1] + 10) * 10)
        theta_index = int((pose_values[2] / (2 * np.pi)) * 36) % 36  # Discretize theta
        return x_index * 360 + y_index * 36 + theta_index

    def plan_path(self, max_steps=500):
        if self.policy is None:
            raise ValueError("Policy not trained. Call train_agent() first.")

        path = [self.start]
        current_pose = self.start
        visited_states = []  # Maintain history of visited states
        steps = 0

        while current_pose != self.goal and steps < max_steps:
            visited_states.append(current_pose.get_pose())
            if len(visited_states) > 10:  # Limit history size
                visited_states.pop(0)

            action = np.argmax(self.policy[self.pose_to_index(current_pose)])
            next_pose = self.get_next_pose(current_pose, action)

            # Check collision before proceeding
            if not collision_free_path(self.env, self.robot, current_pose, next_pose):
                print(f"Collision detected at pose: {next_pose.get_pose()}")
                break

            # Check for oscillations
            if visited_states.count(next_pose.get_pose()) > 3:
                print(f"Oscillation detected at step {steps}, breaking loop.")
                break

            path.append(next_pose)
            current_pose = next_pose
            steps += 1

        if steps >= max_steps:
            print("Path planning exceeded the maximum allowed steps.")

        return path


def main():
    # Parse command-line arguments
    from argparse import ArgumentParser

    parser = ArgumentParser(description="RL Path Planner")
    parser.add_argument("--robot", required=True, choices=["car", "arm"], help="Robot type")
    parser.add_argument("--map", required=True, help="Path to environment map")
    parser.add_argument("--start", nargs="+", type=float, required=True, help="Start pose")
    parser.add_argument("--goal", nargs="+", type=float, required=True, help="Goal pose")
    args = parser.parse_args()

    env = scene_from_file(args.map)
    planner = RLPathPlanner(
        robot=args.robot,
        env=env,
        start=tuple(args.start),
        goal=tuple(args.goal),
        action_space=4,
        state_space=200 * 200 * 36,  # This is still very large, consider reducing or using approximation methods
    )

    print("Training RL Agent...")
    planner.train_agent(episodes=100)
    print("Training Complete!")

    print("Planning Path...")
    path = planner.plan_path()
    print("Planned Path:", path)

    print("Visualizing Path...")
    robot_visualization(env, args.robot, Pose(args.start), Pose(args.goal), {}, path)


if __name__ == "__main__":
    main()
