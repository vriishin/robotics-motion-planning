import pickle
import numpy as np
import random
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import matplotlib.animation as animation 

TWO_PI = 2*np.pi

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y
    
    def get_xy(self):
        return (self.x, self.y)
    
    def __add__(self, other):
        return Point(self.x + other.get_x(), self.y + other.get_y())
    
    def __sub__(self, other):
        return np.array([self.x - other.get_x(), self.y - other.get_y()])
    
    def __str__(self):
        return f"x: {self.x}, y: {self.y}"
    
    def __repr__(self):
        return str(self)

class Rect:
    def __init__(self, xy: Point, w: float, h: float, theta: float):
        self.xy = xy
        self.theta = theta
        self.w = w
        self.h = h

    def get_x(self):
        return self.xy.get_x()

    def get_y(self):
        return self.xy.get_y()
    
    def get_xy(self):
        return (self.get_x(), self.get_y())
    
    def set_xy(self, xy):
        self.xy = xy
    
    def get_point(self):
        return self.xy

    def get_theta(self):
        return self.theta
    
    def set_theta(self, theta):
        self.theta = theta

    def get_xyt(self):
        return self.get_point().get_xy() + (self.theta,)
    
    def get_w(self):
        return self.w
    
    def get_h(self):
        return self.h
    
    def get_rect(self, color) -> Rectangle:
        return create_rec(self, fc=color)
    
    def distance_to_edge(self, point: Point):
        px, py = point.get_x() - self.get_x(), point.get_y() - self.get_y()
        
        cos_theta = np.cos(-self.theta)
        sin_theta = np.sin(-self.theta)
        px_rot = cos_theta * px - sin_theta * py
        py_rot = sin_theta * px + cos_theta * py
        
        dx = max(abs(px_rot) - self.w / 2, 0)
        dy = max(abs(py_rot) - self.h / 2, 0)
        
        distance = np.sqrt(dx**2 + dy**2)
        
        if dx == 0 and dy == 0:
            distance = min(self.w / 2 - abs(px_rot), self.h / 2 - abs(py_rot))
        
        return distance

class Pose:
    def __init__(self, pose):
        self.pose = pose

    def get_pose(self):
        return self.pose
    
    def get_rect(self, color) -> Rectangle:
        return create_rec(self, fc=color)

    def __str__(self):
        return str(self.pose)
    
    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return str(self) == str(other)
    
    def __hash__(self):
        return hash(str(self))

class Enviornment:
    def __init__(self):
        self.obstacles = []
        self.size = [[-10, 10], [-10, 10]]

    def get_obstacles(self):
        return self.obstacles
    
    def get_size(self):
        return self.size

def angle_distance(angle1, angle2):
    # Calculate the absolute difference
    diff = abs(angle1%TWO_PI - angle2%TWO_PI)
    # Return the smaller angle distance by considering wrap-around
    return min(diff, TWO_PI - diff)

def distance(robot, pose1, pose2) -> float:
    pose1 = pose1.get_pose()
    pose2 = pose2.get_pose()
    if robot == "car":
        return ((pose1[0]-pose2[0])**2 + (pose1[1]-pose2[1])**2 + angle_distance(pose1[2], pose2[2])**2) ** 0.5
    if robot == "arm":
        return (angle_distance(pose1[0], pose2[0])**2 + angle_distance(pose1[1], pose2[1])**2) ** 0.5

def rotate(x, y, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pos = np.array([x, y])

    return R @ pos.T

def create_rec(rect: Rect, fc="black") -> Rectangle:
    x, y = rect.get_xy()
    w = rect.get_w()
    h = rect.get_h()
    t = rect.get_theta()
    offset = rotate(w / 2, h / 2, t)
    rec = Rectangle((x-offset[0], y-offset[1]), w, h, fc=fc, zorder=10)
    rec.set_angle(t*180/np.pi)

    return rec

def rotate_point(p: Point, origin: Point, angle: float) -> Point:
    s = np.sin(angle)
    c = np.cos(angle)
    
    # Translate point back to origin:
    p = Point(p.x - origin.x, p.y - origin.y)
    
    # Rotate point
    x_new = p.x * c - p.y * s
    y_new = p.x * s + p.y * c
    
    # Translate point back:
    p = Point(x_new + origin.x, y_new + origin.y)
    
    return p

def get_corners(pose: Rect):
    cx, cy = pose.xy.x, pose.xy.y
    w_half, h_half = pose.w / 2, pose.h / 2
    
    # Unrotated corners
    corners = [
        Point(cx - w_half, cy - h_half),  # Top-left
        Point(cx + w_half, cy - h_half),  # Top-right
        Point(cx + w_half, cy + h_half),  # Bottom-right
        Point(cx - w_half, cy + h_half)   # Bottom-left
    ]
    
    # Rotate each corner around the center of the rectangle
    return [rotate_point(corner, pose.xy, pose.theta) for corner in corners]

def get_axes(corners):
    """Return the axes (normals) to the edges of the polygon."""
    axes = []
    # rectangle axis are parallel so can only do half
    for i in range(len(corners)//2):
        p1 = corners[i]
        p2 = corners[(i + 1) % len(corners)]  # Next corner (wraps around)
        edge = Point(p2.x - p1.x, p2.y - p1.y)
        # Normal to the edge
        normal = Point(-edge.y, edge.x)
        # Normalize the normal vector
        length = np.sqrt(normal.x ** 2 + normal.y ** 2)
        if length != 0:
            normal.x /= length
            normal.y /= length
        axes.append(normal)
    return axes

def project_polygon(axis: Point, corners):
    min_proj = max_proj = corners[0].x * axis.x + corners[0].y * axis.y
    for corner in corners[1:]:
        proj = corner.x * axis.x + corner.y * axis.y
        if proj < min_proj:
            min_proj = proj
        if proj > max_proj:
            max_proj = proj
    return min_proj, max_proj

def overlap(min_a, max_a, min_b, max_b):
    return max_a >= min_b and max_b >= min_a

def separating_axis_theorem(rect1: Rect, rect2: Rect):
    # Get the corners of both rectangles
    corners1 = get_corners(rect1)
    corners2 = get_corners(rect2)
    
    # Get the axes (normals) for both rectangles
    axes = get_axes(corners1) + get_axes(corners2)
    
    # Check for separation on each axis
    for axis in axes:
        min1, max1 = project_polygon(axis, corners1)
        min2, max2 = project_polygon(axis, corners2)
        if not overlap(min1, max1, min2, max2):
            return False  # Separating axis found, no collision
    
    return True  # No separating axis, collision detected

def get_collisions(env: Enviornment, rect: Rect):
    collisions = set()
    for obstacle in env.get_obstacles():
        # if car intersects obstacle
        if separating_axis_theorem(obstacle, rect):
            collisions.add(obstacle)

    return collisions

def collision_free(env: Enviornment, rect: Rect) -> bool:
    for obstacle in env.get_obstacles():
        if separating_axis_theorem(obstacle, rect):
            return False
    return True

def rotate(x, y, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pos = np.array([x, y])

    return R @ pos.T

def create_rec(rect: Rect, fc="black") -> Rectangle:
    x, y = rect.get_xy()
    w = rect.get_w()
    h = rect.get_h()
    t = rect.get_theta()
    offset = rotate(w / 2, h / 2, t)
    rec = Rectangle((x-offset[0], y-offset[1]), w, h, fc=fc, zorder=10)
    rec.set_angle(t*180/np.pi)

    return rec

def car_pose_to_rect(pose: Pose, w, h):
    return Rect(Point(pose.get_pose()[0], pose.get_pose()[1]), w, h, pose.get_pose()[2])

def arm_pose_to_rect(pose: Pose, sizes):
    previous_rect = Rect(Point(0, 0), 0, 0, 0)
    rects = []
    for i, (w, h) in enumerate(sizes):
        if pose is None:
            t = random.uniform(0, 2*np.pi)
        else:
            t = pose.get_pose()[i]
        # get center of current link with respect to origin
        cx, cy = rotate(0, h/2, previous_rect.get_theta() + t)
        # get the remaining half of the previous link
        px, py = rotate(0, previous_rect.get_h()/2, previous_rect.get_theta())

        # position is previous link plus remaning half (since we want the top edge coordinates) plus center of new link
        p = Rect(previous_rect.get_point() + Point(px, py) + Point(cx, cy), w, h, t + previous_rect.get_theta())
        rects.append(p)
        previous_rect = p
            
    return rects

def scene_from_file(filename: str) -> Enviornment:
    with open(filename, 'rb') as f:
        env = pickle.load(f)

    return env

def animate_robot(env: Enviornment, robot: str, path, fig, axis, patches, filename):
    # Initialize the plot by setting the robot to the first position in the path
    def init():
        for rect in env.get_obstacles():
            obstacle = create_rec(rect, fc="cornflowerblue")
            axis.add_patch(obstacle)

        if robot == "car":
            x_coords = [pose.get_pose()[0] for pose in path]
            y_coords = [pose.get_pose()[1] for pose in path]
            plt.plot(x_coords, y_coords, color='cyan', zorder=3, linewidth=1.2)
            
        if robot == "arm":
            link1_x_coords = []
            link1_y_coords = []
            link2_x_coords = []
            link2_y_coords = []

            for pose in path:
                # Get the rectangles representing the two links of the arm at this pose
                link_rects = [rect for rect in arm_pose_to_rect(pose, [[0.2, 2], [0.2, 1.5]])]
                
                # Extract the centers of the two links
                link1_x_coords.append(link_rects[0].get_x())
                link1_y_coords.append(link_rects[0].get_y())
                link2_x_coords.append(link_rects[1].get_x())
                link2_y_coords.append(link_rects[1].get_y())

            # Now we plot the path for both links
            plt.plot(link1_x_coords, link1_y_coords, color='cyan', zorder=3, linewidth=1.2, label='Link 1 Path')
            plt.plot(link2_x_coords, link2_y_coords, color='cyan', zorder=3, linewidth=1.2, label='Link 2 Path')


        return list()

    # Function to update the robot's position for each frame
    def animate(i):
        pose = path[i]
        
        # Clear the previous robot's patches
        for patch in patches.values():
            patch.remove()
        patches.clear()

        # Draw the robot at the new pose
        if robot == "car":
            rects = [car_pose_to_rect(pose, 0.5, 0.3).get_rect("dodgerblue")]
        elif robot == "arm":
            rects = [rect.get_rect("dodgerblue") for rect in arm_pose_to_rect(pose, [[0.2, 2], [0.2, 1.5]])]
        
        for rect in rects:
            axis.add_patch(rect)

            patches[rect] = rect

        return list(patches.values())

    # Create the animation object
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(path), interval=50, blit=True)
    ani.save(filename, writer='ffmpeg')
    plt.show()

def robot_animation(env: Enviornment, robot: str, start: Pose, goal: Pose, path, output_file="video.mp4"):
    fig, axis = plt.subplots()
    
    env_size = env.get_size()
    axis.set_xlim(env_size[0][0], env_size[0][1])
    axis.set_ylim(env_size[1][0], env_size[1][1])

    patches = dict()

    if robot == "car":
        start_rects = [car_pose_to_rect(start, 0.5, 0.3).get_rect("magenta")]
        goal_rects = [car_pose_to_rect(goal, 0.5, 0.3).get_rect("orangered")]
    elif robot == "arm":
        start_rects = [rect.get_rect("magenta") for rect in arm_pose_to_rect(start, [[0.2, 2], [0.2, 1.5]])]
        goal_rects = [rect.get_rect("orangered") for rect in arm_pose_to_rect(goal, [[0.2, 2], [0.2, 1.5]])]
    
    [axis.add_patch(start_rect) for start_rect in start_rects]
    [axis.add_patch(goal_rect) for goal_rect in goal_rects]

    for rect in env.get_obstacles():
        obstacle = create_rec(rect, fc="cornflowerblue")
        axis.add_patch(obstacle)
        patches[rect] = obstacle

    animate_robot(env, robot, path, fig, axis, patches, output_file)

    
def robot_visualization(env: Enviornment, robot: str, start: Pose, goal: Pose, path, output_file="fig.jpg"):
    fig, axis = plt.subplots()
    
    env_size = env.get_size()
    axis.set_xlim(env_size[0][0], env_size[0][1])
    axis.set_ylim(env_size[1][0], env_size[1][1])

    patches = dict()


    if robot == "car":
        start_rects = [car_pose_to_rect(start, 0.5, 0.3).get_rect("magenta")]
        goal_rects = [car_pose_to_rect(goal, 0.5, 0.3).get_rect("orangered")]

    if robot == "arm":
        start_rects = [rect.get_rect("magenta") for rect in arm_pose_to_rect(start, [[0.2, 2], [0.2, 1.5]])]
        goal_rects = [rect.get_rect("orangered") for rect in arm_pose_to_rect(goal, [[0.2, 2], [0.2, 1.5]])]
        
    [axis.add_patch(start_rect) for start_rect in start_rects]
    [axis.add_patch(goal_rect) for goal_rect in goal_rects]


    for rect in env.get_obstacles():
        obstacle = create_rec(rect, fc="cornflowerblue")
        axis.add_patch(obstacle)
        patches[rect] = obstacle
        
    if robot == "car":
        x_coords = [pose.get_pose()[0] for pose in path]
        y_coords = [pose.get_pose()[1] for pose in path]
        plt.plot(x_coords, y_coords, color='cyan', zorder=3, linewidth=1.2)
        
    if robot == "arm":
        link1_x_coords = []
        link1_y_coords = []
        link2_x_coords = []
        link2_y_coords = []

        for pose in path:
            # Get the rectangles representing the two links of the arm at this pose
            link_rects = [rect for rect in arm_pose_to_rect(pose, [[0.2, 2], [0.2, 1.5]])]
            
            # Extract the centers of the two links
            link1_x_coords.append(link_rects[0].get_x())
            link1_y_coords.append(link_rects[0].get_y())
            link2_x_coords.append(link_rects[1].get_x())
            link2_y_coords.append(link_rects[1].get_y())

        # Now we plot the path for both links
        plt.plot(link1_x_coords, link1_y_coords, color='cyan', zorder=3, linewidth=1.2, label='Link 1 Path')
        plt.plot(link2_x_coords, link2_y_coords, color='cyan', zorder=3, linewidth=1.2, label='Link 2 Path')


    plt.savefig(output_file)
    #plt.show(block=True)
