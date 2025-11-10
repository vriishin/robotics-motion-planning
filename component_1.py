from utils import *
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import random
import pickle

def regen_enviornments():
    # scene_to_file(generate_enviornment(5),  "1.txt")
    # scene_to_file(generate_enviornment(10), "2.txt")
    # scene_to_file(generate_enviornment(20), "3.txt")
    # scene_to_file(generate_enviornment(40), "4.txt")
    # scene_to_file(generate_enviornment(80), "5.txt")
    scene_to_file(generate_enviornment(40), "6.txt")


def generate_enviornment(number_of_obstracles: int) -> Enviornment:
    env = Enviornment()

    for _ in range(int(number_of_obstracles/2)):
        x = random.uniform(env.size[0][0], env.size[0][1])
        y = random.uniform(env.size[1][0], env.size[1][1])
        t = random.uniform(0, 2*np.pi)
        env.obstacles.append(Rect(Point(x, y), 0.5, 2, t))

    for _ in range(int(number_of_obstracles/2)):
        x = random.uniform(env.size[0][0], env.size[0][1])
        y = random.uniform(env.size[1][0], env.size[1][1])
        env.obstacles.append(Circ(Point(x,y), .75))
       
        
        
    
    return env

def scene_to_file(env: Enviornment, filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(env, f)

def scene_from_file(filename: str) -> Enviornment:
    with open(filename, 'rb') as f:
        env = pickle.load(f)
    return env

def visualize_scene(env: Enviornment):
    fig, axis = plt.subplots()
    
    axis.set_xlim(env.size[0][0], env.size[0][1])
    axis.set_ylim(env.size[1][0], env.size[1][1])

    for obs in env.get_obstacles():
        if isinstance(obs,Rect):
            obstacle = create_rec(obs, fc="cornflowerblue")
            axis.add_patch(obstacle)
        
        if isinstance(obs,Circ):
            obstacle = create_circ(obs, fc="cornflowerblue")
            axis.add_patch(obstacle)

    plt.show()

def main():
    #regen_enviornments()
    visualize_scene(scene_from_file(f"1.txt"))
    visualize_scene(scene_from_file(f"3.txt"))
    visualize_scene(scene_from_file(f"5.txt"))
    visualize_scene(scene_from_file(f"6.txt"))
    # for i in range(6, 7):
    #     visualize_scene(scene_from_file(f"{i}.txt"))
    # pass
if __name__ == '__main__':
    main()
