import argparse
from component_1 import *
# from hw3utils import *
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='Nearest Neighbors',
                    description='finds the nearest number of neighbors')
    
    parser.add_argument('--robot', type=str, required=True)       
    parser.add_argument('--target', type=float, nargs='+', required=True)      
    parser.add_argument('--k', type=int, required=True, help='nearest neighbors to find')
    parser.add_argument('--configs', type=str, required=True)

    return parser.parse_args()

def load_configs(filename):
    # assuming line in the file is new config, each thing is spaceseparated values
    with open(filename, 'r') as file:
        configs = [list(map(float, line.strip().split())) for line in file.readlines()]
    return configs

def distance(target, config):
    return np.sqrt(np.sum((np.array(target) - np.array(config)) ** 2 ))

def get_neighbors(target, configs, k):
    distances = []
    for config in configs:
        d = distance(target, config)
        distances.append((config, d))
    
    
    distances.sort(key=lambda x: x[1])
    
    return distances[:k]


def main():
    args = parse_arguments()
    print(f"Robot: {args.robot}, Target: {args.target}, k: {args.k}, Configs file: {args.configs}")

    args = parse_arguments()
    configs = load_configs(args.configs)
    neighbors = get_neighbors(args.target, configs, args.k)
    
    print("The nearest neighbors are:")
    for neighbor, dist in neighbors:
        print(f"Configuration: {neighbor}, Distance: {dist:.4f}")

if __name__ == "__main__":
    main()




