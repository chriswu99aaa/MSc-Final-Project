import math
from os import path
import numpy as np
import sys
import argparse
from scipy.spatial import distance_matrix
import logging
from copy import copy

# 把当前文件所在目录（即 gpt_bo.py 所在目录）插到 sys.path 最前面
sys.path.insert(0, path.dirname(__file__))

try:
    from gpt_bo import select_next_node_v2 as select_next_node
except:
    from gpt_bo import select_next_node

class CountIterable:
    def __init__(self, iterable):
        self._iterable = list(iterable)
        self.counter   = 0

    def __iter__(self):
        for x in self._iterable:
            self.counter += 1
            yield x

    def __len__(self):
        return len(self._iterable)

    def __contains__(self, x):
        return x in self._iterable

    def copy(self):
        # 保证 copy() 后还能继续计数
        return self
    


def eval_heuristic(node_positions: np.ndarray) -> float:
    '''
    Generate solution for TSP problem using the GPT-generated heuristic algorithm.
    
    Parameters
    ----------
    node_positions : np.ndarray
        2D array of node positions of shape (problem_size, 2).
    
    Returns
    -------
    tour_length : float
        The length of the generated tour.
    expanded_nodes : int
        The number of nodes expanded during the search.
    '''
    problem_size = node_positions.shape[0]
    # calculate distance matrix
    dist_mat = distance_matrix(node_positions, node_positions)
    # set the starting node
    start_node = 0
    solution = [start_node]
    # init unvisited nodes
    unvisited = set(range(problem_size))
    # remove the starting node
    expanded_nodes = 0
    unvisited.remove(start_node)
    # run the heuristic
    for _ in range(problem_size - 1):
        expanded_nodes += 1
        next_node = select_next_node(
            current_node=solution[-1],
            destination_node=start_node,
            unvisited_nodes=copy(unvisited),
            distance_matrix=dist_mat.copy(),
        )
        solution.append(next_node)
        if next_node in unvisited:
            unvisited.remove(next_node)
        else:
            raise KeyError(f"Node {next_node} is already visited.")
    
    # calculate the length of the tour
    tour_length = 0
    for i in range(problem_size):
        tour_length += dist_mat[solution[i], solution[(i + 1) % problem_size]]
    return tour_length, expanded_nodes
    

if __name__ == '__main__':
    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    assert mood in ['train', 'val']

    basepath = path.join(path.dirname(__file__), "dataset")
    if not path.isfile(path.join(basepath, "train20_dataset.npy")): # Generate datasets if not exist
        from gen_inst import generate_datasets
        generate_datasets()
    
    if mood == 'train':
        dataset_path = path.join(basepath, f"train{problem_size}_dataset.npy")
        node_positions = np.load(dataset_path)
        n_instances = node_positions.shape[0]
        print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
        
        tour_lengths = []
        expanded_nodes = []
        for i in range(n_instances):
            tour_length, expanded_nodes = eval_heuristic(node_positions[i])
            print(f"[*] Instance {i}: {tour_length}, Expanded Nodes: {expanded_nodes}")
            tour_lengths.append(tour_length)
            expanded_nodes.append(expanded_nodes)
        
        print("[*] Average:")
        print(np.mean(tour_lengths))
        print(np.mean(expanded_nodes))
    
    else:
        for problem_size in [20, 50, 100, 200]:
            dataset_path = path.join(basepath, f"val{problem_size}_dataset.npy")
            logging.info(f"[*] Evaluating {dataset_path}")
            node_positions = np.load(dataset_path)
            n_instances = node_positions.shape[0]
            objs = []
            for i in range(n_instances):
                obj = eval_heuristic(node_positions[i])
                objs.append(obj)
                # print(f"[*] Average for {problem_size}: {obj}")
            print(f"[*] Average for {problem_size}: {np.mean(objs)}")