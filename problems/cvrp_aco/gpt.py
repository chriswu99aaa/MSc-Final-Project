import numpy as np

def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    n = len(distance_matrix)
    heuristics_matrix = np.zeros((n, n))
    decay_factor = 0.7  # Adjusted decay factor to reduce influence from distant nodes
    scaling_factor = 1.5  # Dynamic scaling factor based on demand

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                total_demand = demands[i] + demands[j]
                if total_demand <= capacity:
                    demand_to_distance_ratio = demands[i] / (distance_matrix[i][j] + 1e-6)
                    heuristics_matrix[i][j] = scaling_factor * demand_to_distance_ratio * np.exp(-decay_factor * distance_matrix[i][j])
                else:
                    heuristics_matrix[i][j] = -1  # Penalty for exceeding capacity

    return heuristics_matrix
