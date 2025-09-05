import numpy as np

def heuristics_v2(demand, capacity):
    n = len(demand)
    heuristics_matrix = np.zeros((n, n))
    penalty_scaling_exceed = 7.0  # Strong penalty for exceedance
    penalty_scaling_underutil = 2.5  # Moderate penalty for under-utilization
    base_factor = 1.5  # Base factor for contribution scaling

    for i in range(n):
        for j in range(n):
            if i != j:
                combined_size = demand[i] + demand[j]
                normalized_i = demand[i] / capacity
                normalized_j = demand[j] / capacity

                # Cubic compatibility score
                if combined_size <= capacity:
                    heuristics_matrix[i][j] = (normalized_i ** 3 * normalized_j ** 3) * base_factor * (1 + normalized_i + normalized_j)  
                else:
                    exceedance = combined_size - capacity
                    penalty_exceed = (exceedance ** penalty_scaling_exceed)  # Strong non-linear penalty for exceedance
                    heuristics_matrix[i][j] = -penalty_exceed  

                if combined_size < capacity:
                    underutilization = capacity - combined_size
                    underutil_penalty = (underutilization ** penalty_scaling_underutil) / (combined_size + 1)  # Moderate penalty
                    heuristics_matrix[i][j] += -underutil_penalty  

    return heuristics_matrix
