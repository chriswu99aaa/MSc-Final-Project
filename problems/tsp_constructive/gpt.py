import random

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    next_node = None
    min_weighted_distance = float('inf')
    candidates = []

    total_distance = sum(distance_matrix[current_node][node] for node in unvisited_nodes)
    least_visited_count = {node: 0 for node in unvisited_nodes}

    for node in unvisited_nodes:
        distance = distance_matrix[current_node][node]
        connectivity_score = sum(distance_matrix[node][n] for n in unvisited_nodes) / len(unvisited_nodes)
        visit_score = least_visited_count[node]

        weighted_distance = (distance * 0.6) + (0.4 * (total_distance / len(unvisited_nodes))) + (0.3 * visit_score) - (0.5 * connectivity_score) + (distance ** 2 * 0.1)

        if weighted_distance < min_weighted_distance:
            min_weighted_distance = weighted_distance
            candidates = [node]
        elif weighted_distance == min_weighted_distance:
            candidates.append(node)

    if candidates:
        next_node = random.choice(candidates)

    return next_node
