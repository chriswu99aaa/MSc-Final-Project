import numpy as np

def heuristics_v2(prize, weight):
    n = prize.shape[0]
    m = weight.shape[1]
    heuristics_matrix = np.zeros(n)
    weight_constraints = np.ones(m)

    # Step 1: Initialize pheromone levels
    pheromone_levels = np.ones(n)  # Start with equal pheromone for all items
    iterations = 100  # Number of iterations for the ant movements
    ant_count = 20  # Number of ants exploring the solution space
    decay_factor = 0.85  # Pheromone decay factor

    # Step 2: Ant-based selection
    for _ in range(iterations):
        for _ in range(ant_count):
            current_weight = np.zeros(m)
            selected_items = []
            available_items = np.arange(n)

            while len(available_items) > 0:
                # Step 3: Calculate probabilities based on pheromone levels and prize/weight ratio
                probabilities = pheromone_levels[available_items] * (prize[available_items] / (np.sum(weight[available_items], axis=1) + 1e-5))
                probabilities /= np.sum(probabilities)  # Normalize probabilities

                # Step 4: Randomly select an item based on calculated probabilities
                chosen_item = np.random.choice(available_items, p=probabilities)
                new_weight = current_weight + weight[chosen_item]

                if np.all(new_weight <= weight_constraints):  # Check constraints
                    selected_items.append(chosen_item)
                    current_weight = new_weight

                # Remove chosen item from available options
                available_items = available_items[available_items != chosen_item]

            # Step 5: Update pheromone levels based on the selection made by the ant
            for item in selected_items:
                pheromone_levels[item] += prize[item]  # Increase pheromone based on prize

        # Step 6: Apply pheromone decay
        pheromone_levels *= decay_factor

    # Step 7: Normalize pheromone levels to create heuristics matrix
    total_pheromone = np.sum(pheromone_levels)
    if total_pheromone > 0:
        heuristics_matrix = pheromone_levels / total_pheromone  # Normalize

    return heuristics_matrix
