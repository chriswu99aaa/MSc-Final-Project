import numpy as np

def select_next_item(remaining_capacity, weights, values):
    ratios = values / weights
    valid_items = np.where(weights <= remaining_capacity)[0]
    if valid_items.size == 0:
        return None
    next_item = valid_items[np.argmax(ratios[valid_items])]
    return next_item
