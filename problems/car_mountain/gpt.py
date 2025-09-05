def choose_action(pos, v, last_action):
    # Define constants for the strategy
    low_threshold = -1.0
    mid_threshold = -0.45
    high_threshold = 0.35
    velocity_threshold = 0.02
    recovery_factor = 0.5
    acceleration_gain = 1.2

    # Calculate adjusted action based on proximity and velocity
    if pos <= low_threshold:  # Very far left
        action = 2  # Full throttle to the right
    elif low_threshold < pos <= mid_threshold:  # Steep left slope
        if v < -0.05:  # Aggressively moving backwards
            action = 2  # Strong push to the right
        elif v < 0:  # Stationary or backward
            action = 0  # Prepare to move left
        else:  # Moving slightly right
            action = 1 if last_action == 0 else 2  # Dynamic recovery
    elif mid_threshold < pos < high_threshold:  # Middle range
        if v > velocity_threshold:  # Good speed
            action = 1 if (last_action == 0 and recovery_factor < 0.5) else 2  # Sustain/increase speed
        elif v > 0:  # Slightly moving forward
            action = 2 if recovery_factor < 0.5 else 1  # Accelerate with caution
        else:  # Slow or stationary
            action = 0  # Prepare to shift left
    else:  # Nearing the flag
        if abs(v) < velocity_threshold:  # Almost stationary
            action = 2  # Final push to the right
        elif v > 0.04:  # Decent speed
            action = 2 if last_action == 1 else 1  # Maintain or fine-tune speed
        else:  # Fine-tuning
            action = 1 if last_action == 2 else 2  # Alternate actions as needed

    return action
