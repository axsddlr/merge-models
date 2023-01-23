def prune_checkpoint(old_state):
    print(f"Pruning Checkpoint")
    pruned_checkpoint = dict()
    print(f"Checkpoint Keys: {old_state.keys()}")
    for key in old_state.keys():
        if key != "optimizer_states":
            pruned_checkpoint[key] = old_state[key]
    else:
        print("Removing optimizer states from checkpoint")
    if "global_step" in old_state:
        print(f"This is global step {old_state['global_step']}.")
    state_key = None
    if 'state_dict' in old_state:
        state_key = 'state_dict'
    elif 'model_state' in old_state:
        state_key = 'model_state'
    # add more checks for other keys where the model's state might be stored

    if state_key and state_key in pruned_checkpoint:
        old_state = pruned_checkpoint[state_key].copy()
        new_state = dict()
        for key in old_state:
            new_state[key] = old_state[key].half()
        pruned_checkpoint[state_key] = new_state
    return pruned_checkpoint
