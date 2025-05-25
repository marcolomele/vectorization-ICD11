#%% 
import random

people = [1, 2, 5, 10]
best_cost = 99999999
test_runs = 1e7
best_sequence = []

while test_runs > 0:
    test_runs -= 1
    run_cost = 0
    run_sequence = []

    left_side = people.copy()
    right_side = []

    while left_side:
        # pick two people at random
        guide = random.choice(left_side)
        left_side.remove(guide)
        tourist = random.choice(left_side)
        left_side.remove(tourist)
        
        # record pairs crossing bridge
        run_sequence.append((guide, tourist))

        # move them to the other side of the bridge
        right_side.append(guide)
        right_side.append(tourist)

        run_cost += max(guide, tourist)
        
        # if there are people left at the beginning of the bridge
        if left_side:
            # pick randomly person who will return 
            go_back = random.choice(right_side)
            right_side.remove(go_back)
            left_side.append(go_back)
            run_cost += go_back
    

    if run_cost < best_cost:
        best_cost = run_cost
        best_sequence = run_sequence

print(best_cost)
print(best_sequence)

# %%
