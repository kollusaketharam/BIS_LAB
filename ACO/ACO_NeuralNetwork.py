import random

# ------------------ COST FUNCTION ------------------
def compute_cost(allocation, cost_matrix):
    total = 0
    for w in range(len(allocation)):
        for s in range(len(allocation[0])):
            total += allocation[w][s] * cost_matrix[w][s]
    return total


# ------------------ ACO MAIN FUNCTION ------------------
def ACO_inventory(num_ants, alpha, beta, rho, iterations,
                  warehouses, stores, capacity, demand, cost_matrix):

    # Initial pheromones
    pheromone = [[1 for _ in range(stores)] for _ in range(warehouses)]

    best_cost = float("inf")
    best_alloc = None

    for it in range(iterations):
        iteration_best = float("inf")
        iteration_best_alloc = None

        for ant in range(num_ants):

            # Allocation matrix W x S
            alloc = [[0 for _ in range(stores)] for _ in range(warehouses)]

            remaining_cap = capacity[:]
            remaining_demand = demand[:]

            # Build solution store by store
            for s in range(stores):
                while remaining_demand[s] > 0:

                    probabilities = []
                    for w in range(warehouses):
                        if remaining_cap[w] > 0:
                            tau = pheromone[w][s] ** alpha
                            eta = (1 / cost_matrix[w][s]) ** beta
                            probabilities.append(tau * eta)
                        else:
                            probabilities.append(0)

                    if sum(probabilities) == 0:
                        break  # stuck case

                    # Normalize
                    total_p = sum(probabilities)
                    probabilities = [p / total_p for p in probabilities]

                    # Choose warehouse
                    chosen_w = random.choices(range(warehouses), weights=probabilities)[0]

                    # Allocate 1 unit (simple heuristic)
                    alloc[chosen_w][s] += 1
                    remaining_cap[chosen_w] -= 1
                    remaining_demand[s] -= 1

            cost = compute_cost(alloc, cost_matrix)

            # Update best of iteration
            if cost < iteration_best:
                iteration_best = cost
                iteration_best_alloc = alloc

        # -------- PHEROMONE UPDATE --------
        for w in range(warehouses):
            for s in range(stores):
                pheromone[w][s] *= (1 - rho)  # evaporation

        # Reinforce good solution
        for w in range(warehouses):
            for s in range(stores):
                pheromone[w][s] += 1 / iteration_best_alloc[w][s] if iteration_best_alloc[w][s] > 0 else 0

        # -------- TRACK GLOBAL BEST --------
        if iteration_best < best_cost:
            best_cost = iteration_best
            best_alloc = iteration_best_alloc

        # Print iteration result
        print(f"Iteration {it+1}/{iterations} → Best Cost: {best_cost}")

    return best_alloc, best_cost


# ------------------ USER INPUT ------------------

print("\n=== ACO for Inventory Distribution Optimization ===")

warehouses = int(input("Enter number of warehouses: "))
stores = int(input("Enter number of stores/outlets: "))

capacity = []
print("\nEnter warehouse capacities:")
for w in range(warehouses):
    capacity.append(int(input(f"Capacity of warehouse {w}: ")))

demand = []
print("\nEnter store demands:")
for s in range(stores):
    demand.append(int(input(f"Demand of store {s}: ")))

cost_matrix = []
print("\nEnter transport cost matrix (warehouse to store):")
for w in range(warehouses):
    while True:
        try:
            row_str = input(f"Costs from warehouse {w} to all {stores} stores (enter {stores} values separated by spaces): ")
            row = list(map(float, row_str.split()))
            if len(row) != stores:
                print(f"Please enter exactly {stores} cost values, separated by spaces.")
            else:
                cost_matrix.append(row)
                break
        except ValueError:
            print("Invalid input. Please enter numerical values separated by spaces.")

# ACO parameters
num_ants = int(input("\nEnter number of ants: "))
alpha = float(input("Enter alpha (pheromone weight): "))
beta = float(input("Enter beta (cost/heuristic weight): "))
rho = float(input("Enter evaporation rate (0-1): "))
iterations = int(input("Enter number of iterations: "))

# ------------------ RUN ACO ------------------
best_alloc, best_cost = ACO_inventory(
    num_ants, alpha, beta, rho, iterations,
    warehouses, stores, capacity, demand, cost_matrix
)

# ------------------ OUTPUT ------------------
print("\n===== OPTIMAL ALLOCATION FOUND =====")
for w in range(warehouses):
    print(f"Warehouse {w}: {best_alloc[w]}")

print(f"\nMinimum Total Cost: {best_cost}")