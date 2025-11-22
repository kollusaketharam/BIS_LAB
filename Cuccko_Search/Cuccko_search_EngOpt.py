import numpy as np
import random


# ------------------- Spring Design Objective Function -------------------
def spring_weight(x):
    x1, x2, x3 = x
    return (x1 ** 2) * x2 * (x3 + 2)


# ------------------- Lévy Flight -------------------
def levy_flight(Lambda):
    u = np.random.normal(0, 1)
    v = np.random.normal(0, 1)
    step = u / (abs(v) ** (1 / Lambda))
    return step


# ------------------- Cuckoo Search Algorithm -------------------
def cuckoo_search(n_nests, pa, iterations, lower, upper):

    dim = 3  # x1, x2, x3 (spring design variables)

    nests = np.random.uniform(lower, upper, (n_nests, dim))
    fitness = np.array([spring_weight(n) for n in nests])

    best_index = np.argmin(fitness)
    best_nest = nests[best_index].copy()
    best_fitness = fitness[best_index]

    for it in range(iterations):

        # Generate new solutions via Lévy flights
        new_nests = np.copy(nests)
        for i in range(n_nests):
            step = levy_flight(1.5) * (nests[i] - best_nest)
            new_nests[i] = nests[i] + step * np.random.normal(size=dim)

            # Keep variables within bounds
            new_nests[i] = np.clip(new_nests[i], lower, upper)

        new_fitness = np.array([spring_weight(n) for n in new_nests])

        # Replace if improved
        for i in range(n_nests):
            if new_fitness[i] < fitness[i]:
                nests[i], fitness[i] = new_nests[i], new_fitness[i]

        # Abandon worst nests
        abandon = int(pa * n_nests)
        worst_indices = np.argsort(fitness)[-abandon:]

        for i in worst_indices:
            nests[i] = np.random.uniform(lower, upper, dim)
            fitness[i] = spring_weight(nests[i])

        # Update global best
        best_index = np.argmin(fitness)
        if fitness[best_index] < best_fitness:
            best_nest = nests[best_index].copy()
            best_fitness = fitness[best_index]

        print(f"Iteration {it+1}/{iterations} | Best Weight: {best_fitness:.4f}")

    return best_nest, best_fitness


# ------------------- USER INPUT -------------------
print("\n=== Compression Spring Design Optimization using Cuckoo Search ===")

n_nests = int(input("Enter number of nests: "))
pa = float(input("Enter probability of abandonment (0–1): "))
iterations = int(input("Enter number of iterations: "))

lower_bound = float(input("Enter lower bound for variables: "))
upper_bound = float(input("Enter upper bound for variables: "))

# ------------------- RUN CS -------------------
best_x, best_val = cuckoo_search(n_nests, pa, iterations, lower_bound, upper_bound)

# ------------------- OUTPUT -------------------
print("\n========== OPTIMAL SPRING DESIGN ==========")
print(f"x1 (wire diameter): {best_x[0]:.4f}")
print(f"x2 (coil diameter): {best_x[1]:.4f}")
print(f"x3 (# active coils): {best_x[2]:.4f}")
print(f"\nMinimum Spring Weight = {best_val:.4f}")
