import random
import math

def fitness(x):
    return x * math.sin(10 * math.pi * x) + 1.0

def create_individual():
    return random.uniform(0, 1)

def crossover(p1, p2):
    point = random.random()
    return point * p1 + (1 - point) * p2

def mutate(x, rate=0.1):
    if random.random() < rate:
        return x + random.uniform(-0.1, 0.1)
    return x

def genetic_algorithm(pop_size=20, generations=50, mutation_rate=0.1):
    population = [create_individual() for _ in range(pop_size)]
    best = None

    for g in range(generations):
        scored = [(ind, fitness(ind)) for ind in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        if best is None or scored[0][1] > best[1]:
            best = scored[0]
        selected = [ind for ind, _ in scored[:pop_size // 2]]
        children = []
        while len(children) < pop_size:
            p1, p2 = random.sample(selected, 2)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            child = max(0, min(1, child))
            children.append(child)
        population = children

    return best

if __name__ == "__main__":
    best_solution = genetic_algorithm()
    print("Best Solution: x =", best_solution[0], "f(x) =", best_solution[1])
