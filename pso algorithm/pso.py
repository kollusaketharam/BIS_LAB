import random
import math

def fitness(x):
    return x * math.sin(10 * math.pi * x) + 1.0

class Particle:
    def __init__(self):
        self.position = random.uniform(0, 1)
        self.velocity = random.uniform(-0.1, 0.1)
        self.best_position = self.position
        self.best_value = fitness(self.position)

def pso(num_particles=20, iterations=50, w=0.7, c1=1.5, c2=1.5):
    swarm = [Particle() for _ in range(num_particles)]
    gbest_position = max(swarm, key=lambda p: p.best_value).best_position
    gbest_value = fitness(gbest_position)

    for it in range(iterations):
        for particle in swarm:
            value = fitness(particle.position)
            if value > particle.best_value:
                particle.best_value = value
                particle.best_position = particle.position
            if value > gbest_value:
                gbest_value = value
                gbest_position = particle.position

        for particle in swarm:
            r1, r2 = random.random(), random.random()
            cognitive = c1 * r1 * (particle.best_position - particle.position)
            social = c2 * r2 * (gbest_position - particle.position)
            particle.velocity = w * particle.velocity + cognitive + social
            particle.position += particle.velocity
            particle.position = max(0, min(1, particle.position))

        print(f"Iteration {it+1}: Best x = {gbest_position:.4f}, f(x) = {gbest_value:.4f}")

    return gbest_position, gbest_value

if __name__ == "__main__":
    best_x, best_val = pso()
    print("\nFinal Best Solution: x =", best_x, "f(x) =", best_val)
