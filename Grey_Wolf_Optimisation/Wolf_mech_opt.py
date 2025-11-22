import numpy as np

# ===============================================
# Process Model (First Order System)
# Transfer function:  G(s) = 1 / (Ts + 1)
# ===============================================

def simulate_pid(Kp, Ki, Kd, T=1.0, dt=0.01, sim_time=10):
    """Simulates a first-order system with PID controller."""
    n = int(sim_time / dt)
    error_sum = 0
    previous_error = 0
    y = 0  # system output

    ISE = 0  # Integral of Squared Error

    for t in range(n):
        setpoint = 1  # desired output
        error = setpoint - y
        error_sum += error * dt

        derivative = (error - previous_error) / dt
        previous_error = error

        # PID output
        u = Kp * error + Ki * error_sum + Kd * derivative

        # system update: dy/dt = (-y + u) / T
        y += dt * ((-y + u) / T)

        ISE += error ** 2

    return ISE


# ===============================================
# Grey Wolf Optimizer for PID tuning
# ===============================================

def GWO_PID(wolves, iterations, lb, ub):
    dim = 3  # Kp, Ki, Kd
    positions = np.random.uniform(lb, ub, (wolves, dim))

    alpha = np.zeros(dim)
    alpha_score = float("inf")

    beta = np.zeros(dim)
    beta_score = float("inf")

    delta = np.zeros(dim)
    delta_score = float("inf")

    for t in range(iterations):
        for i in range(wolves):
            Kp, Ki, Kd = positions[i]
            fitness = simulate_pid(Kp, Ki, Kd)

            if fitness < alpha_score:
                delta_score, delta = beta_score, beta.copy()
                beta_score, beta = alpha_score, alpha.copy()
                alpha_score, alpha = fitness, positions[i].copy()

            elif fitness < beta_score:
                delta_score, delta = beta_score, beta.copy()
                beta_score, beta = fitness, positions[i].copy()

            elif fitness < delta_score:
                delta_score, delta = fitness, positions[i].copy()

        # Update positions
        a = 2 - t * (2 / iterations)

        for i in range(wolves):
            for j in range(dim):
                r1, r2 = np.random.random(), np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha[j] - positions[i][j])
                X1 = alpha[j] - A1 * D_alpha

                r1, r2 = np.random.random(), np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta[j] - positions[i][j])
                X2 = beta[j] - A2 * D_beta

                r1, r2 = np.random.random(), np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta[j] - positions[i][j])
                X3 = delta[j] - A3 * D_delta

                positions[i][j] = (X1 + X2 + X3) / 3

        print(f"Iteration {t+1}/{iterations} → Best ISE: {alpha_score:.4f}")

    return alpha, alpha_score


# ===============================================
# USER INPUT
# ===============================================
print("GWO for PID Controller Tuning")
wolves = int(input("Enter number of wolves (e.g., 15): "))
iterations = int(input("Enter iterations (e.g., 30): "))

lb = float(input("Enter lower bound for PID gains (e.g., 0): "))
ub = float(input("Enter upper bound for PID gains (e.g., 10): "))

best_pid, best_score = GWO_PID(wolves, iterations, lb, ub)

print("\n============================")
print("    OPTIMAL PID GAINS")
print("============================")
print(f"Kp = {best_pid[0]:.4f}")
print(f"Ki = {best_pid[1]:.4f}")
print(f"Kd = {best_pid[2]:.4f}")
print("\nMinimum ISE:", best_score)
