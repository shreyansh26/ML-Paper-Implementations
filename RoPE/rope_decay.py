import numpy as np
import matplotlib.pyplot as plt

# Parameters
d = 256  # Example dimension
relative_distances = np.arange(1, 260)  # Relative distances to calculate decay over
theta = np.array([10000 ** (-2 * i / d) for i in range(d // 2)])  # Theta values

# Compute the decay
relative_upper_bound = []

for m_n in relative_distances:
    S_i = np.array([np.abs(np.sum(np.exp(1j * m_n * theta[:j]))) for j in range(1, d // 2 + 1)])
    relative_upper_bound.append(np.mean(S_i))

# Plot the result
plt.figure(figsize=(10, 5))
plt.plot(relative_distances, relative_upper_bound, label="Relative upper bound")
plt.xlabel("Relative distance")
plt.ylabel("Relative upper bound")
plt.title("Long-term decay of RoPE")
plt.grid()
plt.show()