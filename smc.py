import numpy as np
import matplotlib.pyplot as plt

# genaerating data in gaussian
true_mean = 5.0
data = np.random.normal(true_mean, 1.0, size=500)


# how good mu explains the data
def likelihood(mu, data):
    return np.exp(-0.5 * np.sum((data - mu)**2))


# number of particles
N = 1000

#initialize particles randomly in guassina distribution
particles = np.random.normal(0, 5, N)


weights = np.ones(N) / N  # intially equal weight 


# SMC loop
for t in range(100):

    #  update weights
    for i in range(N):
        weights[i] = likelihood(particles[i], data)

    # normalize 
    weights = weights / np.sum(weights)

    # resample  based on weights
    indices = np.random.choice(N, size=N, p=weights) 
    particles = particles[indices]

    # move particles a bit (exploration step)
    particles = particles + np.random.normal(0, 0.5, N)


print("Final estimate:", np.mean(particles))



plt.hist(particles, bins=50, density=True)
plt.axvline(true_mean, linestyle='--', color= 'g',label='True Mean')
plt.axvline(np.mean(particles), linestyle='--', color= 'r',label='Estimated Mean')
plt.title("SMC Particle Distribution")
plt.legend()
plt.show()
