import numpy as np
import math as mp
import matplotlib.pyplot as plt

#  Training data

np.random.seed(7)

x = np.arange(1, 60)

noise = np.random.normal(0, 6, size=len(x))  # high noise

y = 2 * x + noise

# force some extra negative values (to make it more chaotic)
y[[3, 10, 18, 25, 33, 41]] -= np.random.uniform(10, 25, size=6)




# prior data

m_v= np.linspace(1,2,100)
b_v= np.linspace(-10,0,100)

sm= 5  # sigma= noise 

# Taking gaussian Likelihood Function

def gaussian(y, mean, sigma):
	return np.exp(-0.5*((y-mean)/sigma)**2) / (sigma*np.sqrt(2*np.pi))

# compute posterior

posterior = []

for m in m_v:
	for  b in b_v:
		likelihood= 1  # Assume

		for xi, yi in zip(x,y):
			y_p= m*xi + b
			likelihood *= gaussian(yi,y_p, sm)

		posterior.append((m,b,likelihood))

# Normalize Posterior

posterior = np.array(posterior)
posterior[:,2] /= np.sum(posterior[:,2])

#  finding best parameter

best_index= np.argmax(posterior[:,2])
best_m, best_b, best_like = posterior[best_index]

print("Best slope (m):", best_m)
print("Best intercept (b):", best_b)
print("Likehood:", best_like)

# Plot

plt.figure(figsize=(8, 5))

plt.scatter(x, y, color='black', label="Data")
x_line = np.linspace(min(x)-2, max(x)+2, 100)
y_line = best_m * x_line + best_b
plt.plot(x_line, y_line, color='red', label="Best Fit")

plt.title("Bayesian Linear Regression (with uncertainty)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()

