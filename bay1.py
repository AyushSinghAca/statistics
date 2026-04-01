import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as b

# Data
H= 8
T=2

# prior
alpha=1
beta= 1

#posterior Parameter
alpha_up= alpha+H
beta_up= beta+T

#estimate bias
theta_alpha = alpha_up/(alpha_up+beta_up)

print("New Estimate", theta_alpha)

# Plot Posterior
x= np.linspace(0,1,100)
y= b.pdf(x,alpha_up,beta_up)

# Plotting
plt.plot(x,y)
plt.title("Posterior Distribution of Coin Bias")
plt.xlabel("Theta (probability of heads)")
plt.ylabel("Density") 
plt.show()

