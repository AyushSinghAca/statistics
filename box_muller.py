import numpy as np
import matplotlib.pyplot as plt

mu= 0    # Mean
std = 1  # standard Deviation
N= 1000000

data= []  # store value from  Muller Method

# Gaussian Using Numpy Function
data_np= np.random.normal(mu, std, N)

# Using Box - Muller Method
for _ in range(N // 2):
	u1= np.random.rand()  
	u2= np.random.rand()

	z0= np.sqrt((-2* np.log(u1)))* np.cos(2*np.pi*u2)
	z1= np.sqrt((-2* np.log(u1)))* np.sin(2*np.pi*u2)

	data.append(mu + std*z0)
	data.append(mu + std*z1)

# Plotting
plt.hist(data, bins=100, density= True, alpha=1, label= "Box-Muller")
plt.hist(data_np, bins=100, density= True, alpha=0.7, label= "Numpy_Gaussian")
plt.title("Gaussian Distribution Comparsion")
plt.legend()
plt.savefig("box_muller.png", dpi=400)
plt.show()
