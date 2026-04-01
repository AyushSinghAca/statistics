import numpy as np
import matplotlib.pyplot as plt

# Data Set
x= np.array([12,35,25,33,54,76,45,76,33,67,86,34,22,66,87,90,33,77,33,99,32,98,78,54,33,25,89,24,65,34])

#  Size of Data set
n=  len(x)

# Means of Value
mean= np.mean(x)

# Standard Deviation
sd= np.sqrt(np.sum((x-mean)**2)/n-1)


print ("For the given List of data we got means and standard deviation as (", int(mean) , int(sd) ,")")
